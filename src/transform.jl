# Invertible online transforms for conjugation. Port of skaters/transform.py.
# A transform is a 2-tuple (forward, inverse_k):
#   forward(y, state)      -> (y', state)
#   inverse_k(dists, state) -> dists
# Buffers live inside the (mutable) state Dict, exactly as in Python.

# --- differencing ---

function difference()
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("last" => y)
        end
        dy = y - state["last"]
        return dy, Dict{String,Any}("last" => y)
    end
    function inverse_k(dists::Vector, state::Dict)
        anchor = state["last"]
        result = Any[]
        cumsum_mean = 0.0
        cumsum_var = 0.0
        for d in dists
            cumsum_mean += dist_mean(d)
            cumsum_var += dist_var(d)
            std = cumsum_var > 0 ? sqrt(cumsum_var) : max(dist_std(d), 1e-12)
            push!(result, dist_gaussian(anchor + cumsum_mean, std))
        end
        return result
    end
    return (forward, inverse_k)
end

# --- fractional differencing ---

function _frac_diff_weights(d::Float64, window::Int)
    w = Float64[1.0]
    for i in 1:(window-1)
        push!(w, -w[end] * (d - i + 1) / i)
    end
    return w
end

function fractional_difference(d::Float64 = 0.4, window::Int = 50)
    w_fwd = _frac_diff_weights(d, window)
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("buffer" => Float64[])
        end
        buf = state["buffer"]::Vector{Float64}
        push!(buf, y)
        if length(buf) > window
            popfirst!(buf)
        end
        n = length(buf)
        y_prime = fsum(w_fwd[j+1] * buf[n-j] for j in 0:(n-1))
        return y_prime, state
    end
    function inverse_k(dists::Vector, state::Dict)
        buf = copy(state["buffer"]::Vector{Float64})
        result = Any[]
        for d_in in dists
            push!(buf, 0.0)
            n = length(buf)
            shift = 0.0
            for j in 1:(min(n, window)-1)
                shift -= w_fwd[j+1] * buf[n-j]
            end
            recovered_mean = dist_mean(d_in) + shift
            buf[end] = recovered_mean
            push!(result, dist_gaussian(recovered_mean, dist_std(d_in)))
            if length(buf) > window
                popfirst!(buf)
            end
        end
        return result
    end
    return (forward, inverse_k)
end

# --- running standardization ---

function standardize(alpha::Float64 = 0.05, eps::Float64 = 1e-8)
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("mu" => y, "var" => 0.0)
        end
        mu = state["mu"]
        var = state["var"]
        diff = y - mu
        # Emit against the PRIOR state, so the forward map is the affine change
        # of coordinates z = (y - mu) / sigma that the inverse applies. Scaling
        # by the post-update std makes the emission self-normalized (bounded by
        # 1/sqrt(alpha)) and the affine inverse is then not its inverse.
        # Cold start: before the variance is informative, scale by the first
        # nonzero residual, so the first informative emission is +-1.
        sigma = if var > eps * eps
            sqrt(var)
        elseif abs(diff) > eps
            abs(diff)
        else
            eps
        end
        y_prime = diff / sigma
        mu_new = mu + alpha * diff
        var_new = (1 - alpha) * var + alpha * diff * diff
        return y_prime, Dict{String,Any}("mu" => mu_new, "var" => var_new)
    end
    function inverse_k(dists::Vector, state::Dict)
        mu = state["mu"]
        var = state["var"]
        sigma = var > 1e-16 ? sqrt(var) : 1e-8
        return Any[dist_affine(d, sigma, mu) for d in dists]
    end
    return (forward, inverse_k)
end

# --- EMA as a transform ---

function ema_transform(alpha::Float64 = 0.05)
    @assert 0 < alpha < 1
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("level" => y)
        end
        residual = y - state["level"]
        level = state["level"] + alpha * residual
        return residual, Dict{String,Any}("level" => level)
    end
    function inverse_k(dists::Vector, state::Dict)
        level = state["level"]
        return Any[dist_shift(d, level) for d in dists]
    end
    return (forward, inverse_k)
end

# --- Ornstein-Uhlenbeck mean reversion ---

function ou_transform(kappa::Float64 = 0.1, alpha::Float64 = 0.02)
    @assert 0.0 < kappa <= 1.0
    @assert 0.0 < alpha < 1.0
    phi = 1.0 - kappa
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing || !isfinite(y)
            y0 = isfinite(y) ? y : 0.0
            return 0.0, Dict{String,Any}("m" => y0, "fc" => y0, "y" => y0)
        end
        resid = y - state["fc"]
        if !isfinite(resid)
            resid = 0.0
        end
        m = state["m"] + alpha * (y - state["m"])
        fc = m + phi * (y - m)
        return resid, Dict{String,Any}("m" => m, "fc" => fc, "y" => y)
    end
    function inverse_k(dists::Vector, state::Dict)
        m = state["m"]; ylast = state["y"]
        out = Any[]
        for (hh, d) in enumerate(dists)
            h = hh  # 1-based horizon matches Python's start=1
            center = m + (phi^h) * (ylast - m)
            if phi < 1.0 - 1e-9
                g = sqrt((1.0 - phi^(2 * h)) / (1.0 - phi * phi))
            else
                g = sqrt(h)
            end
            push!(out, dist_shift(dist_scale(d, g), center))
        end
        return out
    end
    return (forward, inverse_k)
end

# --- Theta method ---

function theta(alpha::Float64 = 0.1)
    @assert 0 < alpha < 1
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("ses" => y, "t" => 1, "sum_t" => 1.0,
                "sum_t2" => 1.0, "sum_y" => y, "sum_ty" => y)
        end
        s = state
        s["t"] += 1
        t = s["t"]
        forecast = s["ses"] + get(s, "slope", 0.0) / 2
        residual = y - forecast
        s["ses"] = alpha * y + (1 - alpha) * s["ses"]
        s["sum_t"] += t
        s["sum_t2"] += t * t
        s["sum_y"] += y
        s["sum_ty"] += t * y
        n = t
        denom = n * s["sum_t2"] - s["sum_t"]^2
        slope = abs(denom) > 1e-12 ? (n * s["sum_ty"] - s["sum_t"] * s["sum_y"]) / denom : 0.0
        s["slope"] = slope
        return residual, s
    end
    function inverse_k(dists::Vector, state::Dict)
        ses = state["ses"]
        slope = get(state, "slope", 0.0)
        result = Any[]
        cumsum_var = 0.0
        for (hh, d) in enumerate(dists)
            h = hh - 1
            cumsum_var += dist_var(d)
            forecast = ses + (h + 1) * slope / 2 + dist_mean(d)
            std = cumsum_var > 0 ? sqrt(cumsum_var) : max(dist_std(d), 1e-12)
            push!(result, dist_gaussian(forecast, std))
        end
        return result
    end
    return (forward, inverse_k)
end

# --- random walk with drift ---

function drift(alpha::Float64 = 0.002, shrinkage::Float64 = 0.001)
    @assert 0 < alpha < 1
    @assert 0 <= shrinkage < 1
    decay = 1 - alpha - shrinkage
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("last" => y, "mu" => 0.0)
        end
        dy = y - state["last"]
        residual = dy - state["mu"]
        mu = decay * state["mu"] + alpha * dy
        return residual, Dict{String,Any}("last" => y, "mu" => mu)
    end
    function inverse_k(dists::Vector, state::Dict)
        anchor = state["last"]
        mu = state["mu"]
        result = Any[]
        cumsum_mean = 0.0
        cumsum_var = 0.0
        for (hh, d) in enumerate(dists)
            h = hh - 1
            cumsum_mean += dist_mean(d)
            cumsum_var += dist_var(d)
            total_mean = anchor + (h + 1) * mu + cumsum_mean
            total_std = cumsum_var > 0 ? sqrt(cumsum_var) : max(dist_std(d), 1e-12)
            push!(result, dist_gaussian(total_mean, total_std))
        end
        return result
    end
    return (forward, inverse_k)
end

# --- Holt linear ---

function holt_linear(alpha::Float64 = 0.1, beta::Float64 = 0.05)
    @assert 0 < alpha < 1
    @assert 0 < beta < 1
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("level" => y, "trend" => 0.0)
        end
        l_prev = state["level"]
        b_prev = state["trend"]
        l_new = alpha * y + (1 - alpha) * (l_prev + b_prev)
        b_new = beta * (l_new - l_prev) + (1 - beta) * b_prev
        residual = y - (l_prev + b_prev)
        return residual, Dict{String,Any}("level" => l_new, "trend" => b_new)
    end
    function inverse_k(dists::Vector, state::Dict)
        level = state["level"]
        trend = state["trend"]
        result = Any[]
        cumsum_var = 0.0
        for (hh, d) in enumerate(dists)
            h = hh - 1
            cumsum_var += dist_var(d)
            forecast = level + (h + 1) * trend + dist_mean(d)
            std = cumsum_var > 0 ? sqrt(cumsum_var) : max(dist_std(d), 1e-12)
            push!(result, dist_gaussian(forecast, std))
        end
        return result
    end
    return (forward, inverse_k)
end

# --- GARCH(1,1) volatility transform ---

function garch(; omega::Float64 = 0.01, alpha::Float64 = 0.1, beta::Float64 = 0.85,
               mean_alpha::Float64 = 0.05, eps::Float64 = 1e-8)
    @assert omega > 0
    @assert alpha >= 0
    @assert beta >= 0
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            persist = alpha + beta
            var0 = persist < 1 ? omega / (1 - persist) : omega / eps
            return 0.0, Dict{String,Any}("var" => var0, "last_dev" => 0.0, "mu" => y)
        end
        # GARCH conditional variance of the DEVIATION from a running mean, not of
        # the raw value. alpha * y^2 treats y as a mean-zero return; on a level
        # series that makes the "volatility" of order |y|, and the inverse then
        # re-inflates it -- unstable, and it widened real level-series forecasts.
        # Tracking a mean (like standardize) makes garch shift-invariant. On the
        # mean-zero returns garch is meant for, mu stays ~0 and this reduces to
        # the original y/sigma exactly.
        mu = state["mu"]
        dev = y - mu
        var = omega + alpha * state["last_dev"]^2 + beta * state["var"]
        sigma = var > eps * eps ? sqrt(var) : eps
        y_prime = dev / sigma
        mu_new = mu + mean_alpha * dev
        return y_prime, Dict{String,Any}("var" => var, "last_dev" => dev, "mu" => mu_new)
    end
    function inverse_k(dists::Vector, state::Dict)
        sigma = state["var"] > 1e-16 ? sqrt(state["var"]) : 1e-8
        return Any[dist_affine(d, sigma, state["mu"]) for d in dists]
    end
    return (forward, inverse_k)
end

# --- seasonal differencing ---

function seasonal_difference(period::Int = 12)
    @assert period >= 1
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("buffer" => Float64[y])
        end
        buf = state["buffer"]::Vector{Float64}
        L = length(buf)
        if L >= period
            y_prime = y - buf[L-period+1]
        else
            y_prime = 0.0
        end
        push!(buf, y)
        if length(buf) > 2 * period
            popfirst!(buf)
        end
        return y_prime, state
    end
    function inverse_k(dists::Vector, state::Dict)
        buf = copy(state["buffer"]::Vector{Float64})
        recovered_means = Float64[]
        recovered_vars = Float64[]
        result = Any[]
        for h in 0:(length(dists)-1)
            lag_idx = h - period
            if lag_idx < 0
                buf_idx = length(buf) - period + h  # 0-based
                anchor_mean = (0 <= buf_idx < length(buf)) ? buf[buf_idx+1] : 0.0
                anchor_var = 0.0
            else
                anchor_mean = recovered_means[lag_idx+1]
                anchor_var = recovered_vars[lag_idx+1]
            end
            d = dists[h+1]
            push!(recovered_means, dist_mean(d) + anchor_mean)
            push!(recovered_vars, dist_var(d) + anchor_var)
            if anchor_var > 0.0
                push!(result, Dist([(w, m + anchor_mean, sqrt(s * s + anchor_var))
                                    for (w, m, s) in d.comps]))
            else
                push!(result, dist_shift(d, anchor_mean))
            end
        end
        return result
    end
    return (forward, inverse_k)
end

# --- hedged seasonal anchor ---

# Residual from a hedged seasonal anchor.
#
# Forward:   y'_t = y_t - a_t,  a_t = weight * phaseEMA_{p(t)} + (1-weight) * y_{t-s}
# Inverse:   shift each horizon's Dist by its anchor; for h >= s the naive
#            component is a value recovered earlier in the same call and its
#            variance is convolved in (as in seasonal_difference).
#
# The phase-EMA is a recency-weighted mean of same-phase values (memory ~1/alpha
# cycles); the seasonal-naive is the single value one period ago. The naive alone
# adapts instantly but is one noisy draw; the phase-EMA averages the noise but
# lags level shifts. The blend beats either alone on seasonal series.
# weight = 0 recovers seasonal_difference.
#
# Forecasting from same-phase component series follows Viole's NNS package
# (NNS.ARMA, CRAN, since 2017).
function seasonal_anchor(period::Int; alpha::Float64 = 0.2, weight::Float64 = 0.5)
    @assert period >= 1
    @assert 0 < alpha < 1
    @assert 0.0 <= weight <= 1.0
    _anchor(ema_p, snaive) =
        ema_p === nothing ? snaive : weight * ema_p + (1.0 - weight) * snaive
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("ema" => Vector{Union{Nothing,Float64}}(nothing, period),
                                         "buffer" => Float64[y], "n" => 1)
        end
        buf = state["buffer"]::Vector{Float64}
        ema = state["ema"]::Vector{Union{Nothing,Float64}}
        p = (state["n"] % period)          # 0-based phase, as in Python
        snaive = length(buf) >= period ? buf[length(buf)-period+1] : buf[end]
        y_prime = y - _anchor(ema[p+1], snaive)
        e = ema[p+1]
        ema[p+1] = e === nothing ? y : e + alpha * (y - e)
        push!(buf, y)
        if length(buf) > 2 * period
            popfirst!(buf)
        end
        state["n"] = state["n"] + 1
        return y_prime, state
    end
    function inverse_k(dists::Vector, state::Dict)
        buf = copy(state["buffer"]::Vector{Float64})
        ema = state["ema"]::Vector{Union{Nothing,Float64}}
        n = state["n"]
        recovered_means = Float64[]
        recovered_vars = Float64[]
        result = Any[]
        for h in 0:(length(dists)-1)
            p = ((n + h) % period)
            lag_idx = h - period
            if lag_idx < 0
                buf_idx = length(buf) - period + h   # 0-based
                snaive = (0 <= buf_idx < length(buf)) ? buf[buf_idx+1] : buf[end]
                snaive_var = 0.0
            else
                snaive = recovered_means[lag_idx+1]
                snaive_var = recovered_vars[lag_idx+1]
            end
            a_mean = _anchor(ema[p+1], snaive)
            a_var = ((1.0 - weight)^2) * snaive_var
            d = dists[h+1]
            push!(recovered_means, dist_mean(d) + a_mean)
            push!(recovered_vars, dist_var(d) + a_var)
            if a_var > 0.0
                push!(result, Dist([(w, m + a_mean, sqrt(s * s + a_var))
                                    for (w, m, s) in d.comps]))
            else
                push!(result, dist_shift(d, a_mean))
            end
        end
        return result
    end
    return (forward, inverse_k)
end

# --- signed power transform ---

function power_transform(p::Float64 = 0.5)
    @assert 0 < p < 1
    inv_p = 1.0 / p
    _fwd(y) = copysign(abs(y)^p, y)
    _inv(yp) = copysign(abs(yp)^inv_p, yp)
    function forward(y::Float64, state::Union{Nothing,Dict})
        return _fwd(y), (state === nothing ? Dict{String,Any}() : state)
    end
    function inverse_k(dists::Vector, state::Dict)
        result = Any[]
        for d in dists
            comps = NTuple{3,Float64}[]
            for (w, mu, sigma) in d.comps
                orig_mean = _inv(mu)
                abs_mu = abs(mu)
                deriv = abs_mu > 1e-12 ? inv_p * abs_mu^(inv_p - 1) : inv_p
                orig_std = max(sigma * deriv, 1e-12)
                push!(comps, (w, orig_mean, orig_std))
            end
            push!(result, Dist(comps))
        end
        return result
    end
    return (forward, inverse_k)
end

# --- Yeo-Johnson coordinate transform ---

function yeo_johnson(lmbda::Float64 = 0.0)
    L = Float64(lmbda)
    function _fwd(y)
        if y >= 0.0
            return L == 0.0 ? log1p(y) : ((y + 1.0)^L - 1.0) / L
        end
        if L == 2.0
            return -log1p(-y)
        end
        return -(((-y + 1.0)^(2.0 - L) - 1.0) / (2.0 - L))
    end
    function _inv(yp)
        if yp >= 0.0
            if L == 0.0
                return expm1(min(yp, 350.0))
            end
            base = max(L * yp + 1.0, 1e-12)
            return base^(1.0 / L) - 1.0
        end
        if L == 2.0
            return 1.0 - exp(min(-yp, 350.0))
        end
        base = max(-(2.0 - L) * yp + 1.0, 1e-12)
        return 1.0 - base^(1.0 / (2.0 - L))
    end
    function _dinv(yp)
        if yp >= 0.0
            if L == 0.0
                return exp(min(yp, 350.0))
            end
            base = max(L * yp + 1.0, 1e-12)
            return base^(1.0 / L - 1.0)
        end
        if L == 2.0
            return exp(min(-yp, 350.0))
        end
        base = max(-(2.0 - L) * yp + 1.0, 1e-12)
        return base^(1.0 / (2.0 - L) - 1.0)
    end
    function forward(y::Float64, state::Union{Nothing,Dict})
        return _fwd(y), (state === nothing ? Dict{String,Any}() : state)
    end
    function inverse_k(dists::Vector, state::Dict)
        result = Any[]
        for d in dists
            comps = NTuple{3,Float64}[(w, _inv(mu), max(sigma * _dinv(mu), 1e-12))
                                      for (w, mu, sigma) in d.comps]
            push!(result, Dist(comps))
        end
        return result
    end
    return (forward, inverse_k)
end

# --- small matrix helpers (flat, 0-based Python formulas + 1 offset) ---

function _eye(n::Int, scale::Float64 = 1.0)
    m = zeros(Float64, n * n)
    for i in 0:(n-1)
        m[i*n+i+1] = scale
    end
    return m
end

function _mat_vec(M::Vector{Float64}, v::Vector{Float64}, n::Int)
    result = zeros(Float64, n)
    for i in 0:(n-1)
        s = 0.0
        for j in 0:(n-1)
            s += M[i*n+j+1] * v[j+1]
        end
        result[i+1] = s
    end
    return result
end

_dot(a::Vector{Float64}, b::Vector{Float64}, n::Int) = fsum(a[i+1] * b[i+1] for i in 0:(n-1))

# --- AR(p) with recursive least squares ---

# Spectral radius of the AR companion matrix (largest |characteristic root|).
# The h-step AR forecast is governed by powers of this matrix: it converges iff
# the radius is < 1 and diverges geometrically otherwise. Closed form for p <= 2;
# power iteration above. The 60 fixed iterations and the zero-guard are
# load-bearing for 1e-6 parity with transform.py::_ar_spectral_radius.
function _ar_spectral_radius(phi::Vector{Float64})
    p = length(phi)
    if p == 1
        return abs(phi[1])
    end
    if p == 2
        a, b = phi[1], phi[2]
        disc = a * a + 4.0 * b
        if disc >= 0.0
            r = sqrt(disc)
            return max(abs((a + r) / 2.0), abs((a - r) / 2.0))
        end
        return hypot(a / 2.0, sqrt(-disc) / 2.0)   # |complex root|
    end
    v = ones(Float64, p)
    rho = 0.0
    for _ in 1:60
        nv = vcat(sum(phi[j] * v[j] for j in 1:p), v[1:(p-1)])
        m = maximum(abs.(nv))
        if m == 0
            m = 1.0
        end
        v = nv ./ m
        rho = m
    end
    return rho
end

# Damp AR coefficients into the stationary region for forecasting. An online fit
# from a handful of warm-up points can land on an explosive root whose multi-step
# forecast diverges. Scaling phi_j by gamma^(j+1) scales every companion
# eigenvalue by gamma, so gamma = margin / rho brings the radius to margin and
# leaves an already-stationary fit untouched.
function _ar_stationary(phi::Vector{Float64}, margin::Float64 = 0.999)
    rho = _ar_spectral_radius(phi)
    if rho <= margin
        return phi
    end
    g = margin / rho
    return [phi[j] * g^j for j in 1:length(phi)]
end

function ar(order::Int = 2; lam::Float64 = 0.99, ridge::Float64 = 1.0, decay::Float64 = 0.0)
    @assert order >= 1
    @assert 0 < lam <= 1
    @assert decay >= 0
    p = order
    function _init_P()
        P = zeros(Float64, p * p)
        for j in 0:(p-1)
            P[j*p+j+1] = decay > 0 ? ridge / ((j + 1)^decay) : ridge
        end
        return P
    end
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("buffer" => Float64[], "phi" => zeros(Float64, p),
                "P" => _init_P(), "n" => 0)
        end
        buf = state["buffer"]::Vector{Float64}
        phi = state["phi"]::Vector{Float64}
        state["n"] += 1
        if length(buf) >= p
            x = Float64[buf[length(buf)-i] for i in 0:(p-1)]
            prediction = fsum(phi[i+1] * x[i+1] for i in 0:(p-1))
            residual = y - prediction
            P = state["P"]::Vector{Float64}
            Px = _mat_vec(P, x, p)
            denom = lam + _dot(x, Px, p)
            if abs(denom) > 1e-15
                K = [px / denom for px in Px]
                for i in 0:(p-1)
                    phi[i+1] += K[i+1] * residual
                end
                for i in 0:(p-1)
                    for j in 0:(p-1)
                        P[i*p+j+1] = (P[i*p+j+1] - K[i+1] * Px[j+1]) / lam
                    end
                end
                if !all(isfinite, P) || maximum(abs, P) > 1e10
                    copyto!(P, _init_P())
                end
            end
        else
            residual = y
        end
        push!(buf, y)
        if length(buf) > 2 * p + 10
            popfirst!(buf)
        end
        return residual, state
    end
    function inverse_k(dists::Vector, state::Dict)
        buf = copy(state["buffer"]::Vector{Float64})
        # Forecast with stationarity-constrained coefficients: a non-stationary
        # online fit has no convergent multi-step forecast. The one-step fit in
        # state is left intact; only the extrapolation is kept well-posed.
        phi = _ar_stationary(state["phi"]::Vector{Float64})
        H = length(dists)
        # Impulse responses psi_0..psi_{H-1}: psi_0 = 1, psi_i = sum_j phi_j psi_{i-j}.
        # The h-step variance is sigma^2 * sum_{i<=h} psi_i^2, NOT a recursion on
        # recovered variances -- successive horizons share one innovation, so
        # summing phi_j^2 * var_{h-j} both mis-weights and double counts it.
        psi = Float64[1.0]
        for i in 1:(H-1)
            s = 0.0
            for j in 0:(p-1)
                if i - 1 - j >= 0
                    s += phi[j+1] * psi[i-j]
                end
            end
            push!(psi, s)
        end
        recovered_means = Float64[]
        result = Any[]
        cum_psi2 = 0.0
        for h in 0:(H-1)
            ar_mean = 0.0
            for j in 0:(p-1)
                lag_h = h - j - 1
                if lag_h < 0
                    buf_idx = length(buf) + lag_h  # 0-based
                    if 0 <= buf_idx < length(buf)
                        ar_mean += phi[j+1] * buf[buf_idx+1]
                    end
                else
                    if lag_h < length(recovered_means)
                        ar_mean += phi[j+1] * recovered_means[lag_h+1]
                    end
                end
            end
            d = dists[h+1]
            total_mean = dist_mean(d) + ar_mean
            cum_psi2 += psi[h+1] * psi[h+1]
            total_var = cum_psi2 * dist_var(d)
            total_std = total_var > 0 ? sqrt(total_var) : max(dist_std(d), 1e-12)
            push!(recovered_means, total_mean)
            push!(result, dist_gaussian(total_mean, total_std))
        end
        return result
    end
    return (forward, inverse_k)
end

# --- grouped AR ---

function _build_groups(max_lag::Int)
    groups = Int[]
    g = 0
    size = 1
    assigned = 0
    while assigned < max_lag
        for _ in 1:size
            if assigned >= max_lag
                break
            end
            push!(groups, g)
            assigned += 1
        end
        g += 1
        size *= 2
    end
    return groups
end

function _group_regressor(buf::Vector{Float64}, groups::Vector{Int}, n_groups::Int, max_lag::Int)
    x = zeros(Float64, n_groups)
    for j in 0:(max_lag-1)
        x[groups[j+1]+1] += buf[length(buf)-j]
    end
    return x
end

function grouped_ar(max_lag::Int = 16; lam::Float64 = 0.99, ridge::Float64 = 1.0)
    @assert max_lag >= 1
    @assert 0 < lam <= 1
    @assert ridge > 0
    groups = _build_groups(max_lag)
    n_groups = maximum(groups) + 1
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("buffer" => Float64[], "theta" => zeros(Float64, n_groups),
                "P" => _eye(n_groups, ridge), "n" => 0)
        end
        buf = state["buffer"]::Vector{Float64}
        theta = state["theta"]::Vector{Float64}
        state["n"] += 1
        if length(buf) >= max_lag
            x = _group_regressor(buf, groups, n_groups, max_lag)
            prediction = fsum(theta[g+1] * x[g+1] for g in 0:(n_groups-1))
            residual = y - prediction
            P = state["P"]::Vector{Float64}
            Px = _mat_vec(P, x, n_groups)
            denom = lam + _dot(x, Px, n_groups)
            if abs(denom) > 1e-15
                K = [px / denom for px in Px]
                for g in 0:(n_groups-1)
                    theta[g+1] += K[g+1] * residual
                end
                for i in 0:(n_groups-1)
                    for j in 0:(n_groups-1)
                        P[i*n_groups+j+1] = (P[i*n_groups+j+1] - K[i+1] * Px[j+1]) / lam
                    end
                end
                if !all(isfinite, P) || maximum(abs, P) > 1e10
                    copyto!(P, _eye(n_groups, ridge))
                end
            end
        else
            residual = y
        end
        push!(buf, y)
        if length(buf) > max_lag + 10
            popfirst!(buf)
        end
        return residual, state
    end
    function inverse_k(dists::Vector, state::Dict)
        buf = copy(state["buffer"]::Vector{Float64})
        theta = state["theta"]::Vector{Float64}
        # Expand group coefficients to per-lag, then constrain to stationarity
        # for a well-posed forecast (same as ar(); grouped AR is AR-family).
        phi = _ar_stationary(Float64[theta[groups[j+1]+1] for j in 0:(max_lag-1)])
        H = length(dists)
        # Impulse responses, as in ar(): h-step variance is sigma^2 * sum psi_i^2.
        psi = Float64[1.0]
        for i in 1:(H-1)
            s = 0.0
            for j in 0:(max_lag-1)
                if i - 1 - j >= 0
                    s += phi[j+1] * psi[i-j]
                end
            end
            push!(psi, s)
        end
        recovered_means = Float64[]
        result = Any[]
        cum_psi2 = 0.0
        for h in 0:(H-1)
            ar_mean = 0.0
            for j in 0:(max_lag-1)
                lag_h = h - j - 1
                if lag_h < 0
                    buf_idx = length(buf) + lag_h
                    if 0 <= buf_idx < length(buf)
                        ar_mean += phi[j+1] * buf[buf_idx+1]
                    end
                else
                    if lag_h < length(recovered_means)
                        ar_mean += phi[j+1] * recovered_means[lag_h+1]
                    end
                end
            end
            d = dists[h+1]
            total_mean = dist_mean(d) + ar_mean
            cum_psi2 += psi[h+1] * psi[h+1]
            total_var = cum_psi2 * dist_var(d)
            total_std = total_var > 0 ? sqrt(total_var) : max(dist_std(d), 1e-12)
            push!(recovered_means, total_mean)
            push!(result, dist_gaussian(total_mean, total_std))
        end
        return result
    end
    return (forward, inverse_k)
end
