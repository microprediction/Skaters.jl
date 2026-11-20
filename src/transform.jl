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

# Weights for (1-B)^{-d}, the inverse operator.
_frac_int_weights(d::Float64, window::Int) = _frac_diff_weights(-d, window)

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
        # Recovered earlier horizons feed back through the same weights, so each
        # recovered value carries a row of responses to the future innovations,
        # built by the same recursion as its mean: Var = sum_i c_{h,i}^2 var_i (#255).
        H = length(dists)
        buf = copy(state["buffer"]::Vector{Float64})
        cbuf = Union{Nothing,Vector{Float64}}[nothing for _ in buf]
        result = Any[]
        for (hh, d_in) in enumerate(dists)
            h = hh - 1
            push!(buf, 0.0)
            push!(cbuf, nothing)
            n = length(buf)
            shift = 0.0
            coef = zeros(H)
            coef[h+1] = 1.0
            for j in 1:(min(n, window)-1)
                shift -= w_fwd[j+1] * buf[n-j]
                cj = cbuf[n-j]
                if cj !== nothing
                    for i in 1:h
                        coef[i] -= w_fwd[j+1] * cj[i]
                    end
                end
            end
            recovered_mean = dist_mean(d_in) + shift
            buf[end] = recovered_mean
            cbuf[end] = coef
            total_var = 0.0
            for i in 1:(h+1)
                total_var += coef[i] * coef[i] * dist_var(dists[i])
            end
            std = total_var > 0 ? sqrt(total_var) : max(dist_std(d_in), 1e-12)
            push!(result, dist_gaussian(recovered_mean, std))
            if length(buf) > window
                popfirst!(buf)
                popfirst!(cbuf)
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
        # Convolve the level's own innovation response into each component:
        # Var(y_{t+h+1}) = var_h + alpha^2 sum_{i<h} var_i (ETS(A,N,N); #254).
        level = state["level"]
        out = Any[]
        add = 0.0
        for d in dists
            if add > 0.0
                push!(out, _widen(d, level, add))
            else
                push!(out, dist_shift(d, level))
            end
            add += alpha * alpha * dist_var(d)
        end
        return out
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
            # Var(y_{t+h}) = sum_{i<h} phi^{2(h-1-i)} var_i (#256), as a scale on
            # this horizon's Dist so the leaf's mixture shape is kept.
            v_h = dist_var(d)
            if v_h > 0.0
                tot = 0.0
                for i in 0:(h-1)
                    tot += (phi^(2 * (h - 1 - i))) * dist_var(dists[i+1])
                end
                g = sqrt(tot / v_h)
            elseif phi < 1.0 - 1e-9
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
        for (hh, d) in enumerate(dists)
            h = hh - 1
            forecast = ses + (h + 1) * slope / 2 + dist_mean(d)
            # SES response: c_0 = 1, c_j = alpha (#253); the OLS slope's O(1/t)
            # response is dropped, as in the reference.
            total_var = 0.0
            for j in 0:h
                c = j == 0 ? 1.0 : alpha
                total_var += c * c * dist_var(dists[h-j+1])
            end
            std = total_var > 0 ? sqrt(total_var) : max(dist_std(d), 1e-12)
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
        rho = decay + alpha
        result = Any[]
        cumsum_mean = 0.0
        for (hh, d) in enumerate(dists)
            h = hh - 1
            cumsum_mean += dist_mean(d)
            total_mean = anchor + (h + 1) * mu + cumsum_mean
            # Innovation at t+i+1 reaches y_{t+h+1} directly and through the drift
            # estimate on each later step: c = 1 + alpha * sum_{m<h-i} rho^m (#257).
            total_var = 0.0
            for i in 0:h
                acc = 0.0
                f = 1.0
                for _ in 1:(h-i)
                    acc += f
                    f *= rho
                end
                c = 1.0 + alpha * acc
                total_var += c * c * dist_var(dists[i+1])
            end
            total_std = total_var > 0 ? sqrt(total_var) : max(dist_std(d), 1e-12)
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
        for (hh, d) in enumerate(dists)
            h = hh - 1
            forecast = level + (h + 1) * trend + dist_mean(d)
            # ETS(A,A,N) innovation weights c_0 = 1, c_j = alpha + alpha*beta*j (#242).
            total_var = 0.0
            for j in 0:h
                c = j == 0 ? 1.0 : alpha + alpha * beta * j
                total_var += c * c * dist_var(dists[h-j+1])
            end
            std = total_var > 0 ? sqrt(total_var) : max(dist_std(d), 1e-12)
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
                push!(result, _widen(d, anchor_mean, anchor_var))
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
        H = length(dists)
        recovered_means = Float64[]
        # Innovation-response rows for each recovered value and each phase's EMA:
        # the forward updates a phase's EMA from recovered values too, so an anchor
        # at h >= period responds through both the naive term (1 - weight) and the
        # EMA term (weight * alpha) (skaters#258).
        rec_resp = Vector{Float64}[]
        ema_resp = [zeros(H) for _ in 1:period]
        result = Any[]
        for h in 0:(H-1)
            p = ((n + h) % period)
            lag_idx = h - period
            if lag_idx < 0
                buf_idx = length(buf) - period + h   # 0-based
                snaive = (0 <= buf_idx < length(buf)) ? buf[buf_idx+1] : buf[end]
                sn_resp = nothing
            else
                snaive = recovered_means[lag_idx+1]
                sn_resp = rec_resp[lag_idx+1]
            end
            e = ema[p+1]
            a_mean = _anchor(e, snaive)
            a_resp = zeros(H)
            for i in 1:h
                sn = sn_resp !== nothing ? sn_resp[i] : 0.0
                a_resp[i] = e !== nothing ? (weight * ema_resp[p+1][i] + (1.0 - weight) * sn) : sn
            end
            a_var = 0.0
            for i in 1:h
                a_var += a_resp[i] * a_resp[i] * dist_var(dists[i])
            end
            r = copy(a_resp)
            r[h+1] = 1.0
            d = dists[h+1]
            push!(recovered_means, dist_mean(d) + a_mean)
            push!(rec_resp, r)
            if e === nothing
                ema_resp[p+1] = copy(r)
            else
                ema_resp[p+1] = [(1.0 - alpha) * ema_resp[p+1][i] + alpha * r[i] for i in 1:H]
            end
            if a_var > 0.0
                push!(result, _widen(d, a_mean, a_var))
            else
                push!(result, dist_shift(d, a_mean))
            end
        end
        return result
    end
    return (forward, inverse_k)
end

# --- seasonal scale ---

# Phase-indexed running standardization (seasonal heteroskedasticity).
#
# Forward:   y'_t = (y_t - mu_{p(t)}) / sigma_{p(t)},   p(t) = t mod period
# Inverse:   affine by the stats of the phase being forecast: dists[h] is the
#            (h+1)-step-ahead predictive, so its phase is (n + h) mod period.
#
# Each phase keeps its own EWMA mean and variance. center=false divides by the
# phase sigma without subtracting the phase mean; use it composed over
# seasonal_difference(period). A phase's variance is shrunk toward the global
# EWMA variance in log space by visit count, w = visits / (visits + 3). The
# global stats run at alpha / period.
function seasonal_scale(period::Int; alpha::Float64 = 0.05, center::Bool = false,
                        eps::Float64 = 1e-8)
    @assert period >= 1
    @assert 0 < alpha < 1
    a_g = alpha / period
    K0 = 3.0
    # Bias-corrected global variance (the slow EWMA starts at zero).
    function _vg(state)
        updates = state["n"] - 1
        if updates <= 0
            return 0.0
        end
        return state["var_g"] / (1.0 - (1.0 - a_g)^updates)
    end
    function _phase_stats(state, p)
        vis = state["vis"]::Vector{Int}
        w = vis[p+1] / (vis[p+1] + K0)
        vp = max(state["var"][p+1], eps * eps)
        vg = max(_vg(state), eps * eps)
        var_eff = exp(w * log(vp) + (1.0 - w) * log(vg))
        mu_eff = center ? (w * state["mu"][p+1] + (1.0 - w) * state["mu_g"]) : 0.0
        return mu_eff, sqrt(var_eff)
    end
    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            return 0.0, Dict{String,Any}("mu_g" => y, "var_g" => 0.0,
                                         "mu" => zeros(period), "var" => zeros(period),
                                         "vis" => zeros(Int, period), "n" => 1)
        end
        p = state["n"] % period
        # Center against the prior mean, scale by the updated variance.
        mu_eff, _ = _phase_stats(state, p)
        dg = y - state["mu_g"]
        state["mu_g"] += a_g * dg
        state["var_g"] = (1 - a_g) * state["var_g"] + a_g * dg * dg
        mu = state["mu"]::Vector{Float64}
        var = state["var"]::Vector{Float64}
        vis = state["vis"]::Vector{Int}
        if vis[p+1] == 0                 # first visit seeds from global stats
            mu[p+1] = y
            var[p+1] = _vg(state)
        else
            dp = y - mu[p+1]
            mu[p+1] += alpha * dp
            var[p+1] = (1 - alpha) * var[p+1] + alpha * dp * dp
        end
        vis[p+1] += 1
        state["n"] += 1
        _, sigma = _phase_stats(state, p)
        return (y - mu_eff) / sigma, state
    end
    function inverse_k(dists::Vector, state::Dict)
        out = Any[]
        for h in 0:(length(dists)-1)
            mu_eff, sigma = _phase_stats(state, (state["n"] + h) % period)
            push!(out, dist_affine(dists[h+1], sigma, mu_eff))
        end
        return out
    end
    return (forward, inverse_k)
end

# --- shared helpers for the nonlinear inverses ---

# Shift by delta and add independent Gaussian variance add. Exact for a
# Gaussian mixture (each component widens). A tail-spliced predictive has no
# components to widen, so its body widens and the GPD splice, which lives in
# the body's PIT space, follows it (the SplicedDist method is in tails.jl).
_widen(d, delta::Float64, add::Float64) =
    Dist([(w, m + delta, sqrt(s * s + add)) for (w, m, s) in _need_components(d)])

_has_components(d) = !(d isa PushforwardDist) && _components(d) !== nothing

# The exact pushforward of d through the monotone map inv_fn. The table window
# adapts to each predictive: [m - 10s, m + 10s] over the mixture. With no
# mixture to read (laplace's tail-spliced SplicedDist), the extreme quantiles
# are padded by their own span on each side.
function _pushforward(d, inv_fn)
    if _has_components(d)
        comps = _components(d)
        lo = minimum(m - 10.0 * s for (_, m, s) in comps)
        hi = maximum(m + 10.0 * s for (_, m, s) in comps)
    else
        q_lo = dist_quantile(d, 1e-6)
        q_hi = dist_quantile(d, 1.0 - 1e-6)
        lo, hi = 2.0 * q_lo - q_hi, 2.0 * q_hi - q_lo
    end
    ys, zs = table_from_map(inv_fn, lo, hi)
    return PushforwardDist(d, ys, zs; inv_fn = inv_fn)
end

# An inverse_k that emits the exact pushforward.
_exact_inverse_k(inv_fn) = (dists::Vector, state::Dict) -> Any[_pushforward(d, inv_fn) for d in dists]

# --- signed power transform ---

# Forward y' = sign(y)|y|^p, inverse sign(y')|y'|^(1/p). By default the
# inverse linearizes around each component's mean (delta method); exact=true
# emits the exact pushforward (PushforwardDist).
function power_transform(p::Float64 = 0.5; exact::Bool = false)
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
            dcomps = _components(d)
            if dcomps === nothing   # tail-spliced: exact pushforward
                push!(result, _pushforward(d, _inv))
                continue
            end
            comps = NTuple{3,Float64}[]
            for (w, mu, sigma) in dcomps
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
    return (forward, exact ? _exact_inverse_k(_inv) : inverse_k)
end

# --- Yeo-Johnson coordinate transform ---

# The signed Box-Cox family. By default each component is mapped by the exact
# inverse on the mean and the delta method on the std; exact=true emits the
# exact pushforward (PushforwardDist), which carries the skew of the
# coordinate change at multi-step horizons.
function yeo_johnson(lmbda::Float64 = 0.0; exact::Bool = false)
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
            dcomps = _components(d)
            if dcomps === nothing   # tail-spliced: exact pushforward
                push!(result, _pushforward(d, _inv))
                continue
            end
            comps = NTuple{3,Float64}[(w, _inv(mu), max(sigma * _dinv(mu), 1e-12))
                                      for (w, mu, sigma) in dcomps]
            push!(result, Dist(comps))
        end
        return result
    end
    return (forward, exact ? _exact_inverse_k(_inv) : inverse_k)
end

# --- log coordinate with a one-sided bound ---

# Forward:   y' = log(y - lower)    (lower bound; the default, lower = 0)
#            y' = -log(upper - y)   (upper bound)
# Inverse:   the exact pushforward through the inverse map, so the predictive
#            keeps the skew of the coordinate change, its logpdf includes the
#            Jacobian, and every quantile lies on the right side of the bound.
#
# A finite observation on or past the bound throws ArgumentError. Missing
# observations pass through for the inner skater to handle. Quantiles that
# round to the bound are rounded inward by one floating-point step.
# Experimental, alongside flesaker.
function log_transform(; lower::Union{Nothing,Real} = nothing,
                       upper::Union{Nothing,Real} = nothing)
    if lower !== nothing && upper !== nothing
        throw(ArgumentError("log_transform takes one bound: lower or upper"))
    end
    b = Float64(upper !== nothing ? upper : (lower === nothing ? 0.0 : lower))
    isfinite(b) || throw(ArgumentError("log_transform needs a finite bound"))
    direction = upper !== nothing ? -1.0 : 1.0
    bound_label = upper !== nothing ? "upper" : "lower"
    log_two = log(2.0)

    function _fwd_query(y::Float64)::Float64
        gap = direction * (y - b)
        if gap <= 0.0
            return -direction * Inf
        end
        if isinf(gap)
            # A gap between finite floats can overflow even when its log fits.
            half_gap = direction * (0.5 * y - 0.5 * b)
            return direction * (log(half_gap) + log_two)
        end
        return direction * log(gap)
    end
    function _inverse_at_scale(z::Float64, weight::Float64)::Float64
        bound = weight * b
        log_gap = direction * z + log(weight)
        gap = _exp_or_inf(log_gap)
        if isinf(gap)
            # The shift by the bound can bring an overflowing gap into range.
            return 2.0 * (0.5 * bound + direction * _exp_or_inf(log_gap - log_two))
        end
        return bound + direction * gap
    end
    function _inv(z::Float64)::Float64
        y = _inverse_at_scale(z, 1.0)
        return y == b ? (direction > 0 ? nextfloat(b) : prevfloat(b)) : y
    end
    _log_jacobian(z::Float64) = -direction * z
    function _moments(m::Float64, s::Float64)
        variance = s * s
        log_mean_gap = direction * m + 0.5 * variance
        mean = _inverse_at_scale(direction * log_mean_gap, 1.0)
        if s == 0.0
            return mean, 0.0
        end
        # log(sqrt(expm1(s^2))) stays stable as the spread approaches zero or
        # grows large, where the correction to s^2 / 2 rounds to zero.
        if variance < 1e-16
            log_relative_std = log(s)
        elseif variance > 40.0
            log_relative_std = 0.5 * variance
        else
            log_relative_std = 0.5 * log(expm1(variance))
        end
        return mean, _exp_or_inf(log_mean_gap + log_relative_std)
    end
    function forward(y::Float64, state::Union{Nothing,Dict})
        st = state === nothing ? Dict{String,Any}() : state
        isfinite(y) || return y, st
        z = _fwd_query(y)
        if z == -direction * Inf
            throw(ArgumentError("log_transform: observation $y must lie inside $bound_label=$b"))
        end
        return z, st
    end
    function inverse_k(dists::Vector, state::Dict)
        return Any[PushforwardDist(d; inv_fn = _inv, fwd_fn = _fwd_query,
                                   log_jacobian_fn = _log_jacobian,
                                   component_moments_fn = _moments,
                                   scaled_inv_fn = _inverse_at_scale)
                   for d in dists]
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
    # Power iteration from one seed can miss the dominant eigenvalue when the
    # seed is (near-)orthogonal to it (skaters#232: phi = [3,-2,0,0,0] seeded at
    # all-ones reads 1, true radius 2). Two seeds, take the max.
    best = 0.0
    for seed in (ones(Float64, p), Float64[(-1.0)^i for i in 0:(p-1)])
        v = copy(seed)
        rho = 0.0
        for _ in 1:60
            nv = vcat(fsum(phi[j] * v[j] for j in 1:p), v[1:(p-1)])
            m = maximum(abs.(nv))
            if m == 0
                m = 1.0
            end
            v = nv ./ m
            rho = m
        end
        best = max(best, rho)
    end
    return best
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
                "phi_used" => zeros(Float64, p), "P" => _init_P(), "n" => 0)
        end
        buf = state["buffer"]::Vector{Float64}
        phi = state["phi"]::Vector{Float64}
        if !haskey(state, "phi_used")
            # a state from before skaters#232/#235 carries only the raw coefficients
            state["phi_used"] = _ar_stationary(copy(phi))
        end
        state["n"] += 1
        if length(buf) >= p
            x = Float64[buf[length(buf)-i] for i in 0:(p-1)]
            # the residual EMITTED (and inverse_k's forecast) uses the stationarity-
            # projected coefficients from the end of the previous tick (skaters#232);
            # the RLS keeps fitting the raw problem with its own prediction error
            phi_used = state["phi_used"]::Vector{Float64}
            prediction = fsum(phi_used[i+1] * x[i+1] for i in 0:(p-1))
            residual = y - prediction
            raw_residual = y - fsum(phi[i+1] * x[i+1] for i in 0:(p-1))
            P = state["P"]::Vector{Float64}
            Px = _mat_vec(P, x, p)
            denom = lam + _dot(x, Px, p)
            if abs(denom) > 1e-15
                K = [px / denom for px in Px]
                for i in 0:(p-1)
                    phi[i+1] += K[i+1] * raw_residual
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
            # re-project after every update so the next emission and inverse_k agree
            state["phi_used"] = _ar_stationary(copy(phi))
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
        phi = haskey(state, "phi_used") ? (state["phi_used"]::Vector{Float64}) :
              _ar_stationary(state["phi"]::Vector{Float64})
        H = length(dists)
        # Impulse responses psi_0..psi_{H-1}: psi_0 = 1, psi_i = sum_j phi_j psi_{i-j}.
        # The h-step variance is sigma^2 * sum_{i<=h} psi_i^2, NOT a recursion on
        # recovered variances -- successive horizons share one innovation, so
        # summing phi_j^2 * var_{h-j} both mis-weights and double counts it.
        psi = Float64[1.0]
        for i in 1:(H-1)
            push!(psi, fsum(Float64[phi[j+1] * psi[i-j] for j in 0:(p-1) if i - 1 - j >= 0]))
        end
        recovered_means = Float64[]
        result = Any[]
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
            # Var(y_{t+h+1}) = sum_i psi_{h-i}^2 var_i: each future innovation
            # enters with its own variance (skaters#245); plain accumulation in
            # the reference's order.
            total_var = 0.0
            for i in 0:h
                total_var += psi[h-i+1] * psi[h-i+1] * dist_var(dists[i+1])
            end
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
                "phi_used" => zeros(Float64, max_lag), "P" => _eye(n_groups, ridge), "n" => 0)
        end
        buf = state["buffer"]::Vector{Float64}
        theta = state["theta"]::Vector{Float64}
        if !haskey(state, "phi_used")
            # a state from before skaters#235: the projected per-lag set follows from theta
            state["phi_used"] = _ar_stationary(Float64[theta[groups[j+1]+1] for j in 0:(max_lag-1)])
        end
        state["n"] += 1
        if length(buf) >= max_lag
            x = _group_regressor(buf, groups, n_groups, max_lag)
            # Emitted residual uses the previous tick's stationarity-projected,
            # per-lag coefficients (skaters#232/#235), so forward's emission and
            # inverse_k's reconstruction agree; the RLS keeps fitting the raw
            # group-level problem with its own prediction error.
            phi_used = state["phi_used"]::Vector{Float64}
            prediction = fsum(phi_used[j+1] * buf[length(buf)-j] for j in 0:(max_lag-1))
            residual = y - prediction
            raw_residual = y - fsum(theta[g+1] * x[g+1] for g in 0:(n_groups-1))
            P = state["P"]::Vector{Float64}
            Px = _mat_vec(P, x, n_groups)
            denom = lam + _dot(x, Px, n_groups)
            if abs(denom) > 1e-15
                K = [px / denom for px in Px]
                for g in 0:(n_groups-1)
                    theta[g+1] += K[g+1] * raw_residual
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
            state["phi_used"] = _ar_stationary(Float64[theta[groups[j+1]+1] for j in 0:(max_lag-1)])
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
        # Same coefficients forward() already emitted residuals under
        # (skaters#232), except for a pre-#235 state that has no cache yet.
        phi = haskey(state, "phi_used") ? (state["phi_used"]::Vector{Float64}) :
              _ar_stationary(Float64[state["theta"][groups[j+1]+1] for j in 0:(max_lag-1)])
        H = length(dists)
        # Impulse responses, as in ar(): h-step variance is sigma^2 * sum psi_i^2.
        psi = Float64[1.0]
        for i in 1:(H-1)
            push!(psi, fsum(Float64[phi[j+1] * psi[i-j] for j in 0:(max_lag-1) if i - 1 - j >= 0]))
        end
        recovered_means = Float64[]
        result = Any[]
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
            # Var(y_{t+h+1}) = sum_i psi_{h-i}^2 var_i: each future innovation
            # enters with its own variance (skaters#245); plain accumulation in
            # the reference's order.
            total_var = 0.0
            for i in 0:h
                total_var += psi[h-i+1] * psi[h-i+1] * dist_var(dists[i+1])
            end
            total_std = total_var > 0 ? sqrt(total_var) : max(dist_std(d), 1e-12)
            push!(recovered_means, total_mean)
            push!(result, dist_gaussian(total_mean, total_std))
        end
        return result
    end
    return (forward, inverse_k)
end

# --- Gaussianize: probit of the smoothed fence-post empirical CDF ---

# Python's statistics.NormalDist().inv_cdf: Wichura's AS241 (PPND16), the
# same rational approximation CPython evaluates, term for term.
function _nd_inv_cdf(p::Float64)::Float64
    (0.0 < p < 1.0) || throw(ArgumentError("p must be in the range 0.0 < p < 1.0"))
    q = p - 0.5
    if abs(q) <= 0.425
        r = 0.180625 - q * q
        num = (((((((2.5090809287301226727e+3 * r +
                     3.3430575583588128105e+4) * r +
                     6.7265770927008700853e+4) * r +
                     4.5921953931549871457e+4) * r +
                     1.3731693765509461125e+4) * r +
                     1.9715909503065514427e+3) * r +
                     1.3314166789178437745e+2) * r +
                     3.3871328727963666080e+0) * q
        den = (((((((5.2264952788528545610e+3 * r +
                     2.8729085735721942674e+4) * r +
                     3.9307895800092710610e+4) * r +
                     2.1213794301586595867e+4) * r +
                     5.3941960214247511077e+3) * r +
                     6.8718700749205790830e+2) * r +
                     4.2313330701600911252e+1) * r +
                     1.0)
        return num / den
    end
    r = q <= 0.0 ? p : 1.0 - p
    r = sqrt(-log(r))
    if r <= 5.0
        r = r - 1.6
        num = (((((((7.74545014278341407640e-4 * r +
                     2.27238449892691845833e-2) * r +
                     2.41780725177450611770e-1) * r +
                     1.27045825245236838258e+0) * r +
                     3.64784832476320460504e+0) * r +
                     5.76949722146069140550e+0) * r +
                     4.63033784615654529590e+0) * r +
                     1.42343711074968357734e+0)
        den = (((((((1.05075007164441684324e-9 * r +
                     5.47593808499534494600e-4) * r +
                     1.51986665636164571966e-2) * r +
                     1.48103976427480074590e-1) * r +
                     6.89767334985100004550e-1) * r +
                     1.67638483018380384940e+0) * r +
                     2.05319162663775882187e+0) * r +
                     1.0)
    else
        r = r - 5.0
        num = (((((((2.01033439929228813265e-7 * r +
                     2.71155556874348757815e-5) * r +
                     1.24266094738807843860e-3) * r +
                     2.65321895265761230930e-2) * r +
                     2.96560571828504891230e-1) * r +
                     1.78482653991729133580e+0) * r +
                     5.46378491116411436990e+0) * r +
                     6.65790464350110377720e+0)
        den = (((((((2.04426310338993978564e-15 * r +
                     1.42151175831644588870e-7) * r +
                     1.84631831751005468180e-5) * r +
                     7.86869131145613259100e-4) * r +
                     1.48753612908506148525e-2) * r +
                     1.36929880922735805310e-1) * r +
                     5.99832206555887937690e-1) * r +
                     1.0)
    end
    x = num / den
    return q < 0.0 ? -x : x
end

# Empirical Gaussianization: z = Phi^{-1}(F_hat(y)) with fence-post weights.
#
# F_hat is built from a sorted buffer of past inputs, sampled at no more than
# max_knots equal-mass centers. smooth=false interpolates the fence-post
# empirical CDF linearly in (y, z); smooth=true uses a Gaussian-kernel CDF
# (Silverman bandwidth) through a monotone C^1 cubic (Fritsch-Butland
# tangents). Knot-segment slopes are guarded against slope_cap times the
# interquartile slope (guard "move" caps them, "delete" drops knots, "emin"
# projects knots into the smallest rank-displacement corridor), and the
# forward output is clamped to |z| <= z_max.
#
# Forward:   z_t = map(y_t) using the sketch built from y_1..y_{t-1}
# Inverse:   each component is mapped through the inverse by n_particles
#            equal-probability local Gaussians at mapped quantile midpoints,
#            widths from the mapped quantile spacing; a tail-spliced
#            predictive gets the exact pushforward.
#
# Before warmup observations the transform standardizes by expanding moments.
function gaussianize(; max_knots::Int = 39, warmup::Int = 30, n_particles::Int = 9,
                     smooth::Bool = false, slope_cap::Float64 = 20.0,
                     z_max::Float64 = 8.0, guard::String = "move")
    SLOPE_CAP = slope_cap
    Z_MAX = z_max
    SQ2 = sqrt(2.0)
    J = n_particles
    _mids = Float64[_nd_inv_cdf((j + 0.5) / J) for j in 0:(J-1)]
    _edges = Float64[_nd_inv_cdf(j / J) for j in 1:(J-1)]

    # Indices below are 0-based as in the reference; arrays are read at i + 1.

    function _iq_cap(ys, zs)
        K = length(ys)
        i1, i3 = K ÷ 4, 3 * K ÷ 4
        if i3 <= i1 || zs[i3+1] <= zs[i1+1]
            return nothing
        end
        cap = SLOPE_CAP * (ys[i3+1] - ys[i1+1]) / (zs[i3+1] - zs[i1+1])
        return cap <= 0 ? nothing : cap
    end

    # Delete knots whose segment slope dy/dz exceeds the cap, walking outward
    # from the median knot.
    function _prune_slopes(ys, zs)
        cap = _iq_cap(ys, zs)
        cap === nothing && return ys, zs
        K = length(ys)
        c = K ÷ 2
        keep = Int[c]
        for i in (c+1):(K-1)
            j = keep[end]
            if (ys[i+1] - ys[j+1]) / (zs[i+1] - zs[j+1]) <= cap
                push!(keep, i)
            end
        end
        for i in (c-1):-1:0
            j = minimum(keep)
            if (ys[j+1] - ys[i+1]) / (zs[j+1] - zs[i+1]) <= cap
                push!(keep, i)
            end
        end
        sort!(keep)
        length(keep) < 2 && return ys, zs
        return Float64[ys[i+1] for i in keep], Float64[zs[i+1] for i in keep]
    end

    # Smallest-rank-displacement guard: find the smallest e such that knot
    # values within e ranks of their order statistics admit the slope cap,
    # then project the knots into that corridor. Levels never move.
    function _emin_knots(buf, ys, zs, idxs)
        cap = _iq_cap(ys, zs)
        cap === nothing && return ys, zs
        nb, K = length(buf), length(ys)
        dz = Float64[zs[j+2] - zs[j+1] for j in 0:(K-2)]
        function corridor(e)
            lo = Float64[buf[max(0, idxs[j+1] - e)+1] for j in 0:(K-1)]
            hi = Float64[buf[min(nb - 1, idxs[j+1] + e)+1] for j in 0:(K-1)]
            U = zeros(K)
            U[K] = hi[K]
            for j in (K-2):-1:0
                U[j+1] = min(hi[j+1], U[j+2])
            end
            Db = zeros(K)
            Db[K] = lo[K]
            for j in (K-2):-1:0
                Db[j+1] = max(lo[j+1], Db[j+2] - cap * dz[j+1])
            end
            Lc = zeros(K)
            Lc[1] = Db[1]
            for j in 1:(K-1)
                Lc[j+1] = max(Db[j+1], Lc[j])
            end
            any(Lc[j] > U[j] for j in 1:K) && return nothing
            return Lc, U
        end
        corridor(0) !== nothing && return ys, zs
        lo_e, hi_e = 0, nb
        while hi_e - lo_e > 1
            mid = (lo_e + hi_e) ÷ 2
            if corridor(mid) !== nothing
                hi_e = mid
            else
                lo_e = mid
            end
        end
        res = corridor(hi_e)
        res === nothing && return ys, zs
        Lc, U = res
        w = zeros(K)
        w[1] = max(Lc[1], min(ys[1], U[1]))
        for j in 1:(K-1)
            a_j = max(Lc[j+1], w[j])
            b_j = min(U[j+1], w[j] + cap * dz[j])
            w[j+1] = max(a_j, min(ys[j+1], b_j))
        end
        # drop any knot that failed to stay strictly increasing
        ks, kz = Float64[w[1]], Float64[zs[1]]
        for j in 1:(K-1)
            if w[j+1] > ks[end]
                push!(ks, w[j+1])
                push!(kz, zs[j+1])
            end
        end
        return ks, kz
    end

    # Cap segment slopes dy/dz, walking outward from the median knot.
    function _cap_slopes!(ys, zs)
        cap = _iq_cap(ys, zs)
        cap === nothing && return
        K = length(ys)
        c = K ÷ 2
        for i in (c+1):(K-1)
            ys[i+1] = min(ys[i+1], ys[i] + cap * (zs[i+1] - zs[i]))
        end
        for i in (c-1):-1:0
            ys[i+1] = max(ys[i+1], ys[i+2] - cap * (zs[i+2] - zs[i+1]))
        end
    end

    # Fritsch-Butland harmonic-mean tangents: monotone C^1 cubic.
    function _tangents(xs, vs)
        n = length(xs)
        d = Float64[(vs[i+1] - vs[i]) / (xs[i+1] - xs[i]) for i in 1:(n-1)]
        t = Float64[d[1]]
        for i in 2:(n-1)
            a, b = d[i-1], d[i]
            push!(t, a * b > 0 ? 2.0 * a * b / (a + b) : 0.0)
        end
        push!(t, d[end])
        return t
    end

    function _herm(s, dx, v0, v1, t0, t1)
        s2 = s * s
        s3 = s2 * s
        return (v0 * (2 * s3 - 3 * s2 + 1) + dx * t0 * (s3 - 2 * s2 + s)
                + v1 * (-2 * s3 + 3 * s2) + dx * t1 * (s3 - s2))
    end

    function _herm_ds(s, dx, v0, v1, t0, t1)
        s2 = s * s
        return (v0 * (6 * s2 - 6 * s) + dx * t0 * (3 * s2 - 4 * s + 1)
                + v1 * (-6 * s2 + 6 * s) + dx * t1 * (3 * s2 - 2 * s))
    end

    function _build(buf)
        n = length(buf)
        m = min(max_knots, n)
        cs = Float64[]
        idxs = Int[]
        for i in 0:(m-1)
            idx = trunc(Int, (i + 0.5) * n / m)
            y = buf[idx+1]
            if !isempty(cs) && y <= cs[end]
                continue
            end
            push!(cs, y)
            push!(idxs, idx)
        end
        length(cs) < 2 && return nothing
        if !smooth
            zs = Float64[_nd_inv_cdf((idx + 1) / (n + 1)) for idx in idxs]
            if guard == "delete"
                cs, zs = _prune_slopes(cs, zs)
            elseif guard == "emin"
                cs, zs = _emin_knots(buf, cs, zs, idxs)
            else
                _cap_slopes!(cs, zs)
            end
            return (cs, zs, nothing)
        end
        # kernel-smoothed CDF on the sketch, Silverman bandwidth
        mm = length(cs)
        i1, i3 = mm ÷ 4, 3 * mm ÷ 4
        iqr = cs[i3+1] - cs[i1+1]
        mean = fsum(cs) / mm
        sd = sqrt(fsum((c - mean)^2 for c in cs) / mm)
        scale = iqr > 0 ? min(sd, iqr / 1.34) : sd
        if scale <= 0
            scale = max(abs(cs[end] - cs[1]), 1e-8)
        end
        h = max(0.9 * scale * Float64(n)^(-0.2), 1e-12)
        R = 8.0 * h
        zs = Float64[]
        lo = 0
        hi = 0
        for c in cs
            while lo < mm && cs[lo+1] < c - R
                lo += 1
            end
            while hi < mm && cs[hi+1] <= c + R
                hi += 1
            end
            f = (lo + fsum(0.5 * (1.0 + erf((c - cs[j+1]) / (SQ2 * h))) for j in lo:(hi-1))) / mm
            push!(zs, _nd_inv_cdf((1.0 + n * f) / (n + 2.0)))
        end
        cs2, zs2 = Float64[cs[1]], Float64[zs[1]]
        for (c, z) in zip(cs[2:end], zs[2:end])
            if z > zs2[end] + 1e-12
                push!(cs2, c)
                push!(zs2, z)
            end
        end
        length(cs2) < 2 && return nothing
        _cap_slopes!(cs2, zs2)
        return (cs2, zs2, _tangents(cs2, zs2))
    end

    function _fwd(y, knots)
        cs, zs, ts = knots
        if y <= cs[1]
            s0 = ts !== nothing ? ts[1] : (zs[2] - zs[1]) / (cs[2] - cs[1])
            return zs[1] + s0 * (y - cs[1])
        end
        if y >= cs[end]
            s1 = ts !== nothing ? ts[end] : (zs[end] - zs[end-1]) / (cs[end] - cs[end-1])
            return zs[end] + s1 * (y - cs[end])
        end
        i = searchsortedlast(cs, y)          # bisect_right - 1, plus the 1 offset
        if ts === nothing
            return zs[i] + (zs[i+1] - zs[i]) * (y - cs[i]) / (cs[i+1] - cs[i])
        end
        dx = cs[i+1] - cs[i]
        return _herm((y - cs[i]) / dx, dx, zs[i], zs[i+1], ts[i], ts[i+1])
    end

    function _inv(z, knots)
        cs, zs, ts = knots
        if z <= zs[1]
            s0 = ts !== nothing ? ts[1] : (zs[2] - zs[1]) / (cs[2] - cs[1])
            return cs[1] + (z - zs[1]) / max(s0, 1e-12)
        end
        if z >= zs[end]
            s1 = ts !== nothing ? ts[end] : (zs[end] - zs[end-1]) / (cs[end] - cs[end-1])
            return cs[end] + (z - zs[end]) / max(s1, 1e-12)
        end
        i = searchsortedlast(zs, z)
        if ts === nothing
            return cs[i] + (cs[i+1] - cs[i]) * (z - zs[i]) / (zs[i+1] - zs[i])
        end
        dx = cs[i+1] - cs[i]
        s = (z - zs[i]) / (zs[i+1] - zs[i])
        for _ in 1:6
            d = _herm_ds(s, dx, zs[i], zs[i+1], ts[i], ts[i+1])
            if d <= 1e-14
                break
            end
            s -= (_herm(s, dx, zs[i], zs[i+1], ts[i], ts[i+1]) - z) / d
            s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s)
        end
        return cs[i] + s * dx
    end

    function _moments(state)
        n, mu, m2 = state["n"], state["mu"], state["m2"]
        var = n > 2 ? m2 / (n - 1) : 0.0
        sigma = var > 0 ? sqrt(var) : 1e-8
        return mu, sigma
    end

    function forward(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("buf" => Float64[], "n" => 0, "mu" => 0.0,
                                     "m2" => 0.0, "knots" => nothing)
        end
        knots = state["knots"]
        if knots !== nothing
            z = _fwd(y, knots)
        else
            mu, sigma = _moments(state)
            if sigma <= 1e-8
                # cold start: scale by the first observed deviation so the
                # early z values are O(1) and cannot poison downstream state
                sigma = max(abs(y - mu), 1e-8)
            end
            z = state["n"] != 0 ? (y - mu) / sigma : 0.0
        end
        z = max(-Z_MAX, min(Z_MAX, z))
        buf = state["buf"]::Vector{Float64}
        insert!(buf, searchsortedlast(buf, y) + 1, y)      # bisect.insort
        n = state["n"] + 1
        d0 = y - state["mu"]
        mu = state["mu"] + d0 / n
        state["n"] = n
        state["mu"] = mu
        state["m2"] = state["m2"] + d0 * (y - mu)
        state["knots"] = n >= warmup ? _build(buf) : nothing
        return z, state
    end

    function inverse_k(dists::Vector, state::Dict)
        knots = state["knots"]
        if knots === nothing
            mu, sigma = _moments(state)
            return Any[dist_affine(d, sigma, mu) for d in dists]
        end
        result = Any[]
        for d in dists
            dcomps = _components(d)
            if dcomps === nothing   # tail-spliced: exact pushforward
                push!(result, _pushforward(d, z -> _inv(z, knots)))
                continue
            end
            comps = NTuple{3,Float64}[]
            for (w, mu, sig) in dcomps
                pts = Float64[_inv(mu + sig * q, knots) for q in _mids]
                eds = Float64[_inv(mu + sig * e, knots) for e in _edges]
                for j in 1:J
                    if j > 1 && j < J
                        std = (eds[j] - eds[j-1]) / 2.0
                    elseif j < J
                        std = eds[j] - pts[j]
                    else
                        std = pts[j] - eds[j-1]
                    end
                    push!(comps, (w / J, pts[j], max(std, 1e-9)))
                end
            end
            push!(result, dist_prune(Dist(comps), 12))
        end
        return result
    end

    return (forward, inverse_k)
end
