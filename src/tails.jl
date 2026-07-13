# GPD tail splice. Port of skaters/tails.py.

const _TEPS = 1e-12
const _REFIT_EVERY = 25

# --- standard normal helpers (Acklam inverse, verbatim) ---

_phi_t(z::Float64) = 0.5 * erfc(-z / sqrt(2.0))
_phi_logpdf(z::Float64) = -0.5 * z * z - LOG_SQRT2PI

const _ACK_A = (-3.969683028665376e+01, 2.209460984245205e+02,
    -2.759285104469687e+02, 1.383577518672690e+02,
    -3.066479806614716e+01, 2.506628277459239e+00)
const _ACK_B = (-5.447609879822406e+01, 1.615858368580409e+02,
    -1.556989798598866e+02, 6.680131188771972e+01,
    -1.328068155288572e+01)
const _ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01,
    -2.400758277161838e+00, -2.549732539343734e+00,
    4.374664141464968e+00, 2.938163982698783e+00)
const _ACK_D = (7.784695709041462e-03, 3.224671290700398e-01,
    2.445134137142996e+00, 3.754408661907416e+00)

function _phi_inv(p::Float64)::Float64
    p = min(max(p, _TEPS), 1.0 - _TEPS)
    if p < 0.02425
        q = sqrt(-2.0 * log(p))
        x = ((((((_ACK_C[1] * q + _ACK_C[2]) * q + _ACK_C[3]) * q
                + _ACK_C[4]) * q + _ACK_C[5]) * q + _ACK_C[6])
             / ((((_ACK_D[1] * q + _ACK_D[2]) * q + _ACK_D[3]) * q
                 + _ACK_D[4]) * q + 1.0))
    elseif p <= 0.97575
        q = p - 0.5
        r = q * q
        x = ((((((_ACK_A[1] * r + _ACK_A[2]) * r + _ACK_A[3]) * r
                + _ACK_A[4]) * r + _ACK_A[5]) * r + _ACK_A[6]) * q
             / (((((_ACK_B[1] * r + _ACK_B[2]) * r + _ACK_B[3]) * r
                  + _ACK_B[4]) * r + _ACK_B[5]) * r + 1.0))
    else
        q = sqrt(-2.0 * log(1.0 - p))
        x = -((((((_ACK_C[1] * q + _ACK_C[2]) * q + _ACK_C[3]) * q
                 + _ACK_C[4]) * q + _ACK_C[5]) * q + _ACK_C[6])
              / ((((_ACK_D[1] * q + _ACK_D[2]) * q + _ACK_D[3]) * q
                  + _ACK_D[4]) * q + 1.0))
    end
    e = _phi_t(x) - p
    u = e * sqrt(2.0 * pi) * exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)
end

# --- GPD helpers ---

function _gpd_logpdf(e::Float64, gamma::Float64, sigma::Float64)::Float64
    if abs(gamma) < 1e-9
        return -log(sigma) - e / sigma
    end
    arg = 1.0 + gamma * e / sigma
    if arg <= 0.0
        return -745.0
    end
    return -log(sigma) - (1.0 / gamma + 1.0) * log(arg)
end

function _gpd_sf(e::Float64, gamma::Float64, sigma::Float64)::Float64
    if e <= 0.0
        return 1.0
    end
    if abs(gamma) < 1e-9
        return exp(-e / sigma)
    end
    arg = 1.0 + gamma * e / sigma
    if arg <= 0.0
        return 0.0
    end
    return arg^(-1.0 / gamma)
end

function _gpd_isf(p::Float64, gamma::Float64, sigma::Float64)::Float64
    p = min(max(p, 1e-300), 1.0)
    if abs(gamma) < 1e-9
        return -sigma * log(p)
    end
    return sigma / gamma * (p^(-gamma) - 1.0)
end

const _TAU_GRID = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0)

function _fit_ml(exc::Vector{Float64}, s1::Float64)
    n = length(exc)
    emean = s1 / n
    if n < 20 || emean <= 0.0
        return 0.0, max(emean, 1e-12)
    end
    emax = maximum(exc)
    best_g = 0.0
    best_s = max(emean, 1e-12)
    best_ll = -1e300
    taus = Float64[t / emean for t in _TAU_GRID]
    push!(taus, -0.5 / emax)
    push!(taus, -0.25 / emax)
    push!(taus, -0.1 / emax)
    for tau in taus
        if tau <= -1.0 / emax || abs(tau) < 1e-12
            continue
        end
        g = 0.0
        for e in exc
            g += log1p(tau * e)
        end
        g /= n
        if g <= 1e-9
            continue
        end
        sigma = g / tau
        if sigma <= 0.0
            continue
        end
        ll = -n * log(sigma) - (1.0 + 1.0 / g) * n * g
        if ll > best_ll
            best_ll = ll
            best_g = g
            best_s = sigma
        end
    end
    return best_g, best_s
end

# --- spliced predictive ---

const _GRID_N = 65

mutable struct SplicedDist <: AbstractDist
    body::Dist
    t_lo::Float64
    t_up::Float64
    zeta_lo::Float64
    zeta_up::Float64
    g_lo::Float64
    s_lo::Float64
    g_up::Float64
    s_up::Float64
    _c::Float64
    _plo::Float64
    _pup::Float64
    _grid::Union{Nothing,Vector{Float64}}
    function SplicedDist(body, t_lo, t_up, zeta_lo, zeta_up, g_lo, s_lo, g_up, s_up)
        plo = _phi_t(t_lo)
        pup = _phi_t(t_up)
        interior = max(pup - plo, 1e-12)
        c = max(1.0 - zeta_lo - zeta_up, 1e-12) / interior
        new(body, t_lo, t_up, zeta_lo, zeta_up, g_lo, s_lo, g_up, s_up, c, plo, pup, nothing)
    end
end

dist_body(d::SplicedDist) = d.body

function _z(d::SplicedDist, x::Float64)::Float64
    u = min(max(dist_cdf(d.body, x), _TEPS), 1.0 - _TEPS)
    return _phi_inv(u)
end

function dist_cdf(d::SplicedDist, x::Float64)::Float64
    z = _z(d, x)
    if z < d.t_lo
        return d.zeta_lo * _gpd_sf(d.t_lo - z, d.g_lo, d.s_lo)
    end
    if z > d.t_up
        return 1.0 - d.zeta_up * _gpd_sf(z - d.t_up, d.g_up, d.s_up)
    end
    return d.zeta_lo + d._c * (_phi_t(z) - d._plo)
end

function dist_logpdf(d::SplicedDist, x::Float64)::Float64
    base = dist_logpdf(d.body, x)
    if !isfinite(base)
        return base
    end
    z = _z(d, x)
    if z < d.t_lo
        corr = log(max(d.zeta_lo, 1e-300)) + _gpd_logpdf(d.t_lo - z, d.g_lo, d.s_lo) - _phi_logpdf(z)
    elseif z > d.t_up
        corr = log(max(d.zeta_up, 1e-300)) + _gpd_logpdf(z - d.t_up, d.g_up, d.s_up) - _phi_logpdf(z)
    else
        corr = log(d._c)
    end
    return base + corr
end

function dist_pdf(d::SplicedDist, x::Float64)::Float64
    lp = dist_logpdf(d, x)
    return lp < 700.0 ? exp(lp) : Inf
end

function dist_quantile(d::SplicedDist, p::Float64; tol::Float64 = 1e-9, max_iter::Int = 100)::Float64
    @assert 0 < p < 1
    if p < d.zeta_lo
        z = d.t_lo - _gpd_isf(p / d.zeta_lo, d.g_lo, d.s_lo)
    elseif p > 1.0 - d.zeta_up
        z = d.t_up + _gpd_isf((1.0 - p) / d.zeta_up, d.g_up, d.s_up)
    else
        u = d._plo + (p - d.zeta_lo) / d._c
        z = _phi_inv(min(max(u, _TEPS), 1.0 - _TEPS))
    end
    ub = min(max(_phi_t(z), _TEPS), 1.0 - _TEPS)
    return dist_quantile(d.body, ub; tol = tol, max_iter = max_iter)
end

function _qgrid(d::SplicedDist)::Vector{Float64}
    if d._grid === nothing
        n = _GRID_N
        d._grid = Float64[dist_quantile(d, (i + 0.5) / n) for i in 0:(n-1)]
    end
    return d._grid
end

function dist_mean(d::SplicedDist)::Float64
    q = _qgrid(d)
    return fsum(q) / length(q)
end

function dist_var(d::SplicedDist)::Float64
    q = _qgrid(d)
    m = fsum(q) / length(q)
    return fsum((x - m) * (x - m) for x in q) / length(q)
end

dist_std(d::SplicedDist)::Float64 = sqrt(dist_var(d))

function dist_crps(d::SplicedDist, x::Float64)::Float64
    q = _qgrid(d)
    n = length(q)
    t1 = fsum(abs(v - x) for v in q) / n
    t2 = 2.0 * fsum(q[i+1] * (2.0 * (i + 0.5) / n - 1.0) for i in 0:(n-1)) / n
    return t1 - t2 * 0.5
end

# --- the wrapper ---

const _GUARD_SF = 1e-3
const _ADAPT_AFTER = 10

_tail_new() = Dict{String,Any}("t" => nothing, "exc" => Float64[], "s1" => 0.0, "nx" => 0,
    "r" => 0.0, "g" => 0.0, "s" => 1.0, "since" => 0, "run" => 0)

function _tail_add!(tail::Dict, e::Float64, nexc::Int)
    exc = tail["exc"]::Vector{Float64}
    if length(exc) >= 20
        cap = _gpd_isf(_GUARD_SF, tail["g"], tail["s"])
        if e > cap
            if tail["run"] < _ADAPT_AFTER
                e = cap
            end
            tail["run"] += 1
        else
            tail["run"] = 0
        end
    end
    push!(exc, e)
    tail["s1"] += e
    tail["nx"] += 1
    if length(exc) > nexc
        tail["s1"] -= popfirst!(exc)
    end
    tail["since"] += 1
    if tail["since"] >= _REFIT_EVERY || length(exc) <= 25
        g, s = _fit_ml(exc, tail["s1"])
        tail["g"] = g
        tail["s"] = s
        tail["since"] = 0
    end
end

function gpdtails(base, k::Int; level::Float64 = 0.98, nexc::Int = 500,
                  warmup::Int = 500, rate_alpha::Float64 = 0.002)
    @assert k >= 1
    @assert 0.5 < level < 1.0
    @assert nexc >= 50 && warmup >= 100
    @assert 0.0 < rate_alpha < 0.1

    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("base" => nothing, "pending" => Any[],
                "tails" => Any[Dict{String,Any}("up" => _tail_new(), "lo" => _tail_new(),
                    "warm" => Float64[], "n" => 0) for _ in 1:k])
        end
        pend = state["pending"]
        n = length(pend)
        tails = state["tails"]
        for m in 1:k
            if m > n
                continue
            end
            d = pend[n-m+1][m]
            u = dist_cdf(d, y)
            if !isfinite(u)
                continue
            end
            z = _phi_inv(min(max(u, _TEPS), 1.0 - _TEPS))
            th = tails[m]
            up = th["up"]::Dict
            lo = th["lo"]::Dict
            if up["t"] === nothing
                warm = th["warm"]::Vector{Float64}
                push!(warm, z)
                if length(warm) >= warmup
                    w = sort(warm)
                    iu = min(trunc(Int, level * length(w)), length(w) - 1)
                    up["t"] = w[iu+1]
                    lo["t"] = w[length(w)-iu]
                    for x in w
                        if x > up["t"]
                            _tail_add!(up, x - up["t"], nexc)
                        elseif x < lo["t"]
                            _tail_add!(lo, lo["t"] - x, nexc)
                        end
                    end
                    th["n"] = length(w)
                    up["r"] = up["nx"] / length(w)
                    lo["r"] = lo["nx"] / length(w)
                    th["warm"] = Float64[]
                end
            else
                th["n"] += 1
                up["r"] += rate_alpha * ((z > up["t"] ? 1.0 : 0.0) - up["r"])
                lo["r"] += rate_alpha * ((z < lo["t"] ? 1.0 : 0.0) - lo["r"])
                if z > up["t"]
                    _tail_add!(up, z - up["t"], nexc)
                elseif z < lo["t"]
                    _tail_add!(lo, lo["t"] - z, nexc)
                end
            end
        end

        dists, state["base"] = base(y, state["base"])
        push!(pend, collect(dists))
        if length(pend) > k
            popfirst!(pend)
        end

        out = Any[]
        for m in 1:k
            d = dists[m]
            th = tails[m]
            up = th["up"]::Dict
            lo = th["lo"]::Dict
            if up["t"] === nothing || length(up["exc"]) < 8 || length(lo["exc"]) < 8 || th["n"] <= 0
                push!(out, d)
                continue
            end
            push!(out, SplicedDist(d, lo["t"], up["t"], lo["r"], up["r"],
                lo["g"], lo["s"], up["g"], up["s"]))
        end
        return out, state
    end
    return _skater
end
