# Online homogenized scale-mixture correction. Port of skaters/homogenize.py.
#
# A small Bayesian-hedged pool of two-point Gaussian scale-mixture hypotheses
#
#     g(z) = 1/2 N(0, v(1 - delta))(z) + 1/2 N(0, v(1 + delta))(z)
#
# (a single N(0, v) when delta == 0) for a skater's own PIT residual stream.
# v plays the H_2 role (the predictable variance level); delta adds excess
# kurtosis from purely Gaussian pieces. A fixed grid of (rho, gain, delta)
# candidates plus a distinguished identity candidate (frozen at v = 1) are
# combined online: each carries a Huber-capped filter q_t on z_t^2 - 1, with a
# changepoint escape, and a discounted Bayesian log-weight. Pooling is exact,
# the union of every candidate's (scale, weight) pairs.
#
# The pooled correction composes back onto the base predictive as in
# residual_transform:  F~_t(y) = H_t(Phi^-1(F_t(y))),
# Q~_t(p) = F_t.quantile(Phi(H_t.quantile(p))), evaluated by HomogenizedDist.
# Apply it explicitly: homogenize(laplace(k = 1)).

const _HG_EPS = 1e-12

# Candidate and pool stability bounds.
const _HG_V_MIN, _HG_V_MAX = 0.25, 9.0
const _HG_DELTA_MAX = 0.8
const _HG_INNOVATION_CAP = 8.0
const _HG_ADAPT_AFTER = 10

# Prequential weight-learning constants.
const _HG_WEIGHT_DECAY = 0.9999
const _HG_LEARNING_RATE = 0.1
const _HG_SCORE_CAP = 20.0

_hg_phi_logpdf(z::Float64) = -0.5 * z * z - LOG_SQRT2PI

# Log-density of a scale mixture {"weights", "scales"} at z (log-sum-exp).
function _hg_logg(z::Float64, weights, scales)::Float64
    best = -Inf
    terms = Float64[]
    for (w, s) in zip(weights, scales)
        w <= 0.0 && continue
        t = log(w) - log(s) + _hg_phi_logpdf(z / s)
        push!(terms, t)
        if t > best
            best = t
        end
    end
    best == -Inf && return -Inf
    return best + log(fsum(exp(t - best) for t in terms))
end

_hg_logg(z::Float64, correction::Dict) = _hg_logg(z, correction["weights"], correction["scales"])

# Static candidate grid, plus a distinguished identity candidate
# (rho = gain = delta = 0, frozen at v = 1). The scale mixture is a density only
# for 0 <= delta < 1, so delta_max and every delta_grid entry must lie there.
function homogenize_candidates(; rho_grid = (0.0, 0.8, 0.95, 0.99), gain_grid = (0.03, 0.12),
                               delta_grid = (0.0, 0.25, 0.50), delta_max::Float64 = _HG_DELTA_MAX)
    (0.0 <= delta_max < 1.0) ||
        throw(ArgumentError("delta_max must satisfy 0 <= delta_max < 1, got $delta_max"))
    for delta in delta_grid
        (0.0 <= Float64(delta) < 1.0) ||
            throw(ArgumentError("delta_grid entries must satisfy 0 <= delta < 1, got $delta"))
    end
    candidates = Dict{String,Any}[Dict{String,Any}("rho" => 0.0, "gain" => 0.0, "delta" => 0.0,
                                                   "is_identity" => true)]
    for rho in rho_grid, gain in gain_grid, delta in delta_grid
        push!(candidates, Dict{String,Any}("rho" => Float64(rho), "gain" => Float64(gain),
                                           "delta" => _rt_clip(Float64(delta), 0.0, delta_max),
                                           "is_identity" => false))
    end
    return Tuple(candidates)
end

# Python's loop nests rho outermost, then gain, then delta; Julia's
# `for a in A, b in B` nests the same way (the last range varies fastest).

_cand_scales(v::Float64, delta::Float64) =
    delta <= 0.0 ? (sqrt(v),) : (sqrt(v * (1.0 - delta)), sqrt(v * (1.0 + delta)))
_cand_weights(delta::Float64) = delta <= 0.0 ? (1.0,) : (0.5, 0.5)
_cand_logpdf(z::Float64, v::Float64, delta::Float64) =
    _hg_logg(z, _cand_weights(delta), _cand_scales(v, delta))

function _hg_init_state(candidates)
    n_other = max(count(c -> !c["is_identity"], candidates), 1)
    dynamic = Dict{String,Any}[]
    for c in candidates
        lw = c["is_identity"] ? log(n_other) : 0.0
        push!(dynamic, Dict{String,Any}("q" => 0.0, "log_weight" => lw, "guard_run" => 0))
    end
    return Dict{String,Any}("dynamic" => dynamic, "issued" => nothing, "n" => 0)
end

# Fixed-gain filter on x = z^2 - 1, Huber-capped, with a changepoint escape.
function _update_q!(d::Dict, cfg::Dict, x::Float64)
    rho, gain = cfg["rho"], cfg["gain"]
    q_minus = rho * d["q"]
    resid = x - q_minus
    capped = abs(resid) > _HG_INNOVATION_CAP
    d["guard_run"] = capped ? d["guard_run"] + 1 : 0
    used = (capped && d["guard_run"] <= _HG_ADAPT_AFTER) ?
           _rt_clip(resid, -_HG_INNOVATION_CAP, _HG_INNOVATION_CAP) : resid
    d["q"] = q_minus + gain * used
    return d
end

function _hg_softmax(dynamic)
    log_weights = Float64[d["log_weight"] for d in dynamic]
    m = maximum(log_weights)
    raw = Float64[exp(lw - m) for lw in log_weights]
    total = fsum(raw)
    return Float64[r / total for r in raw]
end

# Exact Bayesian pooling: the union of each candidate's (scale, weight) pairs.
function _hg_pool(candidates, dynamic)
    omega = _hg_softmax(dynamic)
    weights = Float64[]
    scales = Float64[]
    for (cfg, d, w) in zip(candidates, dynamic, omega)
        v = _rt_clip(1.0 + d["q"], _HG_V_MIN, _HG_V_MAX)
        for (s, cw) in zip(_cand_scales(v, cfg["delta"]), _cand_weights(cfg["delta"]))
            push!(scales, s)
            push!(weights, w * cw)
        end
    end
    return Dict{String,Any}("weights" => weights, "scales" => scales)
end

# Current softmax weights, same order as the candidates. Diagnostics only.
candidate_weights(state::Dict) = _hg_softmax(state["dynamic"])

# Score each candidate's pre-update correction against z, update weights,
# update each candidate's variance state, then pool the correction issued for
# the next observation. Returns (issued_correction, new_state).
function cell_step(z::Float64, state, candidates)
    if state === nothing
        state = _hg_init_state(candidates)
    end
    dynamic = state["dynamic"]
    x = z * z - 1.0
    for (cfg, d) in zip(candidates, dynamic)
        v = _rt_clip(1.0 + d["q"], _HG_V_MIN, _HG_V_MAX)
        r = _cand_logpdf(z, v, cfg["delta"]) - _hg_phi_logpdf(z)
        d["log_weight"] = _HG_WEIGHT_DECAY * d["log_weight"] +
                          _HG_LEARNING_RATE * _rt_clip(r, -_HG_SCORE_CAP, _HG_SCORE_CAP)
    end
    for (cfg, d) in zip(candidates, dynamic)
        _update_q!(d, cfg, x)
    end
    correction = _hg_pool(candidates, dynamic)
    state["issued"] = correction
    state["n"] += 1
    return correction, state
end

# --- HomogenizedDist: F~_t = H_t[Phi^-1(F_t(.))], H_t the pooled scale mixture ---

mutable struct HomogenizedDist <: AbstractDist
    base::Any
    correction::Dict{String,Any}
    _qgrid::Union{Nothing,Vector{Float64}}
end

HomogenizedDist(base, correction::Dict) = HomogenizedDist(base, Dict{String,Any}(correction), nothing)

_hg_z(d::HomogenizedDist, y::Float64) = _rt_phi_inv(_rt_clamp01(dist_cdf(d.base, y)))

function _hg_z_cdf(d::HomogenizedDist, z::Float64)
    total = 0.0
    for (w, s) in zip(d.correction["weights"], d.correction["scales"])
        total += w * dist_cdf(_RT_STD_NORMAL, z / s)
    end
    return _rt_clamp01(total)
end

function _hg_z_quantile(d::HomogenizedDist, p::Float64; tol::Float64 = 1e-9, max_iter::Int = 100)
    lo, hi = -40.0, 40.0
    for _ in 1:max_iter
        mid = 0.5 * (lo + hi)
        if _hg_z_cdf(d, mid) < p
            lo = mid
        else
            hi = mid
        end
        hi - lo < tol && break
    end
    return 0.5 * (lo + hi)
end

dist_cdf(d::HomogenizedDist, y::Float64)::Float64 = _hg_z_cdf(d, _hg_z(d, y))

function dist_logpdf(d::HomogenizedDist, y::Float64)::Float64
    z = _hg_z(d, y)
    return dist_logpdf(d.base, y) + _hg_logg(z, d.correction) - _hg_phi_logpdf(z)
end

function dist_pdf(d::HomogenizedDist, y::Float64)::Float64
    lp = dist_logpdf(d, y)
    return lp > -700.0 ? exp(lp) : 0.0
end

function dist_quantile(d::HomogenizedDist, p::Float64; tol::Float64 = 1e-9, max_iter::Int = 2100)::Float64
    @assert 0 < p < 1
    hp = _hg_z_quantile(d, p; tol = tol, max_iter = max_iter)
    u = _rt_clamp01(dist_cdf(_RT_STD_NORMAL, hp))
    return dist_quantile(d.base, u; tol = tol, max_iter = max_iter)
end

function _quantile_grid(d::HomogenizedDist)
    if d._qgrid === nothing
        d._qgrid = Float64[dist_quantile(d, p) for p in _RT_QGRID_P]
    end
    return d._qgrid
end

dist_mean(d::HomogenizedDist)::Float64 = _qgrid_moments_mean(_quantile_grid(d))
dist_var(d::HomogenizedDist)::Float64 = _qgrid_var(_quantile_grid(d))
function dist_std(d::HomogenizedDist)::Float64
    v = dist_var(d)
    return v > 0 ? sqrt(v) : 0.0
end
dist_crps(d::HomogenizedDist, x::Float64)::Float64 = _qgrid_crps(_quantile_grid(d), x)

# --- the wrapper ---

# Borrow strength across horizons, in place. "weights" pulls each candidate's
# log-weight toward its cross-horizon mean; "state" pulls each candidate's
# filter state q_m toward q_{m-1}. lam = 0 is independence.
function _hg_couple!(cells::Vector, mode::String, lam::Float64)
    live = [c for c in cells if c !== nothing]
    (length(live) < 2 || lam <= 0.0) && return
    n = length(live[1]["dynamic"])
    if mode == "weights"
        for i in 1:n
            mean_w = fsum(c["dynamic"][i]["log_weight"] for c in live) / length(live)
            for c in live
                d = c["dynamic"][i]
                d["log_weight"] = (1.0 - lam) * d["log_weight"] + lam * mean_w
            end
        end
    elseif mode == "state"
        idxs = [j for (j, c) in enumerate(cells) if c !== nothing]
        for i in 1:n
            prev = nothing
            for j in idxs
                d = cells[j]["dynamic"][i]
                if prev !== nothing
                    d["q"] = (1.0 - lam) * d["q"] + lam * prev
                end
                prev = d["q"]
            end
        end
    end
    return
end

# Wrap a skater with an online homogenized scale-mixture correction, one
# independent candidate pool per horizon (couple = "none"), each scored against
# its own matured forecast. couple = "weights" or "state" shrinks across
# horizons by lam; "shared" uses one pool for every horizon. The state carries
# state["pending"], state["cell"][m], state["correction"][m] and state["raw"].
function homogenize(base; candidates = nothing, k::Int = 1, couple::String = "none",
                    lam::Float64 = 0.25)
    @assert k >= 1
    @assert couple in ("none", "weights", "state", "shared")
    cands = candidates !== nothing ? candidates : homogenize_candidates()

    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("base" => nothing, "raw" => nothing, "pending" => Any[],
                                     "cell" => Any[nothing for _ in 1:k],
                                     "correction" => Any[nothing for _ in 1:k])
        end
        s = state
        pend = s["pending"]
        n = length(pend)
        for m in 1:k
            m > n && continue
            d_m = pend[n-m+1][m]
            z = _rt_phi_inv(_rt_clamp01(dist_cdf(d_m, y)))
            z == z || continue                   # non-finite PIT: no information
            slot = couple == "shared" ? 1 : m
            corr, s["cell"][slot] = cell_step(z, s["cell"][slot], cands)
            if couple == "shared"
                for j in 1:k
                    s["correction"][j] = corr
                end
            else
                s["correction"][m] = corr
            end
        end

        if couple in ("weights", "state")
            _hg_couple!(s["cell"], couple, lam)
            for m in 1:k                          # re-issue after shrinkage
                if s["cell"][m] !== nothing
                    s["correction"][m] = _hg_pool(cands, s["cell"][m]["dynamic"])
                end
            end
        end

        dists, s["base"] = base(y, s["base"])
        s["raw"] = dists[1]
        push!(pend, collect(dists))
        if length(pend) > k
            popfirst!(pend)
        end

        out = Any[]
        for m in 1:k
            d = m <= length(dists) ? dists[m] : dists[end]
            corr = s["correction"][m]
            push!(out, corr === nothing ? d : HomogenizedDist(d, corr))
        end
        return out, s
    end
    return _skater
end
