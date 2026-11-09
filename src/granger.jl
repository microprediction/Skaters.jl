# granger: laplace with trained parameters, mixed with laplace so it cannot do
# much worse. Port of the Python reference skaters/granger.py (horizon 1),
# parity-checked against it (parity/check_granger.jl).
#
# After Clive Granger, whose work with Bates (1969) founded forecast
# combination. The issued predictive is the Aggregating-Algorithm mixture of
# laplace and its trained twin under log loss, so its total log score is never
# more than log(1 / pi_lap) behind laplace's (log 4 at the shipped 0.25).

import JSON

const GRANGER_PARAMS_DIR = joinpath(@__DIR__, "..", "params")

"""
    granger_params(frequency = "monthly")

A shipped granger parameter set: "monthly", "weekly" or "hourly".
"""
function granger_params(frequency::String = "monthly")
    path = joinpath(GRANGER_PARAMS_DIR, "$(frequency).json")
    isfile(path) || error("no shipped granger parameters for $(frequency)")
    return JSON.parsefile(path)
end

_f(x) = Float64(x)

function _granger_transform(name::String, kw::Dict)
    if name == "ema_transform"
        return ema_transform(_f(kw["alpha"]))
    elseif name == "difference"
        return difference()
    elseif name == "drift"
        return drift(_f(kw["alpha"]), _f(kw["shrinkage"]))
    elseif name == "theta"
        return theta(_f(kw["alpha"]))
    elseif name == "ar"
        return ar(Int(kw["order"]); lam = _f(kw["lam"]), ridge = _f(kw["ridge"]), decay = _f(kw["decay"]))
    elseif name == "holt_linear"
        return holt_linear(_f(kw["alpha"]), _f(kw["beta"]))
    elseif name == "seasonal_difference"
        return seasonal_difference(Int(kw["period"]))
    elseif name == "seasonal_anchor"
        return seasonal_anchor(Int(kw["period"]); alpha = _f(kw["alpha"]), weight = _f(kw["weight"]))
    elseif name == "standardize"
        return standardize(_f(kw["alpha"]), _f(kw["eps"]))
    elseif name == "fractional_difference"
        return fractional_difference(_f(kw["d"]), Int(kw["window"]))
    elseif name == "garch"
        return garch(; omega = _f(kw["omega"]), alpha = _f(kw["alpha"]), beta = _f(kw["beta"]),
                     mean_alpha = _f(kw["mean_alpha"]), eps = _f(kw["eps"]))
    elseif name == "power_transform"
        get(kw, "exact", false) == true && error("granger: power_transform(exact=True) is not ported")
        return power_transform(_f(kw["p"]))
    elseif name == "yeo_johnson"
        get(kw, "exact", false) == true && error("granger: yeo_johnson(exact=True) is not ported")
        return yeo_johnson(_f(kw["lmbda"]))
    end
    error("granger: unknown transform $(name)")
end

function _granger_candidate(chain::Vector, k::Int)
    f = leaf(k)
    for j in length(chain):-1:1
        name, kw = chain[j][1], chain[j][2]
        f = conjugate(f, _granger_transform(String(name), kw), k)
    end
    return f
end

function _granger_expert(P::Dict, k::Int, gate_hook)
    cands = Any[_granger_candidate(c["chain"], k) for c in P["candidates"]]
    depths = Float64[_f(c["depth"]) for c in P["candidates"]]
    tl = P["tleaf"]
    eta, sa = _f(tl["eta"]), _f(tl["scale_alpha"])
    t = P["trunk"]
    f = terminal_leaf_ensemble(cands; leaf_fn = kk -> crps_leaf(kk; eta = eta, scale_alpha = sa), k = k,
        learning_rate = _f(t["learning_rate"]), complexity_penalty = _f(t["complexity_penalty"]),
        depths = depths, prior_log_weights = Float64[_f(v) for v in t["prior_log_weights"]],
        max_components = 20, forget = _f(t["forget"]), gate_hook = gate_hook)
    f = sticky(f; k = k)
    f = gpdtails(f, k)
    return parade(f, k)
end

function _granger_trunk_state(s)
    for _ in 1:3
        s === nothing && return nothing
        s = s["base"]
    end
    return s
end

function _granger_gate(G::Dict, feats::Vector{Float64})
    hidden = Float64[]
    for (r, row) in enumerate(G["l1_weight"])
        acc = 0.0
        for j in eachindex(row)
            acc += _f(row[j]) * feats[j]
        end
        push!(hidden, tanh(_f(G["l1_bias"][r]) + acc))
    end
    out = Float64[]
    for (r, row) in enumerate(G["l2_weight"])
        acc = 0.0
        for j in eachindex(row)
            acc += _f(row[j]) * hidden[j]
        end
        push!(out, _f(G["l2_bias"][r]) + acc)
    end
    return out
end

function _granger_features(state::Dict, n::Int)
    z = state["z"]
    zs = z === nothing ? 0.0 : clamp(Float64(z), -8.0, 8.0)
    ts = _granger_trunk_state(state["learned"])
    if ts !== nothing && length(ts["log_w"]) == n
        lw = ts["log_w"]::Vector{Float64}
        m = maximum(lw)
        l = Float64[max(v - m, -20.0) / 20.0 for v in lw]
    else
        l = zeros(Float64, n)
    end
    return vcat(Float64[zs, state["repeat_frac"], state["abs_z_mean"]], l)
end

"""The two-component mixture granger issues."""
struct MixtureDist
    dists::Vector{Any}
    log_w::Vector{Float64}
    w::Vector{Float64}
end

function MixtureDist(dists::Vector, lw::Vector{Float64})
    m = maximum(lw)
    s = log(sum(exp.(lw .- m))) + m
    l = lw .- s
    return MixtureDist(Any[dists...], l, exp.(l))
end

function dist_logpdf(d::MixtureDist, x::Float64)::Float64
    t = Float64[d.log_w[i] + dist_logpdf(d.dists[i], x) for i in eachindex(d.dists)]
    m = maximum(t)
    m == -Inf && return -Inf
    return m + log(sum(exp.(t .- m)))
end

dist_cdf(d::MixtureDist, x::Float64)::Float64 = sum(d.w[i] * dist_cdf(d.dists[i], x) for i in eachindex(d.dists))
dist_mean(d::MixtureDist)::Float64 = sum(d.w[i] * dist_mean(d.dists[i]) for i in eachindex(d.dists))

function dist_var(d::MixtureDist)::Float64
    mu = dist_mean(d)
    return sum(d.w[i] * (dist_var(d.dists[i]) + (dist_mean(d.dists[i]) - mu)^2) for i in eachindex(d.dists))
end

dist_std(d::MixtureDist)::Float64 = sqrt(max(dist_var(d), 0.0))

function dist_quantile(d::MixtureDist, p::Float64; tol::Float64 = 1e-9, max_iter::Int = 200)::Float64
    0 < p < 1 || error("quantile needs 0 < p < 1, got $(p)")
    qs = Float64[dist_quantile(q, p) for q in d.dists]
    lo, hi = minimum(qs), maximum(qs)
    hi - lo < tol && return 0.5 * (lo + hi)
    for _ in 1:max_iter
        mid = 0.5 * (lo + hi)
        if dist_cdf(d, mid) < p
            lo = mid
        else
            hi = mid
        end
        hi - lo < tol && break
    end
    return 0.5 * (lo + hi)
end

function dist_crps(d::MixtureDist, x::Float64)::Float64
    n = 65
    q = Float64[dist_quantile(d, (i - 0.5) / n) for i in 1:n]
    t1 = sum(abs.(q .- x)) / n
    t2 = 2.0 * sum(q[i] * (2.0 * (i - 0.5) / n - 1.0) for i in 1:n) / n
    return t1 - 0.5 * t2
end

"""
    granger(params = granger_params("monthly"); k = 1)

laplace's trained twin mixed with laplace (horizon 1). Returns a skater
`f(y, state) -> (dists, state)`; `state` is a Dict of plain data.
"""
function granger(params::Dict = granger_params("monthly"); k::Int = 1)
    k == 1 || error("granger is trained at horizon 1 only")
    P = params
    pi = _f(P["pi_lap"])
    log_prior = [log(pi), log(1.0 - pi)]
    ref = laplace(; k = 1)
    current = Ref{Any}(nothing)           # the gate's offset for the current call only
    ex = _granger_expert(P, 1, () -> current[])
    n = length(P["candidates"])
    function _skater(y, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("lap" => nothing, "learned" => nothing, "pend" => nothing,
                "cum" => [0.0, 0.0], "t" => 0, "z" => nothing, "repeat_frac" => 0.0,
                "abs_z_mean" => 0.0, "n" => 0, "last_y" => nothing)
        end
        yv = y === nothing ? NaN : Float64(y)
        finite = isfinite(yv)
        if finite && state["pend"] !== nothing
            for j in 1:2
                lp = dist_logpdf(state["pend"][j], yv)
                state["cum"][j] += isfinite(lp) ? lp : -1e6
            end
        end
        current[] = _granger_gate(P["gate"], _granger_features(state, n))
        lrn = nothing
        try
            lrn, state["learned"] = ex(yv, state["learned"])
        finally
            current[] = nothing
        end
        lap, state["lap"] = ref(yv, state["lap"])
        state["pend"] = Any[lap[1], lrn[1]]
        pz = state["learned"]["z"]
        state["z"] = pz === nothing ? nothing : pz[1]
        if finite
            nn = state["n"] + 1
            rep = (state["last_y"] !== nothing && yv == state["last_y"]) ? 1.0 : 0.0
            state["repeat_frac"] += (rep - state["repeat_frac"]) / nn
            z0 = state["z"]
            if z0 !== nothing && isfinite(z0)
                state["abs_z_mean"] += (min(abs(z0), 8.0) - state["abs_z_mean"]) / nn
            end
            state["n"] = nn
            state["last_y"] = yv
        end
        out = MixtureDist(Any[lap[1], lrn[1]], log_prior .+ state["cum"])
        state["t"] += 1
        return Any[out], state
    end
    return _skater
end
