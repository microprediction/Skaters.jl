# User-facing API: the laplace forecaster. Port of skaters/api.py.

function _objective_leaf(objective::String, scale_alpha::Float64 = 0.03)
    if objective == "crps"
        return kk -> crps_leaf(kk; scale_alpha = scale_alpha)
    elseif objective == "likelihood"
        return kk -> scale_mixture_leaf(kk; scale_alpha = scale_alpha)
    end
    error("objective must be 'crps' or 'likelihood', got $(objective)")
end

# Candidate population (shared by all search policies). Returns (candidates,
# depths). Order is semantic: the ensemble weights and combines in this order.
function _build_candidates(k::Int, leaf_fn = leaf)
    candidates = Any[]
    depths = Int[]

    push!(candidates, leaf_fn(k)); push!(depths, 0)

    for alpha in (0.01, 0.05, 0.1, 0.3)
        push!(candidates, conjugate(leaf_fn(k), ema_transform(alpha), k)); push!(depths, 1)
    end

    push!(candidates, conjugate(leaf_fn(k), difference(), k)); push!(depths, 1)

    for (a, s) in ((0.05, 0.01), (0.01, 0.002), (0.002, 0.001), (0.0005, 0.0002))
        push!(candidates, conjugate(leaf_fn(k), drift(a, s), k)); push!(depths, 1)
    end

    for a in (0.05, 0.1, 0.3)
        push!(candidates, conjugate(leaf_fn(k), theta(a), k)); push!(depths, 1)
    end

    push!(candidates, conjugate(leaf_fn(k), ar(1), k)); push!(depths, 1)
    push!(candidates, conjugate(leaf_fn(k), ar(2; decay = 1.0), k)); push!(depths, 1)

    for (a, b) in ((0.1, 0.02), (0.1, 0.05), (0.3, 0.1))
        push!(candidates, conjugate(leaf_fn(k), holt_linear(a, b), k)); push!(depths, 1)
    end

    for period in (7, 12, 24)
        push!(candidates, conjugate(leaf_fn(k), seasonal_difference(period), k)); push!(depths, 1)
    end

    for period in (7, 12, 24)
        for alpha in (0.05, 0.1)
            push!(candidates, conjugate(conjugate(leaf_fn(k), ema_transform(alpha), k),
                seasonal_difference(period), k)); push!(depths, 2)
        end
    end

    for alpha in (0.05, 0.1, 0.3)
        push!(candidates, conjugate(conjugate(leaf_fn(k), ema_transform(alpha), k),
            difference(), k)); push!(depths, 2)
    end

    for alpha in (0.05, 0.1)
        push!(candidates, conjugate(conjugate(leaf_fn(k), ema_transform(alpha), k),
            standardize(), k)); push!(depths, 2)
    end

    for d in (0.2, 0.4)
        push!(candidates, conjugate(conjugate(leaf_fn(k), ema_transform(0.1), k),
            fractional_difference(d, 30), k)); push!(depths, 2)
    end

    for (a_drift, s_drift) in ((0.002, 0.001), (0.0005, 0.0002))
        for a_ema in (0.05, 0.1)
            push!(candidates, conjugate(conjugate(leaf_fn(k), ema_transform(a_ema), k),
                drift(a_drift, s_drift), k)); push!(depths, 2)
        end
    end

    push!(candidates, conjugate(conjugate(leaf_fn(k), holt_linear(0.1, 0.05), k),
        drift(0.001, 0.0005), k)); push!(depths, 2)

    push!(candidates, conjugate(conjugate(leaf_fn(k), ema_transform(0.1), k),
        garch(), k)); push!(depths, 2)

    push!(candidates, conjugate(conjugate(leaf_fn(k), ema_transform(0.1), k),
        power_transform(0.5), k)); push!(depths, 2)

    _fast_trackers() = Any[ema_transform(0.3), ema_transform(0.5),
        holt_linear(0.4, 0.2), ar(1), drift(0.05, 0.01), difference()]

    for scale_alpha in (0.02, 0.05)
        for tracker in _fast_trackers()
            slow_scale = standardize(scale_alpha)
            push!(candidates, conjugate(conjugate(leaf_fn(k), slow_scale, k), tracker, k))
            push!(depths, 2)
        end
    end

    for L in (0.0, 0.5)
        for inner_tx in (difference(), ema_transform(0.1))
            push!(candidates, conjugate(conjugate(leaf_fn(k), inner_tx, k),
                yeo_johnson(L), k)); push!(depths, 2)
        end
    end

    if k > 1
        for L in (0.0, 0.5)
            for kappa in (0.03, 0.1, 0.3)
                push!(candidates, conjugate(conjugate(leaf_fn(k), ou_transform(kappa, 0.02), k),
                    yeo_johnson(L), k)); push!(depths, 2)
            end
        end
    end

    return candidates, depths
end

function _laplace_single_scale(k::Int, objective::String, use_sticky::Bool, leaf_override,
                               scale_alpha::Float64)
    candidates, depths = _build_candidates(k)
    lf = leaf_override !== nothing ? leaf_override : _objective_leaf(objective, scale_alpha)
    f = terminal_leaf_ensemble(candidates; k = k, leaf_fn = lf, learning_rate = 0.8,
        complexity_penalty = 0.005, depths = depths, max_components = 20, forget = 0.99)
    if use_sticky
        f = sticky(f; k = k)
    end
    return f
end

function laplace(; k::Int = 1, objective::String = "crps", sticky::Bool = true,
                 leaf = nothing, scales::Union{Nothing,Vector{Int}} = nothing,
                 scale_alpha::Float64 = 0.03, tails::String = "gpd")
    @assert tails in ("gpd", "gaussian")
    use_sticky = sticky
    leaf_override = leaf
    f = multiscale(kk -> _laplace_single_scale(kk, objective, use_sticky, leaf_override, scale_alpha),
        k; scales = scales)
    if tails == "gpd"
        f = gpdtails(f, k)
    end
    f = parade(f, k)
    return f
end
