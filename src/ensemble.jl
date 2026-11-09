# Precision-weighted ensemble. Port of skaters/ensemble.py.

function precision_weighted_ensemble(skaters::Vector; k::Int = 1, floor::Float64 = 1e-6)
    n = length(skaters)
    @assert n > 0
    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}(
                "sub" => Any[nothing for _ in 1:n],
                "queues" => [[Float64[] for _ in 1:k] for _ in 1:n],
                "stats" => [[running_var_init() for _ in 1:k] for _ in 1:n],
            )
        end
        all_dists = Any[]
        sub = state["sub"]
        for i in 1:n
            di, sub[i] = skaters[i](y, sub[i])
            push!(all_dists, di)
        end
        queues = state["queues"]
        stats = state["stats"]
        for i in 1:n
            for h in 0:(k-1)
                q = queues[i][h+1]
                push!(q, dist_mean(all_dists[i][h+1]))
                if length(q) > h + 1
                    pred_mean = popfirst!(q)
                    error = y - pred_mean
                    stats[i][h+1] = running_var_update(stats[i][h+1], error)
                end
            end
        end
        combined = Any[]
        for h in 0:(k-1)
            weights = Float64[]
            for i in 1:n
                mse = running_mse_get(stats[i][h+1])
                # inf exactly when immature; a mature perfect forecaster (mse == 0)
                # gets the highest precision, capped at 1/floor (skaters#220).
                w = isfinite(mse) ? 1.0 / max(mse, floor) : floor
                push!(weights, w)
            end
            horizon_dists = Any[all_dists[i][h+1] for i in 1:n]
            push!(combined, dist_combine(horizon_dists, weights))
        end
        return combined, state
    end
    return _skater
end
