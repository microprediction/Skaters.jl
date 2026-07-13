# Bayesian model averaging ensemble. Port of skaters/bayesian.py.

function bayesian_ensemble(skaters::Vector; k::Int = 1, learning_rate::Float64 = 0.5,
                           complexity_penalty::Float64 = 0.0,
                           depths::Union{Nothing,Vector} = nothing,
                           prior_log_weights::Union{Nothing,Vector} = nothing,
                           max_components::Int = 20)
    n = length(skaters)
    @assert n > 0
    @assert 0 < learning_rate <= 1
    @assert complexity_penalty >= 0
    depths = depths === nothing ? fill(0, n) : depths
    prior_log_weights = prior_log_weights === nothing ? fill(0.0, n) : prior_log_weights

    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}(
                "sub" => Any[nothing for _ in 1:n],
                "queues" => [[Any[] for _ in 1:k] for _ in 1:n],
                "log_w" => [[Float64(prior_log_weights[i]) for _ in 1:k] for i in 1:n],
                "n_obs" => 0,
            )
        end
        state["n_obs"] += 1
        all_dists = Any[]
        sub = state["sub"]
        for i in 1:n
            di, sub[i] = skaters[i](y, sub[i])
            push!(all_dists, di)
        end
        queues = state["queues"]
        log_w = state["log_w"]
        for i in 1:n
            for h in 0:(k-1)
                q = queues[i][h+1]
                push!(q, all_dists[i][h+1])
                if length(q) > h + 1
                    past_dist = popfirst!(q)
                    lp = dist_logpdf(past_dist, y)
                    if lp > 20.0
                        lp = 20.0
                    elseif !(lp >= -20.0)
                        lp = -20.0
                    end
                    log_w[i][h+1] += learning_rate * lp - complexity_penalty * depths[i]
                end
            end
        end
        combined = Any[]
        for h in 0:(k-1)
            log_ws = Float64[log_w[i][h+1] for i in 1:n]
            max_lw = maximum(log_ws)
            if isfinite(max_lw)
                weights = Float64[exp(lw - max_lw) for lw in log_ws]
            else
                weights = fill(1.0, n)
            end
            horizon_dists = Any[all_dists[i][h+1] for i in 1:n]
            dist = dist_combine(horizon_dists, weights)
            if length(dist.comps) > max_components
                dist = dist_prune(dist, max_components)
            end
            push!(combined, dist)
        end
        return combined, state
    end
    return _skater
end
