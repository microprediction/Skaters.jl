# Terminal-leaf ensemble: mix for the mean, model the residual once.
# Port of skaters/terminal.py.

function terminal_leaf_ensemble(skaters::Vector; leaf_fn = kk -> crps_leaf(kk),
                                k::Int = 1, learning_rate::Float64 = 0.5,
                                complexity_penalty::Float64 = 0.0,
                                depths::Union{Nothing,Vector} = nothing,
                                prior_log_weights::Union{Nothing,Vector} = nothing,
                                max_components::Int = 20, forget::Float64 = 1.0)
    n = length(skaters)
    @assert n > 0
    depths = depths === nothing ? fill(0, n) : depths
    prior = prior_log_weights === nothing ? fill(0.0, n) : prior_log_weights
    # One terminal leaf per horizon, held in the closure (never in state).
    tleafs = Any[leaf_fn(1) for _ in 1:k]

    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}(
                "sub" => Any[nothing for _ in 1:n],
                "qdist" => [Any[] for _ in 1:n],
                "log_w" => Float64[Float64(prior[i]) for i in 1:n],
                "leaf_state" => Any[nothing for _ in 1:k],
                "leaf_pred" => Any[nothing for _ in 1:k],
                "mean_q" => [Float64[] for _ in 1:k],
            )
        end
        all_dists = Any[]
        sub = state["sub"]
        for i in 1:n
            di, sub[i] = skaters[i](y, sub[i])
            push!(all_dists, di)
        end
        log_w = state["log_w"]
        qdist = state["qdist"]
        for i in 1:n
            q = qdist[i]
            if !isempty(q)
                lp = dist_logpdf(popfirst!(q), y)
                if lp > 20.0
                    lp = 20.0
                elseif !(lp >= -20.0)
                    lp = -20.0
                end
                log_w[i] = forget * log_w[i] + learning_rate * lp - complexity_penalty * depths[i]
            end
            push!(q, all_dists[i][1])
        end
        max_lw = maximum(log_w)
        w = Float64[exp(lw - max_lw) for lw in log_w]
        tot = fsum(w)

        leaf_state = state["leaf_state"]
        leaf_pred = state["leaf_pred"]
        mean_q = state["mean_q"]
        combined = Any[]
        for h in 0:(k-1)
            mu_h = fsum(w[i] * dist_mean(all_dists[i][h+1]) for i in 1:n) / tot
            mq = mean_q[h+1]
            if length(mq) >= h + 1
                r = y - popfirst!(mq)
                ld, leaf_state[h+1] = tleafs[h+1](r, leaf_state[h+1])
                leaf_pred[h+1] = ld[1]
            end
            if leaf_pred[h+1] !== nothing
                pred = dist_shift(leaf_pred[h+1], mu_h)
            else
                pred = dist_combine(Any[all_dists[i][h+1] for i in 1:n], w)
                if length(pred.comps) > max_components
                    pred = dist_prune(pred, max_components)
                end
            end
            push!(combined, pred)
            push!(mq, mu_h)
        end
        return combined, state
    end
    return _skater
end
