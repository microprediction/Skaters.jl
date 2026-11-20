# Terminal-leaf ensemble: mix for the mean, model the residual once.
# Port of skaters/terminal.py.

# Adaptive selection temperature (skaters#213/#215), opt-in.
const _PIT_DISPERSION_TARGET = 1.0 / 12.0   # E[(u - 0.5)^2] at perfect calibration
const _PIT_EWMA_DECAY = 0.98
const _LOSS_FAST_DECAY = 0.90
const _LOSS_SLOW_DECAY = 0.995
const _TEMP_AMBIG_EPS = 0.15 * _PIT_DISPERSION_TARGET  # |e_t| below this defers to the loss slope

function terminal_leaf_ensemble(skaters::Vector; leaf_fn = kk -> crps_leaf(kk),
                                k::Int = 1, learning_rate::Float64 = 0.5,
                                complexity_penalty::Float64 = 0.0,
                                depths::Union{Nothing,Vector} = nothing,
                                prior_log_weights::Union{Nothing,Vector} = nothing,
                                gate_hook = nothing,
                                max_components::Int = 20, forget::Float64 = 1.0,
                                adaptive_temperature::Bool = false,
                                temp_rho::Float64 = 0.03, temp_band::Float64 = 3.0)
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
                "mean_q" => Any[Float64[] for _ in 1:k],
                "adapt" => adaptive_temperature ? Dict{String,Any}(
                    "log_eta" => log(learning_rate),
                    "pit_ewma" => _PIT_DISPERSION_TARGET,
                    "loss_fast" => nothing,
                    "loss_slow" => nothing,
                    "pred_q" => Any[],      # pending issued h=1 predictive
                ) : nothing,
            )
        elseif !haskey(state, "adapt")
            state["adapt"] = nothing        # an older Julia checkpoint (no adaptive temperature)
        end
        all_dists = Any[]
        sub = state["sub"]
        for i in 1:n
            di, sub[i] = skaters[i](y, sub[i])
            push!(all_dists, di)
        end
        # Adaptive selection temperature: resolve the issued predictive from one
        # step ago against y, turn its PIT dispersion (and, in the dead band,
        # its own logpdf trend) into a heat or cool signal, and use the
        # resulting eta for this step's weight update.
        eta_t = learning_rate
        if adaptive_temperature
            ad = state["adapt"]
            pq = ad["pred_q"]
            if !isempty(pq)
                prev_pred = popfirst!(pq)
                lp_c = dist_logpdf(prev_pred, y)
                if !(lp_c >= -20.0)
                    lp_c = -20.0
                elseif lp_c > 20.0
                    lp_c = 20.0
                end
                u = dist_cdf(prev_pred, y)
                if !(0.0 <= u <= 1.0)
                    u = 0.5                 # degenerate or NaN guard: no signal
                end
                ad["pit_ewma"] = (_PIT_EWMA_DECAY * ad["pit_ewma"]
                                  + (1.0 - _PIT_EWMA_DECAY) * (u - 0.5)^2)
                if ad["loss_fast"] === nothing
                    ad["loss_fast"] = lp_c
                    ad["loss_slow"] = lp_c
                else
                    ad["loss_fast"] = _LOSS_FAST_DECAY * ad["loss_fast"] + (1.0 - _LOSS_FAST_DECAY) * lp_c
                    ad["loss_slow"] = _LOSS_SLOW_DECAY * ad["loss_slow"] + (1.0 - _LOSS_SLOW_DECAY) * lp_c
                end
                e_t = ad["pit_ewma"] - _PIT_DISPERSION_TARGET
                dloss_t = ad["loss_fast"] - ad["loss_slow"]
                if abs(e_t) > _TEMP_AMBIG_EPS
                    direction = e_t > 0.0 ? -1.0 : 1.0
                else
                    direction = dloss_t > 0.0 ? 1.0 : (dloss_t < 0.0 ? -1.0 : 0.0)
                end
                ad["log_eta"] += temp_rho * direction
                lo = log(learning_rate / temp_band)
                hi = log(learning_rate * temp_band)
                ad["log_eta"] = min(max(ad["log_eta"], lo), hi)
            end
            eta_t = exp(ad["log_eta"])
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
                log_w[i] = forget * log_w[i] + eta_t * lp - complexity_penalty * depths[i]
            end
            push!(q, all_dists[i][1])
        end
        # granger's gate shifts the combining weights for this step only; the
        # recursion above is untouched. laplace passes no hook.
        off = gate_hook === nothing ? nothing : gate_hook()
        lwc = off === nothing ? log_w : Float64[log_w[i] + off[i] for i in eachindex(log_w)]
        max_lw = maximum(lwc)
        w = Float64[exp(lw - max_lw) for lw in lwc]
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
        if adaptive_temperature
            push!(state["adapt"]["pred_q"], combined[1])
        end
        return combined, state
    end
    return _skater
end
