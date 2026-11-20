# Multi-scale ensemble over decimated clocks. Port of skaters/multiscale.py.

const _LOGPDF_FLOOR = -20.0

function multiscale(base, k::Int; scales::Union{Nothing,Vector{Int}} = nothing,
                    forget::Float64 = 0.99, max_components::Int = 20)
    if scales === nothing
        scales = sort(collect(Set([1, Int(ceil(sqrt(k))), k])))
    end
    scales = sort(collect(Set([Int(s) for s in scales if 1 <= s <= k])))
    @assert !isempty(scales) && scales[1] == 1
    if scales == [1]
        return base(k)
    end
    subs = Dict{Int,Any}(s => base(max(1, Int(ceil(k / s)))) for s in scales)
    # State tables are keyed by the stride as a decimal string ("1", "3"), the
    # reference's canonical layout.
    keys_ = Dict{Int,String}(s => string(s) for s in scales)

    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}(
                "t" => 0,
                "phase" => Dict{String,Any}(keys_[s] => Any[nothing for _ in 1:s] for s in scales),
                "pending" => Dict{String,Any}(keys_[s] => Any[nothing for _ in 1:s] for s in scales),
                "latest" => Dict{String,Any}(),
                "score" => Dict{String,Any}(keys_[s] => nothing for s in scales),
            )
        elseif !haskey(state["score"], keys_[scales[1]])
            # An older Julia checkpoint keyed these tables by Int.
            for name in ("phase", "pending", "latest", "score")
                state[name] = Dict{String,Any}(string(s) => v for (s, v) in state[name])
            end
        end
        t = state["t"]
        phase = state["phase"]
        pending = state["pending"]
        latest = state["latest"]
        score = state["score"]
        for s in scales
            ph = t % s  # 0-based phase index
            prev = pending[keys_[s]][ph+1]
            if prev !== nothing
                lp = max(dist_logpdf(prev, y), _LOGPDF_FLOOR)
                m = score[keys_[s]]
                score[keys_[s]] = m === nothing ? lp : forget * m + (1.0 - forget) * lp
            end
            dists, phase[keys_[s]][ph+1] = subs[s](y, phase[keys_[s]][ph+1])
            pending[keys_[s]][ph+1] = dists[1]
            latest[keys_[s]] = dists
        end
        state["t"] = t + 1

        present = [m for m in values(score) if m !== nothing]
        top = isempty(present) ? 0.0 : maximum(present)
        out = Any[]
        for h in 1:k
            fcs = Any[]
            wts = Float64[]
            for s in scales
                if s > h || !haskey(latest, keys_[s])
                    continue
                end
                j = max(1, trunc(Int, h / s + 0.5))
                dists = latest[keys_[s]]
                if j > length(dists)
                    continue
                end
                push!(fcs, dists[j])
                m = score[keys_[s]]
                push!(wts, exp((m === nothing ? top : m) - top))
            end
            if length(fcs) == 1
                push!(out, fcs[1])
            else
                push!(out, dist_prune(dist_combine(fcs, wts), max_components))
            end
        end
        return out, state
    end
    return _skater
end
