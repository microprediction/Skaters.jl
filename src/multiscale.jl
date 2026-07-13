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

    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}(
                "t" => 0,
                "phase" => Dict{Int,Any}(s => Any[nothing for _ in 1:s] for s in scales),
                "pending" => Dict{Int,Any}(s => Any[nothing for _ in 1:s] for s in scales),
                "latest" => Dict{Int,Any}(),
                "score" => Dict{Int,Any}(s => nothing for s in scales),
            )
        end
        t = state["t"]
        phase = state["phase"]
        pending = state["pending"]
        latest = state["latest"]
        score = state["score"]
        for s in scales
            ph = t % s  # 0-based phase index
            prev = pending[s][ph+1]
            if prev !== nothing
                lp = max(dist_logpdf(prev, y), _LOGPDF_FLOOR)
                m = score[s]
                score[s] = m === nothing ? lp : forget * m + (1.0 - forget) * lp
            end
            dists, phase[s][ph+1] = subs[s](y, phase[s][ph+1])
            pending[s][ph+1] = dists[1]
            latest[s] = dists
        end
        state["t"] = t + 1

        present = [m for m in values(score) if m !== nothing]
        top = isempty(present) ? 0.0 : maximum(present)
        out = Any[]
        for h in 1:k
            fcs = Any[]
            wts = Float64[]
            for s in scales
                if s > h || !haskey(latest, s)
                    continue
                end
                j = max(1, trunc(Int, h / s + 0.5))
                dists = latest[s]
                if j > length(dists)
                    continue
                end
                push!(fcs, dists[j])
                m = score[s]
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
