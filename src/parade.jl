# Prediction parade: online PIT/z calibration state. Port of skaters/parade.py.

const _STD_NORMAL = dist_gaussian(0.0, 1.0)
const _PEPS = 1e-12

function parade(base, k::Int)
    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("base" => nothing, "pending" => Any[],
                "pit" => Any[nothing for _ in 1:k], "z" => Any[nothing for _ in 1:k])
        end
        pend = state["pending"]
        n = length(pend)
        pit = Any[nothing for _ in 1:k]
        z = Any[nothing for _ in 1:k]
        for m in 1:k
            if m <= n
                d = pend[n-m+1][m]
                u = dist_cdf(d, y)
                if !isfinite(u)
                    continue
                end
                u = min(max(u, _PEPS), 1.0 - _PEPS)
                pit[m] = u
                z[m] = dist_quantile(_STD_NORMAL, u)
            end
        end
        y_fed = y
        if isfinite(y_fed)
            y_fed = min(max(y_fed, -1e60), 1e60)
            if n > 0
                d1 = dist_body(pend[end][1])
                mp = dist_mean(d1)
                sp = dist_std(d1)
                if isfinite(mp) && isfinite(sp)
                    w = 1e12 * (1.0 + abs(mp) + sp)
                    y_fed = min(max(y_fed, mp - w), mp + w)
                end
            end
        end
        dists, state["base"] = base(y_fed, state["base"])
        push!(pend, collect(dists))
        if length(pend) > k
            popfirst!(pend)
        end
        state["pit"] = pit
        state["z"] = z
        return dists, state
    end
    return _skater
end
