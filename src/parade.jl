# Prediction parade: online PIT/z calibration state. Port of skaters/parade.py.

const _STD_NORMAL = dist_gaussian(0.0, 1.0)
const _PEPS = 1e-12
const _NO_INFO_STD = 1e6

function parade(base, k::Int)
    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("base" => nothing, "pending" => Any[],
                "pit" => Any[nothing for _ in 1:k], "z" => Any[nothing for _ in 1:k], "skipped" => 0)
        end
        pend = state["pending"]
        n = length(pend)
        # Missing observation (the reference's parade.py): a non-finite y never
        # reaches the tree -- an EWMA fed NaN is poisoned for good. The base is
        # not stepped and the fan ages one step (the previous tick's horizon h+1
        # is this tick's h); after k gaps the h=k predictive is held.
        if !isfinite(y)
            state["skipped"] = get(state, "skipped", 0) + 1
            state["pit"] = Any[nothing for _ in 1:k]
            state["z"] = Any[nothing for _ in 1:k]
            n == 0 && return Any[dist_gaussian(0.0, _NO_INFO_STD) for _ in 1:k], state
            prev = pend[end]
            shifted = Any[prev[min(h + 1, length(prev))] for h in 1:k]
            push!(pend, shifted)
            length(pend) > k && popfirst!(pend)
            return shifted, state
        end
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
