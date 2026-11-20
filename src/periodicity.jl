# Online periodicity detection via running autocorrelation.
# Port of skaters/periodicity.py. O(n_lags) per observation.

const _DEFAULT_LAGS = [2, 3, 4, 5, 6, 7, 12, 14, 24, 28, 30, 52, 60, 90, 168, 365]

# Returns a callable: scores, state = f(y, state), where scores is a vector
# of (lag, acf) tuples sorted by |acf| descending (stable sort, so ties keep
# lag order, exactly as Python's list.sort).
function period_detector(lags::Union{Nothing,Vector{Int}} = nothing;
                         alpha::Float64 = 0.01, min_observations::Int = 50)
    if lags === nothing
        lags = copy(_DEFAULT_LAGS)
    end
    max_lag = maximum(lags)
    # cross is keyed by the lag as a decimal string (the canonical layout).
    lag_keys = String[string(L) for L in lags]

    function _detect(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("buffer" => Float64[], "n" => 0,
                "mean" => 0.0, "var" => 0.0,
                "cross" => Dict{String,Any}(key => 0.0 for key in lag_keys))
        elseif state["cross"] isa AbstractVector
            # An older Julia checkpoint kept cross as a vector in lag order.
            state["cross"] = Dict{String,Any}(key => c for (key, c) in zip(lag_keys, state["cross"]))
        end
        buf = _fvec!(state, "buffer")
        push!(buf, y)
        state["n"] += 1

        diff = y - state["mean"]
        state["mean"] += alpha * diff
        state["var"] = (1 - alpha) * (state["var"] + alpha * diff * diff)

        mu = Float64(state["mean"])
        var = Float64(state["var"])
        cross = state["cross"]
        nb = length(buf)
        for (i, L) in enumerate(lags)
            if nb > L
                c = (y - mu) * (buf[nb-L] - mu)
                cross[lag_keys[i]] = (1 - alpha) * cross[lag_keys[i]] + alpha * c
            end
        end
        if nb > max_lag + 1
            popfirst!(buf)
        end

        if state["n"] < min_observations || var < 1e-12
            return Tuple{Int,Float64}[], state
        end
        scores = Tuple{Int,Float64}[]
        for (i, L) in enumerate(lags)
            if state["n"] > L
                acf = var > 0 ? cross[lag_keys[i]] / var : 0.0
                push!(scores, (L, acf))
            end
        end
        sort!(scores, by = x -> abs(x[2]), rev = true, alg = MergeSort)
        return scores, state
    end
    return _detect
end

# Extract the best periods from detector scores.
function top_periods(scores::Vector{Tuple{Int,Float64}};
                     threshold::Float64 = 0.3, max_periods::Int = 3)
    out = Int[]
    for s in scores[1:min(length(scores), max_periods)]
        if abs(s[2]) >= threshold
            push!(out, s[1])
        end
    end
    return out
end
