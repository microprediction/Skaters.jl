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

    function _detect(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("buffer" => Float64[], "n" => 0,
                "mean" => 0.0, "var" => 0.0, "cross" => zeros(length(lags)))
        end
        buf = state["buffer"]::Vector{Float64}
        push!(buf, y)
        state["n"] += 1

        diff = y - state["mean"]
        state["mean"] += alpha * diff
        state["var"] = (1 - alpha) * (state["var"] + alpha * diff * diff)

        mu = state["mean"]::Float64
        var = state["var"]::Float64
        cross = state["cross"]::Vector{Float64}
        nb = length(buf)
        for (i, L) in enumerate(lags)
            if nb > L
                c = (y - mu) * (buf[nb-L] - mu)
                cross[i] = (1 - alpha) * cross[i] + alpha * c
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
                acf = var > 0 ? cross[i] / var : 0.0
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
