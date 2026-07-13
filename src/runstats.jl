# Online running statistics (Welford). Port of skaters/runstats.py.
# State is a plain Dict so it serialises trivially.

running_var_init() = Dict{String,Any}("n" => 0, "mean" => 0.0, "m2" => 0.0)

function running_var_update(state::Dict, x::Float64)
    n = state["n"] + 1
    delta = x - state["mean"]
    mean = state["mean"] + delta / n
    delta2 = x - mean
    m2 = state["m2"] + delta * delta2
    return Dict{String,Any}("n" => n, "mean" => mean, "m2" => m2)
end

function running_var_get(state::Dict)
    if state["n"] < 2
        return state["mean"], Inf
    end
    return state["mean"], state["m2"] / (state["n"] - 1)
end

function running_std_get(state::Dict)
    _, var = running_var_get(state)
    return isfinite(var) ? sqrt(var) : Inf
end

function running_mse_get(state::Dict)
    if state["n"] < 1
        return Inf
    end
    mean, var = running_var_get(state)
    if !isfinite(var)
        return Inf
    end
    return mean * mean + var
end
