# Online covariance estimation: running (Welford), EMA, and Ledoit-Wolf
# shrinkage toward identity. Port of skaters/cov/.
#
# API pattern (mirrors the Python tuple):
#   mean, cov, state = f(y, state)
# where y is a vector and cov is a flat row-major n*n vector.

function running_cov(y::Vector{Float64}, state::Union{Nothing,Dict})
    n = length(y)
    if state === nothing
        state = Dict{String,Any}("n" => 0, "mean" => zeros(n), "C" => zeros(n * n))
    end
    state["n"] += 1
    k = state["n"]::Int
    mean = state["mean"]::Vector{Float64}
    C = state["C"]::Vector{Float64}
    delta = [y[i] - mean[i] for i in 1:n]
    for i in 1:n
        mean[i] += delta[i] / k
    end
    delta2 = [y[i] - mean[i] for i in 1:n]
    for i in 1:n
        for j in 1:n
            C[(i-1)*n+j] += delta[i] * delta2[j]
        end
    end
    cov = k < 2 ? zeros(n * n) : [C[i] / (k - 1) for i in 1:(n*n)]
    return copy(mean), cov, state
end

function ema_cov(y::Vector{Float64}, state::Union{Nothing,Dict}, alpha::Float64 = 0.05)
    n = length(y)
    if state === nothing
        state = Dict{String,Any}("mean" => copy(y), "cov" => zeros(n * n), "n" => 1)
        return copy(y), zeros(n * n), state
    end
    mean = state["mean"]::Vector{Float64}
    cov = state["cov"]::Vector{Float64}
    state["n"] += 1
    delta = [y[i] - mean[i] for i in 1:n]
    for i in 1:n
        mean[i] += alpha * delta[i]
    end
    for i in 1:n
        for j in 1:n
            cov[(i-1)*n+j] = (1 - alpha) * (cov[(i-1)*n+j] + alpha * delta[i] * delta[j])
        end
    end
    return copy(mean), copy(cov), state
end

function ledoit_wolf_cov(y::Vector{Float64}, state::Union{Nothing,Dict},
                         alpha::Float64 = 0.05, shrinkage::Float64 = 0.5)
    n = length(y)
    if state === nothing
        corr = [i == j ? 1.0 : 0.0 for i in 1:n for j in 1:n]
        state = Dict{String,Any}("mean" => copy(y), "var" => zeros(n),
            "corr" => corr, "n" => 1)
        return copy(y), zeros(n * n), state
    end
    mean = state["mean"]::Vector{Float64}
    var = state["var"]::Vector{Float64}
    corr = state["corr"]::Vector{Float64}
    state["n"] += 1

    delta = [y[i] - mean[i] for i in 1:n]
    for i in 1:n
        mean[i] += alpha * delta[i]
    end
    delta2 = [y[i] - mean[i] for i in 1:n]
    for i in 1:n
        var[i] = (1 - alpha) * var[i] + alpha * delta[i] * delta2[i]
    end

    # Update correlations in standardized space, clamped to [-1, 1].
    s = [var[i] > 1e-16 ? sqrt(var[i]) : 1e-8 for i in 1:n]
    for i in 1:n
        for j in (i+1):n
            z_cross = (delta[i] / s[i]) * (delta[j] / s[j])
            idx = (i - 1) * n + j
            r = (1 - alpha) * corr[idx] + alpha * z_cross
            r = max(-1.0, min(1.0, r))
            corr[idx] = r
            corr[(j-1)*n+i] = r
        end
    end

    # Shrink correlation toward identity and reconstitute the covariance.
    shrunk = zeros(n * n)
    for i in 1:n
        for j in 1:n
            idx = (i - 1) * n + j
            shrunk[idx] = i == j ? var[i] : (1 - shrinkage) * corr[idx] * s[i] * s[j]
        end
    end
    return copy(mean), shrunk, state
end
