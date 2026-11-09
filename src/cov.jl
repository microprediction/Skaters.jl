# Online covariance estimation: running (Welford), EMA, and Ledoit-Wolf
# shrinkage toward the diagonal. Port of skaters/cov/.
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

# Shrinks the ema_cov covariance toward its own diagonal: (1-rho)*C + rho*diag(C)
# is PSD whenever C is (skaters#224). The inner update is delegated to ema_cov.
function ledoit_wolf_cov(y::Vector{Float64}, state::Union{Nothing,Dict},
                         alpha::Float64 = 0.05, shrinkage::Float64 = 0.5)
    n = length(y)
    inner_state = state !== nothing ? state["inner"] : nothing
    mean, cov, inner_state = ema_cov(y, inner_state, alpha)
    state = Dict{String,Any}("inner" => inner_state)
    shrunk = zeros(n * n)
    for i in 1:n
        for j in 1:n
            idx = (i - 1) * n + j
            shrunk[idx] = i == j ? cov[idx] : (1.0 - shrinkage) * cov[idx]
        end
    end
    return mean, shrunk, state
end
