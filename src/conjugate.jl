# Conjugation: change of reference frame for a skater.
# Port of skaters/conjugate.py. A transform is a 2-tuple (forward, inverse_k).

function conjugate(skater, transform, k::Int = 1)
    forward, inverse_k = transform
    function _conjugated(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("t_state" => nothing, "s_state" => nothing)
        end
        y_prime, state["t_state"] = forward(y, state["t_state"])
        dists_prime, state["s_state"] = skater(y_prime, state["s_state"])
        @assert length(dists_prime) == k
        dists = inverse_k(dists_prime, state["t_state"])
        return dists, state
    end
    return _conjugated
end
