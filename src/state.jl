# Checkpoint and restore for skater state: the fold contract, in the
# reference's canonical layout (skaters docs/STATE.md, src/skaters/state.py).
#
#     plain = state_to_json(state)             # Dist -> Dict("components" => ...)
#     text  = JSON.json(plain)
#     back  = state_from_json(JSON.parse(text)) # back to live state, no renormalising
#
# State holds only nothing, Bools, Strings, finite numbers, lists, Dicts with
# String keys, and distributions. A Dist is {"components": [[w, mean, std], ...]};
# a SplicedDist is {"spliced": true, "body": {...}, "t_lo": ..., ...}. Keys are
# the reference's snake_case names, an integer-indexed table uses decimal
# string keys, and every queue is a list, oldest first. A checkpoint written
# here loads in Python, JS or any other port that reads this layout, and the
# other way round.

function _state_offend(path::String, what::String)
    throw(ArgumentError("skater state is not plain data at $(isempty(path) ? "<root>" : path): $what"))
end

_dist_to_plain(d::Dist) = Dict{String,Any}("components" => Any[Any[w, m, s] for (w, m, s) in d.comps])

function _dist_to_plain(d::SplicedDist)
    return Dict{String,Any}("spliced" => true, "body" => _dist_to_plain(d.body),
                            "t_lo" => d.t_lo, "t_up" => d.t_up,
                            "zeta_lo" => d.zeta_lo, "zeta_up" => d.zeta_up,
                            "g_lo" => d.g_lo, "s_lo" => d.s_lo,
                            "g_up" => d.g_up, "s_up" => d.s_up)
end

function _state_walk(v, path::String, build::Bool)
    if v === nothing || v isa Bool || v isa AbstractString
        return v
    elseif v isa Integer
        return Int(v)
    elseif v isa AbstractFloat
        isfinite(v) || _state_offend(path, "non-finite float $v (JSON has no literal for it)")
        return Float64(v)
    elseif v isa Dist || v isa SplicedDist
        return build ? _dist_to_plain(v) : v
    elseif v isa Union{AbstractVector,Tuple}
        out = build ? Any[] : nothing
        for (i, x) in enumerate(v)
            r = _state_walk(x, "$path[$(i-1)]", build)
            build && push!(out, r)
        end
        return build ? out : v
    elseif v isa AbstractDict
        out = build ? Dict{String,Any}() : nothing
        for (k, x) in v
            k isa AbstractString ||
                _state_offend(path, "a dict key $(repr(k)) of type $(typeof(k)) (JSON keys are strings)")
            r = _state_walk(x, "$path.$k", build)
            build && (out[String(k)] = r)
        end
        return build ? out : v
    elseif v isa Function
        _state_offend(path, "a function (closures belong in the skater, not in state)")
    end
    _state_offend(path, "an instance of $(typeof(v)) (not plain data)")
end

# Throw ArgumentError, naming the path, if state holds anything that would
# not survive JSON. Returns the state unchanged.
assert_plain_state(state) = (_state_walk(state, "", false); state)

# Plain-data image of state: every Dist and SplicedDist becomes its dict form,
# tuples become lists. Throws ArgumentError, naming the path, on anything
# else non-plain. Also accepts a list of dists (a skater output).
state_to_json(state) = _state_walk(state, "", true)

_is_dist_dict(v::AbstractDict) =
    get(v, "spliced", nothing) === true ||
    (length(v) == 1 && haskey(v, "components") && v["components"] isa AbstractVector)

function _dist_from_plain(v::AbstractDict)
    if get(v, "spliced", nothing) === true
        return SplicedDist(_dist_from_plain(v["body"]),
                           Float64(v["t_lo"]), Float64(v["t_up"]),
                           Float64(v["zeta_lo"]), Float64(v["zeta_up"]),
                           Float64(v["g_lo"]), Float64(v["s_lo"]),
                           Float64(v["g_up"]), Float64(v["s_up"]))
    end
    comps = NTuple{3,Float64}[(Float64(c[1]), Float64(c[2]), Float64(c[3])) for c in v["components"]]
    # Weights that already sum to one (within 1e-9 of the compensated sum)
    # are kept bit for bit; renormalising could move the last bit.
    if abs(fsum(c[1] for c in comps) - 1.0) <= 1e-9
        return Dist(comps, Val(:trusted))
    end
    return Dist(comps)
end

# A JSON array becomes the narrowest vector Julia's skaters work with: all
# floats (ints allowed, since a port without an int type writes 3 for 3.0)
# a Vector{Float64}, all ints a Vector{Int}, floats with nulls a
# Vector{Union{Nothing,Float64}}, anything else a Vector{Any}. Skaters
# coerce on read where they need a particular element type.
function _vector_from_plain(xs::AbstractVector)
    items = Any[state_from_json(x) for x in xs]
    isempty(items) && return Any[]
    if all(x -> x isa Int, items)
        return Int[x for x in items]
    elseif all(x -> x isa Real && !(x isa Bool), items)
        return Float64[x for x in items]
    elseif all(x -> x === nothing || (x isa Real && !(x isa Bool)), items) &&
           any(x -> x isa AbstractFloat, items)
        return Union{Nothing,Float64}[x === nothing ? nothing : Float64(x) for x in items]
    end
    return items
end

# Inverse of state_to_json: a {"components": [...]} dict becomes a Dist
# without renormalising, a {"spliced": true, ...} dict a SplicedDist, other
# dicts a Dict{String,Any}, lists vectors (see _vector_from_plain).
function state_from_json(obj)
    if obj isa AbstractVector
        return _vector_from_plain(obj)
    elseif obj isa AbstractDict
        _is_dist_dict(obj) && return _dist_from_plain(obj)
        return Dict{String,Any}(String(k) => state_from_json(x) for (k, x) in obj)
    end
    return obj
end

# --- typed reads of restored state ---

# Read state[key] as the vector type the skater works with, converting a
# restored vector (or an empty Any[]) in place.
function _fvec!(state::AbstractDict, key)
    v = state[key]
    v isa Vector{Float64} && return v
    w = Float64[Float64(x) for x in v]
    state[key] = w
    return w
end

function _ivec!(state::AbstractDict, key)
    v = state[key]
    v isa Vector{Int} && return v
    w = Int[Int(x) for x in v]
    state[key] = w
    return w
end

function _ofvec!(state::AbstractDict, key)
    v = state[key]
    v isa Vector{Union{Nothing,Float64}} && return v
    w = Union{Nothing,Float64}[x === nothing ? nothing : Float64(x) for x in v]
    state[key] = w
    return w
end
