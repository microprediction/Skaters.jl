# Symbolic specification for skater pipelines. Port of skaters/spec.py.
#
# A spec is a plain Dict{String,Any} that fully describes how to build a
# skater. It serializes to JSON, compares, and materializes via
# spec_build(spec). Grammar (op key required):
#   leaf(k) | ema(alpha, k) | ensemble(k, skaters) | conjugate(skater, transform)
#   transforms: diff | frac(d, window) | std(alpha) | ema_t(alpha)

function spec_build(spec::Dict)
    return _spec_build(spec)
end

# The reference's name for spec_build.
build(spec::Dict) = spec_build(spec)

function _spec_build(spec::Dict)
    op = spec["op"]
    if op == "leaf"
        return leaf(spec["k"])
    elseif op == "ema"
        return ema(spec["alpha"]; k = spec["k"])
    elseif op == "ensemble"
        subs = Any[_spec_build(s) for s in spec["skaters"]]
        fl = get(spec, "floor", 1e-6)
        return precision_weighted_ensemble(subs; k = spec["k"], floor = fl)
    elseif op == "conjugate"
        inner = _spec_build(spec["skater"])
        tr = _spec_build_transform(spec["transform"])
        return conjugate(inner, tr, _spec_infer_k(spec["skater"]))
    end
    error("Unknown op: $op")
end

function _spec_build_transform(spec::Dict)
    op = spec["op"]
    op == "diff" && return difference()
    op == "frac" && return fractional_difference(spec["d"], get(spec, "window", 50))
    op == "std" && return standardize(get(spec, "alpha", 0.05))
    op == "ema_t" && return ema_transform(spec["alpha"])
    error("Unknown transform op: $op")
end

function _spec_infer_k(spec::Dict)::Int
    haskey(spec, "k") && return spec["k"]
    haskey(spec, "skater") && return _spec_infer_k(spec["skater"])
    haskey(spec, "skaters") && return _spec_infer_k(spec["skaters"][1])
    error("Cannot infer k from spec")
end

# Python's spelling of a float. `digits` and `dexp` are the decimal digits
# and exponent of x (x = 0.d1d2... * 10^dexp). With g = nothing this is
# repr(x), the shortest round-trip form: positional when -4 <= exponent < 16,
# always with a fractional part, else "1e-05" style. With g = 6 it is
# format(x, ".6g"): rounded to six significant digits, trailing zeros
# stripped, scientific when the exponent is below -4 or at least 6.
function _py_float(x::Float64; g::Union{Nothing,Int} = nothing)::String
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Infinity" : "-Infinity"
    x == 0.0 && return g === nothing ? (signbit(x) ? "-0.0" : "0.0") : (signbit(x) ? "-0" : "0")
    neg = x < 0
    r = g === nothing ? abs(x) : round(abs(x), sigdigits = g)
    # Julia's string(r) is the shortest round-trip decimal; read its digits.
    t = string(r)
    mant, ex = occursin('e', t) ? (split(t, 'e')[1], parse(Int, split(t, 'e')[2])) : (t, 0)
    ip, fp = occursin('.', mant) ? split(mant, '.') : (mant, "")
    digits = ip * fp
    point = length(ip) + ex                 # position of the decimal point in `digits`
    nz = findfirst(c -> c != '0', digits)
    digits = digits[nz:end]
    point -= nz - 1
    digits = rstrip(digits, '0')
    isempty(digits) && (digits = "0")
    e = point - 1                           # scientific exponent
    hi = g === nothing ? 16 : g
    if -4 <= e < hi
        if point <= 0
            body = "0." * "0"^(-point) * digits
        elseif point >= length(digits)
            body = digits * "0"^(point - length(digits)) * (g === nothing ? ".0" : "")
        else
            body = digits[1:point] * "." * digits[point+1:end]
        end
    else
        m = length(digits) > 1 ? digits[1:1] * "." * digits[2:end] : digits
        body = m * "e" * (e < 0 ? "-" : "+") * lpad(string(abs(e)), 2, '0')
    end
    return neg ? "-" * body : body
end

# Compact float formatting for canonical names (Python's _fmt: an integral
# value prints as an integer, anything else as f"{x:.6g}").
function _spec_fmt(x::Real)::String
    if x == trunc(Int, x)
        return string(trunc(Int, x))
    end
    return _py_float(Float64(x); g = 6)
end

function spec_name(spec::Dict)::String
    op = spec["op"]
    op == "leaf" && return "leaf"
    op == "ema" && return "ema($(_spec_fmt(spec["alpha"])))"
    if op == "ensemble"
        inner = join([spec_name(s) for s in spec["skaters"]], ",")
        return "ensemble($inner)"
    end
    if op == "conjugate"
        return "$(_spec_transform_name(spec["transform"]))|$(spec_name(spec["skater"]))"
    end
    error("Unknown op: $op")
end

function _spec_transform_name(spec::Dict)::String
    op = spec["op"]
    op == "diff" && return "diff"
    if op == "frac"
        w = get(spec, "window", 50)
        w == 50 && return "frac($(_spec_fmt(spec["d"])))"
        return "frac($(_spec_fmt(spec["d"])),w=$w)"
    end
    op == "std" && return "std($(_spec_fmt(get(spec, "alpha", 0.05))))"
    op == "ema_t" && return "ema_t($(_spec_fmt(spec["alpha"])))"
    error("Unknown transform op: $op")
end

# Constructors (convenience, mirror spec.py).
leaf_spec(k::Int = 1) = Dict{String,Any}("op" => "leaf", "k" => k)
ema_spec(alpha::Float64 = 0.05, k::Int = 1) =
    Dict{String,Any}("op" => "ema", "alpha" => alpha, "k" => k)
ensemble_spec(specs...; k::Int = 1) =
    Dict{String,Any}("op" => "ensemble", "k" => k, "skaters" => Any[specs...])
conjugate_spec(skater_spec::Dict, transform_spec::Dict) =
    Dict{String,Any}("op" => "conjugate", "skater" => skater_spec,
        "transform" => transform_spec)
diff_spec() = Dict{String,Any}("op" => "diff")
frac_spec(d::Float64 = 0.4, window::Int = 50) =
    Dict{String,Any}("op" => "frac", "d" => d, "window" => window)
std_spec(alpha::Float64 = 0.05) = Dict{String,Any}("op" => "std", "alpha" => alpha)
ema_t_spec(alpha::Float64 = 0.05) = Dict{String,Any}("op" => "ema_t", "alpha" => alpha)

# --- serialization (mirrors spec.py's to_json / from_json) ---

# Python writes json.dumps(spec, separators=(",", ":")) of a dict whose keys
# sit in constructor insertion order. Julia's Dict is unordered, so keys are
# written in that same canonical order (op, alpha, d, window, k, skaters,
# floor, skater, transform, then any others sorted), and floats in Python's
# repr, so a spec serialized here is byte-identical to the reference's.
const _SPEC_KEY_ORDER = ("op", "alpha", "d", "window", "k", "skaters", "floor", "skater", "transform")

function _spec_key_rank(key::AbstractString)
    i = findfirst(==(key), _SPEC_KEY_ORDER)
    return i === nothing ? (length(_SPEC_KEY_ORDER) + 1, key) : (i, "")
end

function _json_write(io::IO, x)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        print(io, _py_float(Float64(x)))
    elseif x isa AbstractString
        JSON.print(io, String(x))
    elseif x isa AbstractDict
        print(io, "{")
        for (i, key) in enumerate(sort!(collect(keys(x)); by = k -> _spec_key_rank(string(k))))
            i > 1 && print(io, ",")
            JSON.print(io, string(key))
            print(io, ":")
            _json_write(io, x[key])
        end
        print(io, "}")
    elseif x isa Union{AbstractVector,Tuple}
        print(io, "[")
        for (i, v) in enumerate(x)
            i > 1 && print(io, ",")
            _json_write(io, v)
        end
        print(io, "]")
    else
        throw(ArgumentError("cannot serialize $(typeof(x)) in a spec"))
    end
end

# Serialize a spec to compact JSON, byte-identical to the reference.
function to_json(spec::AbstractDict)::String
    io = IOBuffer()
    _json_write(io, spec)
    return String(take!(io))
end

# Deserialize a spec from JSON (a spec written by either port).
from_json(s::AbstractString) = JSON.parse(s)
