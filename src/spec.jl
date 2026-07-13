# Symbolic specification for skater pipelines. Port of skaters/spec.py.
#
# A spec is a plain Dict{String,Any} that fully describes how to build a
# skater. It serializes to JSON, compares, and materializes via
# spec_build(spec). Grammar (op key required):
#   leaf(k) | ema(alpha, k) | ensemble(k, skaters) | conjugate(skater, transform)
#   transforms: diff | frac(d, window) | std(alpha) | ema_t(alpha)

function spec_build(spec::Dict)
    op = spec["op"]
    if op == "leaf"
        return leaf(spec["k"])
    elseif op == "ema"
        return ema(spec["alpha"]; k = spec["k"])
    elseif op == "ensemble"
        subs = Any[spec_build(s) for s in spec["skaters"]]
        fl = get(spec, "floor", 1e-6)
        return precision_weighted_ensemble(subs; k = spec["k"], floor = fl)
    elseif op == "conjugate"
        inner = spec_build(spec["skater"])
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

# Compact float formatting for canonical names (Python's f"{x:.6g}").
function _spec_fmt(x::Real)::String
    if x == trunc(Int, x)
        return string(trunc(Int, x))
    end
    return string(round(Float64(x), sigdigits = 6))
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
