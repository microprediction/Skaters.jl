# Contract tests: determinism and checkpoint-resume equivalence.
# Mirrors the R port's tests/contract.R and the Rust crate's serde tests.
#
# Determinism: two independently constructed skaters fed the same series
# must emit bit-identical probe values at every step (compared through
# reinterpret(UInt64, x), so this is the IEEE bit pattern, not a tolerance).
#
# Checkpoint-resume: the state is pure data (closures live in the skater,
# never in the state), so serializing the state mid-stream with
# Serialization.serialize and resuming with a freshly built skater must
# reproduce the uninterrupted run exactly.
#
# Run: julia --project=. test/contract.jl
using Skaters
using Serialization

failures = 0

function check(cond::Bool, label::String)
    global failures
    if !cond
        failures += 1
        println("FAIL $label")
    end
end

mutable struct LCG
    s::UInt32
end
LCG(seed::Int) = LCG(UInt32(seed))

function nextu!(r::LCG)::Float64
    r.s = UInt32(1664525) * r.s + UInt32(1013904223)
    return Float64(r.s) / 4294967296.0
end

function gauss!(r::LCG)::Float64
    u = 0.0
    while u == 0.0
        u = nextu!(r)
    end
    v = 0.0
    while v == 0.0
        v = nextu!(r)
    end
    return sqrt(-2.0 * log(u)) * cos(2.0 * pi * v)
end

function make_series(n::Int, seed::Int)::Vector{Float64}
    r = LCG(seed)
    ys = zeros(n)
    lvl = 0.0
    for t in 1:n
        lvl += 0.02 + 0.3 * gauss!(r)
        ys[t] = lvl + sin(2.0 * pi * t / 12.0) + gauss!(r)
    end
    return ys
end

probe(dists) = Float64[v for d in dists for v in
    (dist_mean(d), dist_std(d), dist_logpdf(d, 0.3), dist_cdf(d, 0.3),
     dist_quantile(d, 0.1), dist_quantile(d, 0.9), dist_crps(d, 0.3))]

function run_probed(f, ys::Vector{Float64}; st = nothing, from::Int = 1)
    probes = Vector{Float64}[]
    for i in from:length(ys)
        dists, st = f(ys[i], st)
        push!(probes, probe(dists))
    end
    return probes, st
end

# Bit-pattern equality of two probe streams.
function bits_equal(a::Vector{Vector{Float64}}, b::Vector{Vector{Float64}})
    length(a) == length(b) || return false
    for i in eachindex(a)
        length(a[i]) == length(b[i]) || return false
        for j in eachindex(a[i])
            reinterpret(UInt64, a[i][j]) == reinterpret(UInt64, b[i][j]) || return false
        end
    end
    return true
end

# Structural deep equality for states: Dicts, vectors, tuples, Dist structs,
# and floats compared by bit pattern (so NaN == NaN, -0.0 != 0.0).
deepeq(a::Float64, b::Float64) = reinterpret(UInt64, a) == reinterpret(UInt64, b)
deepeq(a::Dict, b::Dict) =
    Set(keys(a)) == Set(keys(b)) && all(deepeq(a[k], b[k]) for k in keys(a))
deepeq(a::AbstractVector, b::AbstractVector) =
    length(a) == length(b) && all(deepeq(a[i], b[i]) for i in eachindex(a))
deepeq(a::Tuple, b::Tuple) =
    length(a) == length(b) && all(deepeq(a[i], b[i]) for i in eachindex(a))
deepeq(a::Dist, b::Dist) = deepeq(a.comps, b.comps)
# SplicedDist is a mutable struct (it can appear inside parade's pending
# buffer), so compare it field by field, not by identity.
deepeq(a::SplicedDist, b::SplicedDist) =
    all(deepeq(getfield(a, f), getfield(b, f)) for f in fieldnames(SplicedDist))
deepeq(a, b) = isequal(a, b)

roundtrip(x) = begin
    io = IOBuffer()
    serialize(io, x)
    seekstart(io)
    deserialize(io)
end

# --- determinism -----------------------------------------------------------
# Two separately constructed instances, one series, exact agreement at
# every step (not just the end).
for k in (1, 3)
    ys = make_series(800, 7)
    a_probes, a_state = run_probed(laplace(k = k), ys)
    b_probes, b_state = run_probed(laplace(k = k), ys)
    check(bits_equal(a_probes, b_probes), "determinism k=$k: probe streams differ")
    check(deepeq(a_state, b_state), "determinism k=$k: final states differ")
    println("ok   determinism k=$k (800 ticks, $(length(a_probes[1])) probes/tick)")
end

# --- checkpoint-resume -----------------------------------------------------
# Run to the midpoint, serialize the state, deserialize it, and continue
# with a FRESHLY BUILT skater. The resumed tail must equal the
# uninterrupted run's tail exactly. 1200 ticks puts the midpoint (600)
# beyond the GPD tail warm-up (500), so the splice and its excess buffers
# are part of the round-tripped state.
for k in (1, 3)
    ys = make_series(1200, 11)
    full_probes, full_state = run_probed(laplace(k = k), ys)

    first_probes, first_state = run_probed(laplace(k = k), ys[1:600])
    restored = roundtrip(first_state)
    check(deepeq(restored, first_state),
        "resume k=$k: state changed by serialize round-trip")

    resumed_probes, resumed_state = run_probed(laplace(k = k), ys; st = restored, from = 601)
    check(bits_equal(resumed_probes, full_probes[601:1200]),
        "resume k=$k: resumed probes differ from uninterrupted run")
    check(deepeq(resumed_state, full_state), "resume k=$k: final states differ")
    println("ok   checkpoint-resume k=$k (save at 600/1200)")
end

# Adaptive search: its state holds only recipes and plain data (the Python
# reference keeps live callables in the state; this port rebuilds them from
# recipes), so it must satisfy the same resume contract, across an
# expansion boundary (expand at 100 and 200 with the save at 150).
let
    ys = make_series(300, 13)
    full_probes, full_state = run_probed(adaptive_search(k = 1), ys)
    first_probes, first_state = run_probed(adaptive_search(k = 1), ys[1:150])
    restored = roundtrip(first_state)
    resumed_probes, resumed_state = run_probed(adaptive_search(k = 1), ys; st = restored, from = 151)
    check(bits_equal(resumed_probes, full_probes[151:300]),
        "search resume: probes differ from uninterrupted run")
    check(deepeq(resumed_state, full_state), "search resume: final states differ")
    println("ok   checkpoint-resume adaptive_search (save at 150/300)")
end

# The splice must actually be active in the resumed run at k=1, otherwise
# the checkpoint test is not exercising the tail state.
let
    ys = make_series(1200, 11)
    f = laplace(k = 1)
    st = nothing
    d = nothing
    for y in ys
        dists, st = f(y, st)
        d = dists[1]
    end
    check(d isa SplicedDist, "resume: splice not active at 1200 ticks")
    println("ok   splice active across checkpoint")
end

if failures > 0
    println("CONTRACT FAILED: $failures violation(s)")
    error("CONTRACT FAILED: $failures violation(s)")
end
println("CONTRACT OK (determinism, checkpoint-resume)")
