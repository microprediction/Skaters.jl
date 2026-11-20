# Julia-written checkpoints for the reference to resume, the other direction
# of test/state_resume.jl. For every scenario of test/state_scenarios.jl, run
# Julia over the series of parity/state_vectors.json and write state_to_json
# at the same checkpoints. parity/check_julia_checkpoints.py then loads each
# one in Python, steps 20 and compares with Python's own continuation.
#
#   julia --project=. parity/write_julia_checkpoints.jl /tmp/julia_checkpoints.json
#   PYTHONPATH=../skaters/src python parity/check_julia_checkpoints.py /tmp/julia_checkpoints.json
using JSON
using Skaters
include(joinpath(@__DIR__, "..", "test", "state_scenarios.jl"))

v = JSON.parsefile(joinpath(@__DIR__, "state_vectors.json"))
parseexp(x::AbstractString) = x == "nan" ? NaN : x == "inf" ? Inf : x == "-inf" ? -Inf : error(x)
parseexp(x) = Float64(x)
out = Dict{String,Any}()
for (name, mk, sname) in state_scenarios()
    ys = Float64[parseexp(x) for x in v["series"][sname]]
    cps = Set(Int(c["t"]) for c in v["scenarios"][name]["checkpoints"])
    f = mk()
    st = nothing
    rows = Any[]
    for t in 0:maximum(cps)
        _, st = f(ys[t+1], st)
        t in cps && push!(rows, Dict("t" => t, "state" => state_to_json(st)))
    end
    out[name] = rows
end
open(ARGS[1], "w") do io
    JSON.print(io, out)
end
println("wrote $(length(out)) scenarios to $(ARGS[1])")
