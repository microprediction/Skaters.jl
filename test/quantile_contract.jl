# Shared cross-language inverse-CDF contract (parity/quantile_contract.json,
# copied verbatim from the Python reference; see its parity/QUANTILE_CONTRACT.md).
# Same assertions as tests/test_quantile_inverse_contract.py,
# parity/quantile_contract.mjs and rust/tests/quantile_contract.rs.
# Run:  julia --project=. test/quantile_contract.jl
using JSON
using Skaters

const qc_path = joinpath(@__DIR__, "..", "parity", "quantile_contract.json")
cases = JSON.parsefile(qc_path)

qc_fails = 0
function qc_check(ok::Bool, msg)
    global qc_fails
    if !ok
        qc_fails += 1
        println("FAIL ", msg)
    end
end

for c in cases
    comps = NTuple{3,Float64}[(Float64(x[1]), Float64(x[2]), Float64(x[3])) for x in c["components"]]
    d = Dist(comps)
    p = Float64(c["p"])
    q = dist_quantile(d, p)
    name = c["name"]
    if haskey(c, "expected")
        # The fixture carries a single tolerance key, `atol` (the one the
        # Rust and JS runners use); no per-language override exists.
        x = Float64(c["expected"])
        atol = Float64(c["atol"])
        qc_check(abs(q - x) <= atol, "$name: q=$q expected=$x gap=$(abs(q - x)) atol=$atol")
    else
        lo = Float64(c["lower"])
        hi = Float64(c["upper"])
        qc_check(lo < q < hi, "$name: q=$q not in ($lo, $hi)")
    end
    if haskey(c, "cdf_tol")
        t = Float64(c["cdf_tol"])
        f = dist_cdf(d, q)
        qc_check(abs(f - p) <= t, "$name: q=$q cdf=$f p=$p cdf_tol=$t")
    end
end
println("$(length(cases)) quantile contract cases checked")
qc_check(length(cases) == 64, "expected 64 contract cases, got $(length(cases))")

# Tiny spread is preserved.
let d = Dist([(1.0, 0.0, 1e-11)])
    spread = dist_quantile(d, 0.99) - dist_quantile(d, 0.01)
    qc_check(abs(spread - 4.6526957480816816e-11) < 2e-20, "tiny spread: $spread")
end

# Explicit budget and resolution stop.
let d = Dist([(1.0, 0.0, 1.0)])
    qc_check(dist_quantile(d, 0.5; max_iter = 0) == 0.0, "max_iter=0")
    qc_check(dist_quantile(d, 0.5; max_iter = 1) == -4.0, "max_iter=1")
    for tol in (0.0, -1.0)
        q = dist_quantile(d, 0.5; tol = tol)
        qc_check(abs(q) < 1e-15 && dist_cdf(d, q) >= 0.5, "tol=$tol: q=$q")
    end
    # Unit-scale work is unchanged: the reference counts 36 CDF calls at
    # p = 0.9 (two bracket checks, 34 bisections). Julia cannot subclass Dist
    # to count calls, so pin the bisection count through the budget instead:
    # 34 iterations reproduce the default answer, 33 do not.
    q = dist_quantile(d, 0.9)
    qc_check(dist_quantile(d, 0.9; max_iter = 34) == q, "unit scale: 34 bisections suffice")
    qc_check(dist_quantile(d, 0.9; max_iter = 33) != q, "unit scale: 33 bisections do not")
end

# Variance overflow fails explicitly.
let threw = false
    try
        dist_quantile(Dist([(0.5, -1e155, 1.0), (0.5, 1e155, 1.0)]), 0.5)
    catch e
        threw = occursin("finite mixture variance", sprint(showerror, e))
    end
    qc_check(threw, "variance overflow must raise 'finite mixture variance'")
end

if qc_fails > 0
    println("QUANTILE CONTRACT FAILED: $qc_fails")
    exit(1)
end
println("QUANTILE CONTRACT OK")
