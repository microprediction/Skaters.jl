# Parity for the rest of the reference's public API: homogenize,
# residual_transform with ResidualDiagnostics, dantzig and search, and the
# spec system (build, spec_name, to_json, from_json). The vectors come from
# the Python reference (parity/gen_wave2_vectors.py writes
# parity/wave2_vectors.json); the probes are those of test/parity.jl.
#
# Every value holds at 1e-6, including the long-series runs that wrap
# laplace past its GPD splice (their 32-point quantile grids stay clear of
# the extreme quantiles that test/transform_parity.jl has to allow for).
#
# Run: julia --project=. test/wave2_parity.jl
using JSON
using Skaters

v = JSON.parsefile(joinpath(@__DIR__, "..", "parity", "wave2_vectors.json"))
short = Float64.(JSON.parsefile(joinpath(@__DIR__, "..", "parity", "vectors.json"))["series"])
long = Float64.(v["long_series"])
const PROBE = Float64(v["probe"])
const Q_LO = Float64(v["q_lo"])
const Q_HI = Float64(v["q_hi"])
const ATOL = 1e-6
const RTOL = 1e-6

probe(d) = (dist_mean(d), dist_std(d), dist_logpdf(d, PROBE), dist_cdf(d, PROBE),
            dist_quantile(d, Q_LO), dist_quantile(d, Q_HI), dist_crps(d, PROBE))

parseexp(x::AbstractString) = x == "nan" ? NaN : x == "inf" ? Inf : x == "-inf" ? -Inf : parse(Float64, x)
parseexp(x) = Float64(x)

fails = 0
checked = 0

function compare!(label, got, want, atol, rtol)
    global fails, checked
    bad = 0
    for (j, w) in enumerate(want)
        we = parseexp(w)
        g = Float64(got[j])
        checked += 1
        same = (isnan(we) && isnan(g)) || (isinf(we) && g == we) ||
               (isfinite(we) && abs(g - we) <= atol + rtol * abs(we))
        if !same
            bad += 1
            fails += 1
            fails <= 10 && println("FAIL $label item $j: julia $g python $we")
        end
    end
    return bad
end

function check(cond::Bool, msg::String)
    global fails, checked
    checked += 1
    if !cond
        fails += 1
        println("FAIL: ", msg)
    end
end

# --- specs: same constructors, same bytes ---
specs = Any[
    leaf_spec(1),
    ema_spec(0.05, 1),
    ensemble_spec(ema_spec(0.01, 1), ema_spec(0.1, 1); k = 1),
    conjugate_spec(ensemble_spec(ema_spec(0.01, 1), ema_spec(0.1, 1); k = 1), diff_spec()),
    conjugate_spec(leaf_spec(3), frac_spec(0.35, 30)),
    conjugate_spec(conjugate_spec(leaf_spec(1), std_spec(0.02)), ema_t_spec(1e-05)),
    conjugate_spec(ema_spec(0.2, 3), frac_spec(0.4)),
]
length(specs) == length(v["specs"]) || error("spec registry mismatch")
for (i, (sp, want)) in enumerate(zip(specs, v["specs"]))
    check(to_json(sp) == want["json"], "spec $i to_json: $(to_json(sp)) vs $(want["json"])")
    loaded = from_json(want["json"])
    check(spec_name(loaded) == want["name"], "spec $i name: $(spec_name(loaded)) vs $(want["name"])")
    check(spec_name(sp) == want["name"], "spec $i name from constructors")
    check(to_json(loaded) == want["roundtrip"], "spec $i round trip")
end
println("ok   specs ($(length(specs)): to_json bytes, from_json, spec_name)")

# --- candidates ---
let cands = homogenize_candidates(), want = v["candidates"]
    check(length(cands) == length(want), "candidate count")
    for (c, w) in zip(cands, want), key in ("rho", "gain", "delta", "is_identity")
        check(c[key] == w[key], "candidate $key: $(c[key]) vs $(w[key])")
    end
    println("ok   homogenize_candidates ($(length(cands)))")
end

# --- scenario registry (mirrors parity/gen_wave2_vectors.py) ---
small_candidates() = homogenize_candidates(rho_grid = (0.0, 0.9), gain_grid = (0.1,),
                                           delta_grid = (0.0, 0.5))
scenarios = Tuple{String,Any}[
    ("rt_m0", residual_transform(leaf(1); model = "m0")),
    ("rt_m1", residual_transform(leaf(1); model = "m1")),
    ("rt_m2", residual_transform(leaf(1); model = "m2")),
    ("rt_m2_ema", residual_transform(conjugate(leaf(1), ema_transform(0.1), 1); model = "m2", lr = 0.05)),
    ("rt_laplace", residual_transform(laplace(k = 1))),
    ("hg", homogenize(leaf(1))),
    ("hg_small", homogenize(leaf(1); candidates = small_candidates())),
    ("hg_laplace", homogenize(laplace(k = 1))),
    ("dantzig", dantzig(k = 1, expand_interval = 50)),
    ("search_k3", search(k = 3, expand_interval = 40)),
]
for couple in ("none", "weights", "state", "shared")
    push!(scenarios, ("hg_k3_$couple", homogenize(leaf(3); k = 3, couple = couple, lam = 0.4)))
end
for (i, sp) in enumerate(specs)
    push!(scenarios, ("spec_$(i-1)", build(from_json(v["specs"][i]["json"]))))
end
push!(scenarios, ("hg_laplace_long", homogenize(laplace(k = 1))))
push!(scenarios, ("rt_laplace_long", residual_transform(laplace(k = 1))))
push!(scenarios, ("hg_laplace_k3_long", homogenize(laplace(k = 3); k = 3, couple = "weights")))

length(scenarios) == length(v["scenarios"]) ||
    error("registry has $(length(scenarios)) scenarios, vectors have $(length(v["scenarios"]))")

zstream = Any[]
for (name, sk) in scenarios
    spec = v["scenarios"][name]
    is_long = spec["series"] == "long"
    series = is_long ? long : short
    rows = Dict(Int(r[1]) => r[2] for r in spec["out"])
    st = nothing
    sc = 0
    for i in eachindex(series)
        dists, st = sk(series[i], st)
        name == "rt_m2" && push!(zstream, st["z"])
        haskey(rows, i - 1) || continue
        want_h = rows[i-1]
        length(dists) == length(want_h) || error("$name step $(i-1): horizon count")
        for (h, d) in enumerate(dists)
            sc += compare!("$name step $(i-1) h$h", probe(d), want_h[h], ATOL, RTOL)
        end
    end
    println(sc == 0 ? "ok   " : "FAIL ", name)
end

# --- residual_transform's z stream and ResidualDiagnostics fed from it ---
let want = v["rt_m2_z"]
    check(length(zstream) == length(want), "z stream length")
    check(zstream[1] === nothing && want[1] === nothing, "z is nothing on the first tick")
    compare!("rt_m2 z", Float64[z for z in zstream[2:end]], want[2:end], ATOL, RTOL)
    rd = ResidualDiagnostics(K = 3, halflife = 20.0)
    for z in zstream
        z === nothing || Skaters.update!(rd, z)
    end
    s = summary(rd)
    w = v["diagnostics"]
    check(s["n"] == w["n"], "diagnostics n")
    compare!("diagnostics moments", [s["E[z]"], s["E[z2]"], s["E[z3]"], s["E[z4]"]],
             [w["E[z]"], w["E[z2]"], w["E[z3]"], w["E[z4]"]], ATOL, RTOL)
    compare!("diagnostics L", s["L"], w["L"], ATOL, RTOL)
    compare!("diagnostics cross_sq", s["cross_sq"], w["cross_sq"], ATOL, RTOL)
    println("ok   residual z stream and ResidualDiagnostics")
end

println("$checked values checked")
fails == 0 || error("WAVE2 PARITY FAILED: $fails value(s)")
println("WAVE2 PARITY OK")
