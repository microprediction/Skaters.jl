# Parity against the Python-generated vectors, 1e-6 discipline (same gate as
# the R and Rust ports). Run:  julia --project=. test/parity.jl
using JSON
using Skaters

const vec_path = joinpath(@__DIR__, "..", "parity", "vectors.json")
v = JSON.parsefile(vec_path)
series = Float64.(v["series"])
const BURN = Int(v["burn"])
const PROBE = Float64(v["probe"])
const Q_LO = Float64(v["q_lo"])
const Q_HI = Float64(v["q_hi"])
const ATOL = 1e-6
const RTOL = 1e-6

probe(d) = (dist_mean(d), dist_std(d), dist_logpdf(d, PROBE), dist_cdf(d, PROBE),
            dist_quantile(d, Q_LO), dist_quantile(d, Q_HI), dist_crps(d, PROBE))

parseexp(x::AbstractString) = x == "nan" ? NaN : x == "inf" ? Inf : x == "-inf" ? -Inf : parse(Float64, x)
parseexp(x) = Float64(x)

# --- scenario registry (mirrors parity/gen_vectors.py, minus search/spec/cov) ---
scenarios = Tuple{String,Int,Any}[]
for k in (1, 3)
    suf = k == 1 ? "" : "_k$k"
    add(name, sk) = push!(scenarios, (name * suf, k, sk))
    add("leaf", leaf(k))
    add("diff", conjugate(leaf(k), difference(), k))
    add("ema_t", conjugate(leaf(k), ema_transform(0.1), k))
    add("standardize", conjugate(leaf(k), standardize(), k))
    add("theta", conjugate(leaf(k), theta(0.1), k))
    add("drift", conjugate(leaf(k), drift(0.05, 0.01), k))
    add("holt", conjugate(leaf(k), holt_linear(0.1, 0.05), k))
    add("garch", conjugate(leaf(k), garch(), k))
    add("seasonal", conjugate(leaf(k), seasonal_difference(7), k))
    add("power", conjugate(leaf(k), power_transform(0.5), k))
    add("ar1", conjugate(leaf(k), ar(1), k))
    add("ar2", conjugate(leaf(k), ar(2; decay = 1.0), k))
    add("frac", conjugate(leaf(k), fractional_difference(0.4, 30), k))
    add("grouped_ar", conjugate(leaf(k), grouped_ar(8), k))
    add("yeojohnson_log", conjugate(leaf(k), yeo_johnson(0.0), k))
    add("yeojohnson_half", conjugate(leaf(k), yeo_johnson(0.5), k))
    add("ou", conjugate(leaf(k), ou_transform(0.1), k))
    add("ou_sqrt", conjugate(conjugate(leaf(k), ou_transform(0.1), k), yeo_johnson(0.5), k))
    add("ema_skater", ema(0.05; k = k))
    add("pw_ensemble", precision_weighted_ensemble(Any[ema(0.05; k = k), ema(0.2; k = k)]; k = k))
    add("multiscale", multiscale(kk -> conjugate(leaf(kk), ema_transform(0.1), kk), k))
    add("bayes_ensemble", bayesian_ensemble(Any[ema(0.05; k = k), conjugate(leaf(k), difference(), k)];
        k = k, learning_rate = 0.5, complexity_penalty = 0.02, depths = [1, 1]))
end
push!(scenarios, ("scale_mixture_leaf", 1, scale_mixture_leaf(1)))
push!(scenarios, ("crps_leaf", 1, crps_leaf(1)))
push!(scenarios, ("garch_leaf", 1, garch_leaf(1)))
push!(scenarios, ("scalemix_ema", 1, conjugate(scale_mixture_leaf(1), ema_transform(0.1), 1)))
push!(scenarios, ("gpd_tails", 1, gpdtails(conjugate(leaf(1), ema_transform(0.1), 1), 1;
    level = 0.9, nexc = 50, warmup = 100)))
push!(scenarios, ("pol_laplace", 1, laplace(k = 1)))
push!(scenarios, ("pol_laplace_k3", 3, laplace(k = 3)))

fails = 0
checked = 0

function check_block(scenarios, series, expected_block)
    global fails, checked
    for (name, k, sk) in scenarios
        expected = expected_block[name]["out"]
        st = nothing
        row = 0
        for i in eachindex(series)
            dists, st = sk(series[i], st)
            if i - 1 >= BURN
                row += 1
                for h in 1:k
                    got = probe(dists[h])
                    exprow = expected[row][h]
                    for j in 1:7
                        checked += 1
                        e = parseexp(exprow[j])
                        isnan(e) && continue
                        if abs(got[j] - e) > ATOL + RTOL * abs(e)
                            fails += 1
                            if fails < 8
                                println("FAIL $name row $row h $h probe $j: got $(got[j]) want $e")
                            end
                        end
                    end
                end
            end
        end
        println("ok   $name")
    end
end

check_block(scenarios, series, v["scenarios"])

# Sticky/dirac on the repeat-heavy series (exact repeats + 0.25-grid jumps).
repeat_series = Float64.(v["repeat_series"])
repeat_scenarios = Tuple{String,Int,Any}[
    ("sticky_ema", 1, sticky(conjugate(leaf(1), ema_transform(0.1), 1); k = 1)),
]
check_block(repeat_scenarios, repeat_series, v["repeat_scenarios"])

println("$checked values checked")
if fails > 0
    println("PARITY FAILED: $fails")
    exit(1)
end
println("PARITY OK")
