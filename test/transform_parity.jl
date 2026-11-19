# Parity for seasonal_scale, gaussianize, log_transform, the exact flag on
# power_transform and yeo_johnson, stacked and combined pushforwards, affine
# maps of pushforwards (standardize or seasonal_scale around them),
# flesaker, and laplace's tail-spliced predictive through each
# component-reading transform. The vectors come from the Python reference
# (parity/gen_transform_vectors.py writes parity/transform_vectors.json);
# the probes are those of test/parity.jl.
#
# Tolerances. Every short-series scenario holds at 1e-6, as in parity.jl.
# The rest read a tail-spliced predictive's extreme quantiles: the
# pushforward table window at quantile(1e-6) and quantile(1 - 1e-6), and the
# 257-point quantile grid behind the moment shim and the quantile-quadrature
# CRPS reach p near 1e-12. Out there a Gaussian body's cdf is 0.5(1 + erf(z))
# with erf near -1, and openlibm's erf differs from the erf CPython gets from
# the platform libm by one ulp on roughly a quarter of arguments. That moves
# the body's quantile at p = 1e-12 by about 1e-6, and exponential inverses
# (yeo_johnson at 0, log_transform) amplify it. So:
#
#   fixed_spliced: the exact SplicedDist the reference captured from laplace,
#     rebuilt on both sides and pushed through every inverse, at 1e-5 relative.
#   long-series end-to-end runs of laplace past its splice, at 1e-3: after
#     500+ steps the two laplace states also differ in their last ulps, and a
#     window endpoint where the body's cdf is clamped at 1 - 1e-12 and flat
#     moves by about 1e-4, the table interpolation by about 1e-5.
#
# That sensitivity is in the reference's rule, not in the port.
#
using JSON
using Skaters

const vec_path = joinpath(@__DIR__, "..", "parity", "transform_vectors.json")
v = JSON.parsefile(vec_path)
short = Float64.(JSON.parsefile(joinpath(@__DIR__, "..", "parity", "vectors.json"))["series"])
long = Float64.(v["long_series"])
lo, hi = Float64.(v["short_bounds"])
llo, lhi = Float64.(v["long_bounds"])
const PROBE = Float64(v["probe"])
const Q_LO = Float64(v["q_lo"])
const Q_HI = Float64(v["q_hi"])
const ATOL = 1e-6
const RTOL = 1e-6
const FIXED_RTOL = 1e-5
const LONG_TOL = 1e-3

probe(d) = (dist_mean(d), dist_std(d), dist_logpdf(d, PROBE), dist_cdf(d, PROBE),
            dist_quantile(d, Q_LO), dist_quantile(d, Q_HI), dist_crps(d, PROBE))

parseexp(x::AbstractString) = x == "nan" ? NaN : x == "inf" ? Inf : x == "-inf" ? -Inf : parse(Float64, x)
parseexp(x) = Float64(x)

# --- scenario registry (mirrors parity/gen_transform_vectors.py) ---
scenarios = Tuple{String,Any}[]
for k in (1, 3)
    suf = k == 1 ? "" : "_k$k"
    add(name, sk) = push!(scenarios, (name * suf, sk))
    add("seasonal_scale", conjugate(leaf(k), seasonal_scale(7), k))
    add("seasonal_scale_center", conjugate(leaf(k), seasonal_scale(7; center = true), k))
    add("seasonal_scale_over_sdiff",
        conjugate(conjugate(leaf(k), seasonal_scale(7), k), seasonal_difference(7), k))
    add("gaussianize", conjugate(leaf(k), gaussianize(), k))
    add("gaussianize_cap", conjugate(leaf(k), gaussianize(slope_cap = 2.0), k))
    add("gaussianize_smooth", conjugate(leaf(k), gaussianize(smooth = true), k))
    add("gaussianize_delete", conjugate(leaf(k), gaussianize(guard = "delete", slope_cap = 2.0), k))
    add("gaussianize_emin", conjugate(leaf(k), gaussianize(guard = "emin", slope_cap = 2.0), k))
    add("power_exact", conjugate(leaf(k), power_transform(0.5; exact = true), k))
    add("yeojohnson_log_exact", conjugate(leaf(k), yeo_johnson(0.0; exact = true), k))
    add("yeojohnson_half_exact", conjugate(leaf(k), yeo_johnson(0.5; exact = true), k))
    add("stack_exact_delta",
        conjugate(conjugate(leaf(k), yeo_johnson(0.5; exact = true), k), power_transform(0.5), k))
    add("stack_exact_exact",
        conjugate(conjugate(leaf(k), yeo_johnson(0.5; exact = true), k),
                  power_transform(0.5; exact = true), k))
    add("combine_pushforwards", precision_weighted_ensemble(
        Any[conjugate(leaf(k), yeo_johnson(0.5; exact = true), k),
            conjugate(leaf(k), power_transform(0.5; exact = true), k)]; k = k))
    add("log_lower", conjugate(leaf(k), log_transform(lower = lo), k))
    add("log_upper", conjugate(leaf(k), log_transform(upper = hi), k))
    add("std_yeojohnson_exact",
        conjugate(conjugate(leaf(k), yeo_johnson(0.0; exact = true), k), standardize(), k))
    add("std_log", conjugate(conjugate(leaf(k), log_transform(lower = -10.0), k), standardize(), k))
    add("sscale_log", conjugate(conjugate(leaf(k), log_transform(lower = -10.0), k),
                                seasonal_scale(7; center = true), k))
end
push!(scenarios, ("laplace_power", conjugate(laplace(k = 1), power_transform(0.5), 1)))
push!(scenarios, ("laplace_yeojohnson", conjugate(laplace(k = 1), yeo_johnson(0.5), 1)))
push!(scenarios, ("laplace_gaussianize", conjugate(laplace(k = 1), gaussianize(), 1)))
push!(scenarios, ("laplace_stack",
    conjugate(conjugate(laplace(k = 1), yeo_johnson(0.5), 1), power_transform(0.5), 1)))
push!(scenarios, ("laplace_seasonal_scale", conjugate(laplace(k = 1), seasonal_scale(7), 1)))
push!(scenarios, ("laplace_ema_k3", conjugate(laplace(k = 3), ema_transform(0.05), 3)))
push!(scenarios, ("laplace_sdiff_k3", conjugate(laplace(k = 3), seasonal_difference(2), 3)))
push!(scenarios, ("laplace_sanchor_k3", conjugate(laplace(k = 3), seasonal_anchor(2), 3)))
push!(scenarios, ("flesaker", flesaker(k = 1, lower = llo)))
push!(scenarios, ("flesaker_upper_k3", flesaker(k = 3, upper = lhi)))
push!(scenarios, ("laplace_std_yeojohnson_exact",
    conjugate(conjugate(laplace(k = 1), yeo_johnson(0.0; exact = true), 1), standardize(0.05), 1)))

expected_all = v["scenarios"]
length(scenarios) == length(expected_all) ||
    error("registry has $(length(scenarios)) scenarios, vectors have $(length(expected_all))")

fails = 0
checked = 0

function compare!(label, got, want, atol, rtol)
    global fails, checked
    bad = 0
    for (j, w) in enumerate(want)
        we = parseexp(w)
        g = got[j]
        checked += 1
        same = (isnan(we) && isnan(g)) || (isinf(we) && g == we) ||
               (isfinite(we) && abs(g - we) <= atol + rtol * abs(we))
        if !same
            bad += 1
            fails += 1
            fails <= 10 && println("FAIL $label probe $j: julia $g python $we")
        end
    end
    return bad
end

for (name, sk) in scenarios
    spec = expected_all[name]
    is_long = spec["series"] == "long"
    series = is_long ? long : short
    atol = is_long ? LONG_TOL : ATOL
    rtol = is_long ? LONG_TOL : RTOL
    rows = Dict(Int(r[1]) => r[2] for r in spec["out"])
    st = nothing
    sc_fails = 0
    for i in eachindex(series)
        dists, st = sk(series[i], st)
        haskey(rows, i - 1) || continue
        want_h = rows[i-1]
        length(dists) == length(want_h) || error("$name step $(i-1): horizon count")
        for (h, d) in enumerate(dists)
            sc_fails += compare!("$name step $(i-1) h$h", probe(d), want_h[h], atol, rtol)
        end
    end
    println(sc_fails == 0 ? "ok   " : "FAIL ", name)
end

# --- fixed_spliced: the exact captured SplicedDist through every inverse ---
function state_after(tr, series)
    fwd, inv = tr
    st = nothing
    for y in series
        _, st = fwd(y, st)
    end
    return inv, st
end

let (g_inv, g_st) = state_after(gaussianize(), short),
    (s_inv, s_st) = state_after(seasonal_scale(7; center = true), short)
    nfixed = 0
    for c in v["fixed_spliced"]
        comps = NTuple{3,Float64}[(Float64(x[1]), Float64(x[2]), Float64(x[3])) for x in c["comps"]]
        d = SplicedDist(Dist(comps), Float64.(c["par"])...)
        pw = power_transform(0.5)[2]
        pwx = power_transform(0.5; exact = true)[2]
        yj = yeo_johnson(0.5)[2]
        yjx = yeo_johnson(0.0; exact = true)[2]
        lg = log_transform(lower = llo)[2]
        lgu = log_transform(upper = -llo)[2]
        e = Dict{String,Any}()
        a = yj(Any[d], Dict{String,Any}())[1]
        b = pwx(Any[d], Dict{String,Any}())[1]
        got = Dict{String,Any}(
            "power" => pw(Any[d], e)[1],
            "power_exact" => b,
            "yeojohnson" => a,
            "yeojohnson_log_exact" => yjx(Any[d], e)[1],
            "stack" => pw(Any[a], e)[1],
            "stack_exact" => pwx(Any[a], e)[1],
            "log_lower" => lg(Any[d], e)[1],
            "log_upper" => lgu(Any[d], e)[1],
            "gaussianize" => g_inv(Any[d], g_st)[1],
            "seasonal_scale" => s_inv(Any[d], s_st)[1],
            "widen" => Skaters._widen(d, 0.7, 0.3),
            "widen_pushforward" => Skaters._widen(a, 0.7, 0.3),
            "combine" => dist_combine(Any[a, b], [0.3, 0.7]),
            "affine_power_exact" => dist_affine(b, 2.5, -1.0),
            "affine_power_exact_neg" => dist_affine(b, -1.7, 0.5),
            "affine_log_lower_neg" => dist_affine(lg(Any[d], e)[1], -0.3, 4.0),
            "shift_yeojohnson_log_exact" => dist_shift(yjx(Any[d], e)[1], 1.5),
            "scale_log_upper" => dist_scale(lgu(Any[d], e)[1], -2.0),
        )
        length(got) == length(c["out"]) || error("fixed_spliced: probe set mismatch")
        for (key, want) in c["out"]
            nfixed += compare!("fixed_spliced $key", probe(got[key]), want, ATOL, FIXED_RTOL)
        end
    end
    println(nfixed == 0 ? "ok   " : "FAIL ", "fixed_spliced ($(length(v["fixed_spliced"])) captured predictives)")
end

println("$checked values checked")
fails == 0 || error("TRANSFORM PARITY FAILED: $fails value(s)")
println("TRANSFORM PARITY OK")
