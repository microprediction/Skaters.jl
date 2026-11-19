# Transforms that read mixture components, fed laplace's tail-spliced
# predictive. Mirrors the reference's tests/test_spliced_nonaffine.py
# (skaters bc67bd8). The nonlinear inverses (power_transform, yeo_johnson,
# gaussianize) emit the exact pushforward of the spliced law; the k>1
# widening of ema_transform, seasonal_difference and seasonal_anchor widens
# the body and lets the splice follow it.
#
# Run: julia --project=. test/spliced_nonaffine.jl
using Skaters

# Deterministic UInt32 LCG plus Box-Muller, as in robustness.jl.
mutable struct LCG
    s::UInt32
end

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

function series(n::Int; seed::Int = 5)
    r = LCG(UInt32(seed))
    x = 0.0
    out = Float64[]
    for _ in 1:n
        x = 0.9 * x + gauss!(r) * (1.0 + 0.5 * sin(length(out) / 7.0))
        push!(out, x)
    end
    return out
end

failures = 0
function check(cond::Bool, msg::String)
    global failures
    if !cond
        failures += 1
        println("FAIL: ", msg)
    end
end

# pytest.approx semantics: |x - y| <= max(rel * |y|, abs)
approx(x, y; rtol = 0.0, atol = 0.0) = abs(x - y) <= max(atol, rtol * abs(y))

function run(f, n::Int)
    st = nothing
    d = nothing
    for y in series(n)
        d, st = f(y, st)
    end
    return d
end

function spliced()
    d = run(laplace(), 1500)[1]
    @assert d isa SplicedDist "laplace did not reach the GPD splice in 1500 steps"
    return d
end

function sane(name, d)
    check(isfinite(dist_logpdf(d, dist_quantile(d, 0.5))), "$name: logpdf at median not finite")
    qs = [dist_quantile(d, p) for p in (0.001, 0.1, 0.5, 0.9, 0.999)]
    check(all(qs[i] < qs[i+1] for i in 1:length(qs)-1), "$name: quantiles not increasing $qs")
    check(isfinite(dist_mean(d)) && dist_std(d) > 0, "$name: mean or std not usable")
end

# The nonlinear inverse is the exact pushforward.
for (name, make, inv) in (
        ("power_transform(0.5)", () -> power_transform(0.5), z -> copysign(abs(z)^2.0, z)),
        ("yeo_johnson(0.0)", () -> yeo_johnson(0.0),
         z -> z >= 0 ? expm1(z) : 1.0 - sqrt(1.0 - 2.0 * z)))
    d = spliced()
    _, inverse_k = make()
    e = inverse_k(Any[d], Dict{String,Any}())[1]
    check(e isa PushforwardDist, "$name: inverse is not a PushforwardDist")
    for p in (0.001, 0.05, 0.5, 0.95, 0.999)
        want = inv(dist_quantile(d, p))
        check(approx(dist_quantile(e, p), want; rtol = 1e-4, atol = 1e-6),
              "$name quantile p=$p: $(dist_quantile(e, p)) vs $want")
    end
    for z in (-1.0, 0.2, 1.5)
        # table resolution
        check(approx(dist_cdf(e, inv(z)), dist_cdf(d, z); atol = 1e-3),
              "$name cdf z=$z: $(dist_cdf(e, inv(z))) vs $(dist_cdf(d, z))")
    end
    check(isfinite(dist_crps(e, inv(0.2))) && dist_crps(e, inv(0.2)) > 0, "$name: crps not usable")
end

# Widening moves the body and keeps the splice.
let d = spliced()
    e = Skaters._widen(d, 2.0, 0.5)
    check(e isa SplicedDist, "_widen is not a SplicedDist")
    check((e.t_lo, e.t_up, e.zeta_lo, e.zeta_up) == (d.t_lo, d.t_up, d.zeta_lo, d.zeta_up),
          "_widen changed the splice")
    check(approx(dist_mean(e.body), dist_mean(d.body) + 2.0; rtol = 1e-6), "_widen body mean")
    check(approx(dist_var(e.body), dist_var(d.body) + 0.5; rtol = 1e-6), "_widen body var")
end

# Conjugated laplace runs past the splice (live from about step 510).
for (name, make, k) in (("power_transform", () -> power_transform(0.5), 1),
                        ("yeo_johnson", () -> yeo_johnson(0.5), 1),
                        ("gaussianize", () -> gaussianize(), 1),
                        ("ema_transform", () -> ema_transform(0.05), 3),
                        ("seasonal_difference", () -> seasonal_difference(2), 3),
                        ("seasonal_anchor", () -> seasonal_anchor(2), 3))
    ds = run(conjugate(laplace(k = k), make(), k), 600)
    check(length(ds) == k, "$name: expected $k dists, got $(length(ds))")
    for d in ds
        sane(name, d)
    end
end

if failures > 0
    error("SPLICED NONAFFINE FAILED: $failures check(s)")
end
println("SPLICED NONAFFINE OK")
