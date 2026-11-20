# dist_affine, dist_shift and dist_scale on PushforwardDist: the law of
# a*Y + b. Mirrors the reference's tests/test_pushforward_affine.py. Needed so
# a pushforward predictive (exact yeo_johnson, log_transform, a spliced law
# through a nonlinear inverse) can be handed to standardize, seasonal_scale
# or any other affine transform.
#
# Run: julia --project=. test/pushforward_affine.jl
using Skaters

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

function dists()
    inner = Dist(NTuple{3,Float64}[(0.6, 0.2, 0.7), (0.4, -0.5, 1.3)])
    _, inv_k = yeo_johnson(0.0; exact = true)
    table = inv_k(Any[inner], Dict{String,Any}())[1]          # table + inv_fn
    _, inv_log = log_transform()
    analytic = inv_log(Any[inner], Dict{String,Any}())[1]     # analytic fwd/inv
    return Dict("table" => table, "analytic" => analytic)
end

# Change of variables, for both map kinds and positive and negative a.
for kind in ("table", "analytic"), (a, b) in ((2.5, -1.0), (0.3, 4.0), (-1.7, 0.5))
    d = dists()[kind]
    e = dist_affine(d, a, b)
    tag = "$kind a=$a b=$b"
    check(e isa PushforwardDist, "$tag: not a PushforwardDist")
    for p in (0.01, 0.2, 0.5, 0.8, 0.99)
        want = a * dist_quantile(d, a > 0 ? p : 1.0 - p) + b
        check(approx(dist_quantile(e, p), want; rtol = 1e-7, atol = 1e-9),
              "$tag quantile p=$p: $(dist_quantile(e, p)) vs $want")
    end
    for p in (0.05, 0.4, 0.9)
        x = dist_quantile(d, p)
        y = a * x + b
        want_cdf = a > 0 ? dist_cdf(d, x) : 1.0 - dist_cdf(d, x)
        check(approx(dist_cdf(e, y), want_cdf; atol = 1e-7), "$tag cdf p=$p")
        check(approx(dist_logpdf(e, y), dist_logpdf(d, x) - log(abs(a)); atol = 1e-4), "$tag logpdf p=$p")
    end
    check(approx(dist_mean(e), a * dist_mean(d) + b; rtol = 1e-6, atol = 1e-9), "$tag mean")
    check(approx(dist_std(e), abs(a) * dist_std(d); rtol = 1e-6), "$tag std")
    x3 = dist_quantile(d, 0.3)
    check(approx(dist_crps(e, a * x3 + b), abs(a) * dist_crps(d, x3); rtol = 1e-3), "$tag crps")
end

# shift and scale, and the zero multiplier.
let d = dists()["table"]
    check(approx(dist_quantile(dist_shift(d, 1.5), 0.3), dist_quantile(d, 0.3) + 1.5; rtol = 1e-6), "shift")
    check(approx(dist_quantile(dist_scale(d, -2.0), 0.3), -2.0 * dist_quantile(d, 0.7); rtol = 1e-6), "scale(-2)")
    for f in (() -> dist_scale(d, 0.0), () -> dist_affine(d, 0.0, 1.0))
        threw = false
        try
            f()
        catch err
            threw = err isa ArgumentError
        end
        check(threw, "zero multiplier did not throw ArgumentError")
    end
end

# standardize around exact yeo_johnson around laplace. The reference draws
# abs(gauss(3, 1)); this uses the deterministic LCG of the other suites.
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

let r = LCG(UInt32(7))
    f = conjugate(conjugate(laplace(), yeo_johnson(0.0; exact = true), 1), standardize(0.05), 1)
    st = nothing
    d = nothing
    for _ in 1:300
        d, st = f(abs(3.0 + gauss!(r)), st)
    end
    check(d[1] isa PushforwardDist, "standardize around exact yeo_johnson: not a PushforwardDist")
    check(dist_quantile(d[1], 0.1) < dist_quantile(d[1], 0.9), "standardize around exact yeo_johnson: quantiles")
    check(isfinite(dist_logpdf(d[1], dist_quantile(d[1], 0.5))), "standardize around exact yeo_johnson: logpdf")
end

if failures > 0
    error("PUSHFORWARD AFFINE FAILED: $failures check(s)")
end
println("PUSHFORWARD AFFINE OK")
