# dist_affine, dist_shift and dist_scale on SplicedDist. Mirrors the
# reference's tests/test_spliced_affine.py (skaters #358).
#
# Conjugating laplace with an affine transform (standardize, garch) calls
# dist_affine on the predictive, which is a SplicedDist once the GPD tail
# splice is live. The splice is defined in the body's PIT space, so an
# affine map moves only the body; the cdf, quantile and logpdf of the mapped
# law must equal the change of variables.
#
# The reference test also wraps seasonal_scale(7); that transform is not yet
# ported here. The shift and scale transforms (ema_transform, seasonal
# difference and anchor, ou_transform) are covered too, with difference as a
# control that only reads moments.
#
# Run: julia --project=. test/spliced_affine.jl
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

function series(n::Int; seed::Int = 3)
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

approx(x, y; rtol = 0.0, atol = 0.0) = abs(x - y) <= max(atol, rtol * abs(y))

function spliced()
    f = laplace()
    st = nothing
    d = nothing
    for y in series(1500)
        d, st = f(y, st)
    end
    @assert d[1] isa SplicedDist "laplace did not reach the GPD splice in 1500 steps"
    return d[1]
end

# Change of variables, for a positive, a small positive and a negative multiplier.
for (a, b) in ((2.5, -1.0), (0.1, 3.0), (-1.5, 0.7))
    d = spliced()
    e = dist_affine(d, a, b)
    check(e isa SplicedDist, "affine($a, $b) is not a SplicedDist")
    for p in (0.001, 0.05, 0.5, 0.95, 0.999)
        want = a * dist_quantile(d, a > 0 ? p : 1.0 - p) + b
        check(approx(dist_quantile(e, p), want; rtol = 1e-6, atol = 1e-9),
              "quantile a=$a b=$b p=$p: $(dist_quantile(e, p)) vs $want")
    end
    for x in (-2.0, 0.3, 4.0)
        want = a > 0 ? dist_cdf(d, x) : 1.0 - dist_cdf(d, x)
        check(approx(dist_cdf(e, a * x + b), want; atol = 1e-9),
              "cdf a=$a b=$b x=$x: $(dist_cdf(e, a * x + b)) vs $want")
        want = dist_logpdf(d, x) - log(abs(a))
        check(approx(dist_logpdf(e, a * x + b), want; atol = 1e-6),
              "logpdf a=$a b=$b x=$x: $(dist_logpdf(e, a * x + b)) vs $want")
    end
end

# shift and scale are the same change of variables.
let d = spliced()
    for (name, e, a, b) in (("shift(1.7)", dist_shift(d, 1.7), 1.0, 1.7),
                            ("scale(2.5)", dist_scale(d, 2.5), 2.5, 0.0),
                            ("scale(-0.4)", dist_scale(d, -0.4), -0.4, 0.0))
        check(e isa SplicedDist, "$name is not a SplicedDist")
        for p in (0.001, 0.05, 0.5, 0.95, 0.999)
            want = a * dist_quantile(d, a > 0 ? p : 1.0 - p) + b
            check(approx(dist_quantile(e, p), want; rtol = 1e-6, atol = 1e-9),
                  "$name quantile p=$p: $(dist_quantile(e, p)) vs $want")
        end
        for x in (-2.0, 0.3, 4.0)
            want = a > 0 ? dist_cdf(d, x) : 1.0 - dist_cdf(d, x)
            check(approx(dist_cdf(e, a * x + b), want; atol = 1e-9), "$name cdf x=$x")
            want = dist_logpdf(d, x) - log(abs(a))
            check(approx(dist_logpdf(e, a * x + b), want; atol = 1e-6), "$name logpdf x=$x")
        end
    end
end

# Zero multiplier rejects.
let threw = false
    try
        dist_affine(spliced(), 0.0, 1.0)
    catch err
        threw = err isa ArgumentError
    end
    check(threw, "affine(0, 1) did not throw ArgumentError")
end
let threw = false
    try
        dist_scale(spliced(), 0.0)
    catch err
        threw = err isa ArgumentError
    end
    check(threw, "scale(0) did not throw ArgumentError")
end

# Transforms wrap laplace past the splice. Before the fix the affine, shift
# and scale ones threw a MethodError once the GPD splice was live.
for (name, tr) in (("standardize(0.05)", () -> standardize(0.05)),
                   ("garch()", () -> garch()),
                   ("ema_transform(0.05)", () -> ema_transform(0.05)),
                   ("difference()", () -> difference()),
                   ("seasonal_difference(7)", () -> seasonal_difference(7)),
                   ("seasonal_anchor(7)", () -> seasonal_anchor(7)),
                   ("ou_transform()", () -> ou_transform()))
    f = conjugate(laplace(), tr(), 1)
    st = nothing
    d = nothing
    for y in series(1500)
        d, st = f(y, st)
    end
    check(isfinite(dist_logpdf(d[1], 0.0)), "$name: logpdf(0) not finite")
    check(dist_quantile(d[1], 0.1) < dist_quantile(d[1], 0.9), "$name: quantiles not ordered")
end

if failures > 0
    error("SPLICED AFFINE FAILED: $failures check(s)")
end
println("SPLICED AFFINE OK")
