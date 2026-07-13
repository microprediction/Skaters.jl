# Adversarial release gate on the deployed default (laplace, GPD tails).
# Mirrors the R port's tests/robustness.R and the Rust crate's robustness.rs,
# which in turn mirror the reference repo's parity/adversarial.mjs and
# tests/test_tails_robustness.py.
#
# Runs laplace(1) over the pathological streams a production deployment will
# eventually meet: constant series, lattice/repeats, a monster spike, an
# extreme finite tick, scale collapse, vol whiplash on a trend, and a long
# soak that must activate the GPD splice. Asserts the contract at
# checkpoints: forecasts stay finite and well-formed, the cdf stays a cdf,
# quantiles stay ordered, and the detector recovers after the insult.
#
# Run: julia --project=. test/robustness.jl
using Skaters

# Deterministic RNG (same LCG as the JS gate; UInt32 wraparound, then
# Box-Muller). No dependence on Julia's default RNG.
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

failures = 0

function check(cond::Bool, label::String)
    global failures
    if !cond
        failures += 1
        println("FAIL $label")
    end
end

function assert_wellformed(d, y_near::Float64, label::String)
    lp = dist_logpdf(d, y_near)
    check(!isnan(lp), "$label: logpdf NaN")            # +/-Inf tolerated
    cv = dist_cdf(d, y_near)
    check(0.0 <= cv <= 1.0, "$label: cdf out of [0,1]")
    qs = [dist_quantile(d, p) for p in (0.001, 0.25, 0.5, 0.75, 0.999)]
    check(all(isfinite, qs), "$label: non-finite quantile")
    check(all(qs[i+1] >= qs[i] - 1e-9 for i in 1:length(qs)-1),
        "$label: quantiles unordered")
    probes = vcat(qs[1] - 1.0, qs, qs[end] + 1.0)
    cs = [dist_cdf(d, x) for x in probes]
    check(all(cs[i+1] >= cs[i] - 1e-9 for i in 1:length(cs)-1),
        "$label: cdf not monotone")
end

function soak(ys::Vector{Float64}, label::String)
    f = laplace(k = 1)
    st = nothing
    dists = nothing
    for t in eachindex(ys)
        dists, st = f(ys[t], st)
        if t > 1 && (t - 1) % 997 == 0
            assert_wellformed(dists[1], ys[t], label)
        end
    end
    assert_wellformed(dists[1], ys[end], label)
    return st
end

# 1. constant series
soak(fill(3.7, 4000), "constant")
println("ok   constant")

# 2. lattice / repeats (exact repeats + 0.25-grid jumps)
let r = LCG(17)
    v = 1.0
    ys = zeros(6000)
    steps = (-0.25, 0.25, 0.5)
    for i in 1:6000
        if nextu!(r) >= 0.7
            v += steps[Int(floor(nextu!(r) * 3.0)) + 1]
        end
        ys[i] = v
    end
    soak(ys, "lattice")
    println("ok   lattice")
end

# 3. monster spike, then recovery (no deafness, no permanent alarm)
let r = LCG(23)
    f = laplace(k = 1)
    st = nothing
    for _ in 1:3000
        _, st = f(gauss!(r), st)
    end
    dists, st = f(1e9, st)                             # the insult
    assert_wellformed(dists[1], 0.0, "spike:after")
    alarms = 0
    n = 0
    for i in 1:3000
        y = gauss!(r)
        dists, st = f(y, st)
        z = st["z"][1]
        if i > 1001 && z !== nothing                   # mjs: i > 1000, 0-based
            n += 1
            if abs(z) > 2.5758                         # ~1e-2 two-sided
                alarms += 1
            end
        end
        if (i - 1) % 500 == 0
            assert_wellformed(dists[1], y, "spike:recovery")
        end
    end
    check(n > 1500, "spike: too few matured ticks")
    rate = alarms / n
    check(rate < 0.06, "spike: alarm rate $rate after recovery")
    println("ok   spike (alarm rate $(round(rate, digits = 4)) on $n matured ticks)")
end

# 3b. extreme finite tick: the input gate must keep the tree alive
let r = LCG(29)
    f = laplace(k = 1)
    st = nothing
    for _ in 1:1500
        _, st = f(gauss!(r), st)
    end
    dists, st = f(1e300, st)                           # near the double limit
    assert_wellformed(dists[1], 0.0, "extreme:after")
    for i in 1:1500
        dists, st = f(gauss!(r), st)
        if (i - 1) % 500 == 0
            assert_wellformed(dists[1], 0.0, "extreme:recovery")
        end
    end
    println("ok   extreme finite tick")
end

# 4. scale collapse and recovery
let r = LCG(31)
    ys = vcat([gauss!(r) for _ in 1:2000], zeros(2000), [gauss!(r) for _ in 1:2000])
    soak(ys, "collapse")
    println("ok   collapse")
end

# 5. vol whiplash on a trend
let r = LCG(41)
    ys = zeros(8000)
    lvl = 0.0
    for t in 0:7999
        vol = (div(t, 700) % 2 == 1) ? 10.0 : 1.0
        lvl += 0.05 + vol * gauss!(r)
        ys[t+1] = lvl
    end
    soak(ys, "whiplash")
    println("ok   whiplash")
end

# 6. long-series soak: 3000 gaussian ticks so the default 500-tick tail
# warm-up activates and the SplicedDist path is exercised in-package.
let r = LCG(53)
    f = laplace(k = 1)
    st = nothing
    dists = nothing
    spliced_seen = false
    for i in 1:3000
        y = gauss!(r)
        dists, st = f(y, st)
        if dists[1] isa SplicedDist
            spliced_seen = true
        end
        if (i - 1) % 500 == 0 && i > 1
            assert_wellformed(dists[1], y, "soak3000")
        end
    end
    d = dists[1]
    check(spliced_seen, "soak3000: splice never activated")
    check(d isa SplicedDist, "soak3000: final predictive not spliced")
    if d isa SplicedDist
        check(isfinite(d.t_lo) && isfinite(d.t_up) && d.t_lo < d.t_up,
            "soak3000: splice thresholds malformed")
        check(0.0 < d.zeta_lo < 1.0 && 0.0 < d.zeta_up < 1.0,
            "soak3000: splice tail masses out of (0,1)")
        check(isfinite(d.g_lo) && d.s_lo > 0.0 && isfinite(d.g_up) && d.s_up > 0.0,
            "soak3000: GPD parameters malformed")
    end
    assert_wellformed(d, 0.0, "soak3000:final")
    println("ok   soak3000 (splice active)")
end

if failures > 0
    println("ROBUSTNESS FAILED: $failures violation(s)")
    error("ROBUSTNESS FAILED: $failures violation(s)")
end
println("ROBUSTNESS OK (constant, lattice, spike, extreme, collapse, whiplash, soak)")
