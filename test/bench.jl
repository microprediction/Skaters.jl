# Quick throughput benchmark: laplace(k=1) microseconds per tick over 3000 ticks.
# Run:  julia --project=. test/bench.jl
using Skaters

function make_series(n::Int)
    s = zeros(Float64, n)
    lvl = 0.0
    x = 0.123456789
    for t in 1:n
        x = mod(1103515245.0 * x + 12345.0, 2147483648.0)   # deterministic LCG
        u = x / 2147483648.0
        lvl += 0.3 * (u - 0.5)
        s[t] = lvl + 2.0 * sin(2 * pi * t / 7) + (u - 0.5)
    end
    return s
end

const N = 3000
series = make_series(N)

# Warm up (JIT compile the full laplace path).
let f = laplace(k = 1), st = nothing
    for i in 1:50
        _, st = f(series[i], st)
    end
end

f = laplace(k = 1)
st = nothing
t0 = time_ns()
for y in series
    global st
    _, st = f(y, st)
end
elapsed = (time_ns() - t0) / 1e9
us_per_tick = elapsed / N * 1e6
println("laplace(1): $N ticks in $(round(elapsed, digits=4)) s")
println("$(round(us_per_tick, digits=2)) us/tick")
