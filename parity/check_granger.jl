# granger Julia parity against the Python reference (skaters/parity/granger_vectors.json).
#   julia --project=. parity/check_granger.jl ../skaters/parity/granger_vectors.json
using Skaters, JSON
const ATOL = 1e-6
const RTOL = 1e-6
V = JSON.parsefile(isempty(ARGS) ? "../skaters/parity/granger_vectors.json" : ARGS[1])
worst = 0.0
bad = 0
for (key, st) in V
    freq = split(key, "/")[1]
    g = granger(granger_params(String(freq)))
    ys = Float64[v === nothing ? NaN : Float64(v) for v in st["ys"]]   # null = missing
    rows = st["rows"]
    s = nothing
    for t in 1:(length(ys)-1)
        d, s = g(ys[t], s)
        m = d[1]
        got = [Skaters.dist_logpdf(m, ys[t+1]), Skaters.dist_mean(m), Skaters.dist_std(m), m.w[1],
               Skaters.dist_quantile(m, 0.1), Skaters.dist_quantile(m, 0.9)]
        want = rows[t]
        for i in eachindex(got)
            want[i] === nothing && continue        # no score at a missing next value
            w = Float64(want[i]); dev = abs(got[i] - w)
            global worst = max(worst, dev / max(1.0, abs(w)))
            if dev > ATOL + RTOL * abs(w)
                global bad += 1
                bad <= 5 && println("$key step $(t-1) q$i: julia $(got[i]) python $(w)")
            end
        end
    end
end
println("[granger julia parity] worst relative deviation $(round(worst; sigdigits=3)), failures $bad")
exit(bad == 0 ? 0 : 1)
