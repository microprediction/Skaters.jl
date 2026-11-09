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
    ys = Float64.(st["ys"]); rows = st["rows"]
    s = nothing
    for t in 1:(length(ys)-1)
        d, s = g(ys[t], s)
        m = d[1]
        got = [Skaters.dist_logpdf(m, ys[t+1]), Skaters.dist_mean(m), Skaters.dist_std(m), m.w[1],
               Skaters.dist_quantile(m, 0.1), Skaters.dist_quantile(m, 0.9)]
        want = Float64.(rows[t])
        for i in eachindex(got)
            dev = abs(got[i] - want[i])
            global worst = max(worst, dev / max(1.0, abs(want[i])))
            if dev > ATOL + RTOL * abs(want[i])
                global bad += 1
                bad <= 5 && println("$key step $(t-1) q$i: julia $(got[i]) python $(want[i])")
            end
        end
    end
end
println("[granger julia parity] worst relative deviation $(round(worst; sigdigits=3)), failures $bad")
exit(bad == 0 ? 0 : 1)
