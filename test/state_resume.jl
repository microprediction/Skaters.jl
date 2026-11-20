# Cross-language resume in the canonical state layout (skaters docs/STATE.md).
#
# parity/state_vectors.json holds, for every scenario of
# test/state_scenarios.jl, Python checkpoints written with state_to_json at
# t = 60, 250 and 530 and the probes of the next 20 Python outputs. For each:
#
#   1. Julia loads the checkpoint with state_from_json and steps 20; every
#      output must match Python to 1e-6 and a Julia run from scratch to 1e-6;
#   2. the Julia run from scratch is checkpointed at the same t through
#      state_to_json, JSON text and state_from_json, and the restored copy
#      must continue bit for bit;
#   3. the Julia checkpoint has the same layout as Python's: the same keys at
#      every level and the same JSON kinds (numbers, null, lists, objects).
#
# Run: julia --project=. test/state_resume.jl
using JSON
using Skaters

include(joinpath(@__DIR__, "state_scenarios.jl"))

v = JSON.parsefile(joinpath(@__DIR__, "..", "parity", "state_vectors.json"))
const PROBE = Float64(v["probe"])
const Q_LO = Float64(v["q_lo"])
const Q_HI = Float64(v["q_hi"])
const CONTINUE = Int(v["continue"])
const ATOL = 1e-6
const RTOL = 1e-6

parseexp(x::AbstractString) = x == "nan" ? NaN : x == "inf" ? Inf : x == "-inf" ? -Inf : error("bad sentinel $x")
parseexp(x) = Float64(x)
series = Dict(k => Float64[parseexp(x) for x in xs] for (k, xs) in v["series"])

probe(d) = (dist_mean(d), dist_std(d), dist_logpdf(d, PROBE), dist_cdf(d, PROBE),
            dist_quantile(d, Q_LO), dist_quantile(d, Q_HI), dist_crps(d, PROBE))

close_to(g, w) = (isnan(w) && isnan(g)) || (isinf(w) && g == w) ||
                 (isfinite(w) && abs(g - w) <= ATOL + RTOL * abs(w))
same_bits(a, b) = reinterpret(UInt64, a) == reinterpret(UInt64, b)

jkind(x) = x === nothing ? "null" : x isa Bool ? "bool" : x isa Real ? "number" :
           x isa AbstractString ? "string" : x isa AbstractVector ? "list" : "object"

function layout_diff!(out, a, b, path)
    length(out) >= 5 && return
    ka, kb = jkind(a), jkind(b)
    if ka != kb
        push!(out, "$path: julia $ka, python $kb")
    elseif a isa AbstractDict
        sa, sb = Set(keys(a)), Set(keys(b))
        sa == sb || push!(out, "$path: keys differ, julia only $(setdiff(sa, sb)), python only $(setdiff(sb, sa))")
        for key in intersect(sa, sb)
            layout_diff!(out, a[key], b[key], "$path.$key")
        end
    elseif a isa AbstractVector
        length(a) == length(b) || push!(out, "$path: length julia $(length(a)), python $(length(b))")
        for i in 1:min(length(a), length(b))
            layout_diff!(out, a[i], b[i], "$path[$(i-1)]")
        end
    end
end

fails = 0
checked = 0
function fail(msg)
    global fails
    fails += 1
    fails <= 15 && println("FAIL: ", msg)
end

registry = state_scenarios()
names_here = Set(first.(registry))
for name in keys(v["scenarios"])
    name in names_here || fail("scenario $name in the vectors but not in the registry")
end

only = isempty(ARGS) ? nothing : Set(ARGS)
for (name, mk, sname) in registry
    global checked
    only !== nothing && !(name in only) && continue
    haskey(v["scenarios"], name) || (fail("$name missing from the vectors"); continue)
    ys = series[sname]
    cps = v["scenarios"][name]["checkpoints"]
    # One Julia run from scratch, keeping the probes of every step and a
    # JSON checkpoint at each checkpoint step.
    f = mk()
    st = nothing
    scratch = Dict{Int,Any}()
    jcheck = Dict{Int,String}()
    cp_ts = Set(Int(c["t"]) for c in cps)
    last_t = maximum(cp_ts) + CONTINUE
    for t in 0:last_t
        d, st = f(ys[t+1], st)
        scratch[t] = [probe(x) for x in d]
        if t in cp_ts
            jcheck[t] = JSON.json(state_to_json(st))
        end
    end
    bad = 0
    try
    for c in cps
        t = Int(c["t"])
        # 3. layout
        lay = String[]
        layout_diff!(lay, JSON.parse(jcheck[t]), c["state"], "")
        isempty(lay) || (bad += 1; fail("$name t=$t layout: " * join(lay, "; ")))
        # 1. resume from Python's checkpoint
        g = mk()
        sp = state_from_json(c["state"])
        # 2. resume from Julia's own JSON checkpoint
        h = mk()
        sj = state_from_json(JSON.parse(jcheck[t]))
        for m in 1:CONTINUE
            y = ys[t+m+1]
            dp, sp = g(y, sp)
            dj, sj = h(y, sj)
            got = [probe(x) for x in dp]
            want = c["next"][m]
            own = scratch[t+m]
            length(got) == length(want) || (bad += 1; fail("$name t=$t +$m: horizon count"); break)
            for hh in eachindex(got), j in 1:7
                checked += 1
                w = parseexp(want[hh][j])
                if !close_to(got[hh][j], w)
                    bad += 1
                    fail("$name t=$t +$m h$hh probe $j: resumed from Python $(got[hh][j]), Python $w")
                end
                if !close_to(got[hh][j], own[hh][j])
                    bad += 1
                    fail("$name t=$t +$m h$hh probe $j: resumed from Python $(got[hh][j]), Julia from scratch $(own[hh][j])")
                end
            end
            jj = [probe(x) for x in dj]
            if !all(same_bits(jj[hh][j], own[hh][j]) for hh in eachindex(jj), j in 1:7)
                bad += 1
                fail("$name t=$t +$m: Julia JSON checkpoint does not resume bit for bit")
            end
        end
    end
    catch err
        bad += 1
        fail("$name: " * sprint(showerror, err)[1:min(end, 400)])
    end
    println(bad == 0 ? "ok   " : "FAIL ", name, " (checkpoints at ", join(sort(collect(cp_ts)), ", "), ")")
end

# --- older Julia checkpoints still resume ---
# Before the canonical layout this port kept sticky's table as parallel
# vals/wts vectors, Int-keyed multiscale tables, the period detector's cross
# as a vector in lag order, granger's last pair, cum and z as scalars, and no
# terminal "adapt" field. Each skater rewrites its part on its next step, so a
# state rewritten into that layout must continue bit for bit.
function to_legacy(x)
    x isa AbstractVector && return Any[to_legacy(v) for v in x]
    x isa AbstractDict || return x
    d = Dict{String,Any}(k => to_legacy(v) for (k, v) in x)
    if haskey(d, "counts") && haskey(d, "base")
        d["vals"] = Float64[p[1] for p in d["counts"]]
        d["wts"] = Float64[p[2] for p in d["counts"]]
        delete!(d, "counts")
    end
    if all(haskey(d, k) for k in ("t", "phase", "pending", "latest", "score"))
        for k in ("phase", "pending", "latest", "score")
            d[k] = Dict{Int,Any}(parse(Int, s) => v for (s, v) in d[k])
        end
    end
    if haskey(d, "cross") && d["cross"] isa AbstractDict
        d["cross"] = Float64[d["cross"][string(L)] for L in Skaters._DEFAULT_LAGS]
    end
    if all(haskey(d, k) for k in ("lap", "learned", "pend", "cum"))
        d["pend"] = isempty(d["pend"]) ? nothing : Any[d["pend"][end][1][1], d["pend"][end][2][1]]
        d["cum"] = Float64[d["cum"][1][1][1], d["cum"][1][1][2]]
        d["z"] = d["z"][1]
    end
    if haskey(d, "qdist") && haskey(d, "adapt") && d["adapt"] === nothing
        delete!(d, "adapt")
    end
    return d
end

for (name, stop) in (("laplace", 530), ("laplace_k3", 250), ("granger", 250),
                     ("sticky_ema_repeat", 250), ("dantzig", 250), ("multiscale_k4", 250))
    global checked
    only !== nothing && !(name in only) && continue
    mk, sname = Dict(n => (m, s) for (n, m, s) in registry)[name]
    ys = series[sname]
    f = mk()
    st = nothing
    for t in 0:stop
        _, st = f(ys[t+1], st)
    end
    legacy = to_legacy(state_from_json(JSON.parse(JSON.json(state_to_json(st)))))
    g = mk()
    ok = true
    for m in 1:CONTINUE
        da, st = f(ys[stop+m+1], st)
        db, legacy = g(ys[stop+m+1], legacy)
        pa, pb = [probe(x) for x in da], [probe(x) for x in db]
        checked += 7 * length(pa)
        if !all(same_bits(pa[h][j], pb[h][j]) for h in eachindex(pa), j in 1:7)
            ok = false
            fail("$name: an older Julia checkpoint does not resume bit for bit at +$m")
            break
        end
    end
    # compare parsed JSON: Dict iteration order is not part of the layout
    ok && isequal(JSON.parse(JSON.json(state_to_json(st))), JSON.parse(JSON.json(state_to_json(legacy)))) ||
        (ok && (ok = false; fail("$name: an older Julia checkpoint is not rewritten to the canonical layout")))
    println(ok ? "ok   " : "FAIL ", "older Julia layout: $name")
end

# --- the walkers (mirrors the reference's tests/test_state_json.py) ---
function throws_with(f, needle)
    try
        f()
    catch err
        return err isa ArgumentError && occursin(needle, sprint(showerror, err))
    end
    return false
end
if only === nothing
    throws_with(() -> state_to_json(Dict("a" => Dict("b" => [1.0, NaN]))), "at .a.b[1]: non-finite") ||
        fail("walker: non-finite path")
    throws_with(() -> state_to_json(Dict("counts" => Dict(1.5 => 0.1))), "dict key 1.5") ||
        fail("walker: non-string key")
    throws_with(() -> state_to_json(Dict("f" => sin)), "a function") || fail("walker: function")
    throws_with(() -> assert_plain_state(Dict("p" => Set([7]))), "an instance of Set") || fail("walker: set")
    state_to_json(Dict("x" => (1, 2.5))) == Dict("x" => Any[1, 2.5]) || fail("walker: tuple is a list")
    # Dists round-trip exactly and are not renormalised; a loose dict is normalised.
    let r = UInt32(7)
        nxt() = (r = UInt32(1664525) * r + UInt32(1013904223); Float64(r) / 4294967296.0)
        for _ in 1:300
            comps = NTuple{3,Float64}[(nxt() + 1e-3, 8 * nxt() - 4, 0.1 + nxt()) for _ in 1:(1 + trunc(Int, 9 * nxt()))]
            d = Dist(comps)
            back = state_from_json(JSON.parse(JSON.json(state_to_json(d))))
            back.comps == d.comps || (fail("Dist encoding is not exact"); break)
        end
    end
    loose = state_from_json(Dict("components" => Any[Any[1.0, 0.0, 1.0], Any[3.0, 1.0, 1.0]]))
    loose.comps[1][1] == 0.25 || fail("a loose components dict is not normalised")
    # laplace's GPD splice is live in the state at t = 530
    let f = laplace(k = 1), st = nothing
        for t in 0:530
            _, st = f(series["series"][t+1], st)
        end
        text = JSON.json(state_to_json(st))
        occursin("\"spliced\":true", text) || fail("no SplicedDist in laplace's state at t = 530")
        back = state_from_json(JSON.parse(text))
        back["pending"][end][1] isa SplicedDist || fail("SplicedDist does not restore as a SplicedDist")
    end
    println("ok   walkers, Dist encoding, SplicedDist in state")
end

println("$checked values checked")
fails == 0 || error("STATE RESUME FAILED: $fails check(s)")
println("STATE RESUME OK")
