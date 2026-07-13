# Adaptive search over the transform tree. Port of skaters/search.py.
#
# Beam search over the transform grammar: score candidates by cumulative
# clamped log-likelihood, expand top performers with new transforms
# (replaying recent history so children join warm), prune losers.
# Named adaptive_search to match the R port.
#
# One deliberate departure from the Python reference, which stores live
# callables in the state: here the state holds only recipes and plain data
# (the package convention), and candidate skaters are rebuilt on demand
# from their recipes through a per-instance memo. Rebuilding is
# deterministic, so values match the reference exactly and the state
# survives Serialization round-trips.

# The grammar: (name, factory, cost) triples.
const _SEARCH_TRANSFORMS = Tuple{String,Function,Float64}[
    ("ema_t(0.05)", () -> ema_transform(0.05), 1.0),
    ("ema_t(0.1)", () -> ema_transform(0.1), 1.0),
    ("ema_t(0.3)", () -> ema_transform(0.3), 1.0),
    ("diff", () -> difference(), 1.0),
    ("std(0.05)", () -> standardize(0.05), 1.0),
    ("frac(0.3)", () -> fractional_difference(0.3, 30), 3.0),
    ("garch", () -> garch(), 1.0),
    ("pow(0.5)", () -> power_transform(0.5), 1.0),
    ("ar(2)", () -> ar(2), 2.0),
    ("ar(5)", () -> ar(5; decay = 1.0), 3.0),
    ("gar(16)", () -> grouped_ar(16), 2.0),
    ("theta(0.1)", () -> theta(0.1), 1.0),
    ("theta(0.3)", () -> theta(0.3), 1.0),
    ("drift", () -> drift(), 1.0),
    ("drift(0.01)", () -> drift(0.01, 0.002), 1.0),
    ("holt(0.1,0.05)", () -> holt_linear(0.1, 0.05), 1.0),
    ("holt(0.3,0.1)", () -> holt_linear(0.3, 0.1), 1.0),
    ("seas(7)", () -> seasonal_difference(7), 1.0),
    ("seas(12)", () -> seasonal_difference(12), 1.0),
    ("seas(24)", () -> seasonal_difference(24), 1.0),
]

# The grammar in play: base transforms plus one seasonal per detected
# period, in detection order (derived from state, never stored in it).
function _search_transforms(detected_periods::Vector{Int})
    out = copy(_SEARCH_TRANSFORMS)
    for p in detected_periods
        push!(out, ("seas($p)", () -> seasonal_difference(p), 2.0))
    end
    return out
end

function _search_entry(depth::Int, recipe::Vector{String}, k::Int, cost::Float64)
    return Dict{String,Any}("s" => nothing, "depth" => depth, "recipe" => copy(recipe),
        "cost" => cost, "age" => 0, "warmed" => false, "log_w" => zeros(k),
        "queues" => Any[Any[] for _ in 1:k], "dists" => nothing)
end

function _search_init_pool(k::Int, cost_budget::Float64)
    pool = Any[]
    e = _search_entry(0, String[], k, 1.0)
    e["warmed"] = true
    push!(pool, e)
    for (t_name, _, t_cost) in _SEARCH_TRANSFORMS
        cand_cost = 1.0 + t_cost
        cand_cost > cost_budget && continue
        e = _search_entry(1, [t_name], k, cand_cost)
        e["warmed"] = true
        push!(pool, e)
    end
    return pool
end

function _search_build_from_recipe(recipe::Vector{String}, k::Int, transforms)
    lookup = Dict{String,Function}()
    for (t_name, t_factory, _) in transforms
        lookup[t_name] = t_factory       # last wins, as in Python's dict comp
    end
    f = leaf(k)
    for t_name in recipe
        f = conjugate(f, lookup[t_name](), k)
    end
    return f
end

function _search_expand(pool, k::Int, top_n::Int, max_depth::Int, transforms,
                        cost_budget::Float64)
    scores = [fsum(e["log_w"]) / k for e in pool]
    # Python sorts (score, index) tuples reverse=True: score desc, index desc.
    ord = sort(collect(1:length(pool)), by = i -> (scores[i], i), rev = true)
    existing = Set{String}(join(e["recipe"], "|") for e in pool)
    children = Any[]
    for pi in ord[1:min(top_n, length(ord))]
        parent = pool[pi]
        parent["depth"] >= max_depth && continue
        for (t_name, _, t_cost) in transforms
            child_cost = parent["cost"] + t_cost
            child_cost > cost_budget && continue
            recipe = parent["recipe"]::Vector{String}
            if !isempty(recipe) && recipe[end] == t_name
                continue
            end
            new_recipe = vcat(recipe, t_name)
            key = join(new_recipe, "|")
            key in existing && continue
            push!(existing, key)
            push!(children, _search_entry(length(new_recipe), new_recipe, k, child_cost))
        end
    end
    return children
end

function _search_prune!(pool, threshold::Float64, max_pool::Int, k::Int)
    length(pool) <= 1 && return pool
    avg(e) = fsum(e["log_w"]) / k
    best = maximum(avg(e) for e in pool)
    i = 1
    while i <= length(pool)
        if avg(pool[i]) < best + threshold && length(pool) > 1
            deleteat!(pool, i)
        else
            i += 1
        end
    end
    while length(pool) > max_pool
        worst = argmin([avg(e) for e in pool])   # first argmin, as in Python
        deleteat!(pool, worst)
    end
    return pool
end

function adaptive_search(; k::Int = 1, learning_rate::Float64 = 0.5,
                         complexity_penalty::Float64 = 0.02, max_pool::Int = 30,
                         expand_interval::Int = 100, expand_top_n::Int = 3,
                         max_depth::Int = 3, replay_buffer::Int = 500,
                         prune_threshold::Float64 = -50.0, max_components::Int = 20,
                         cost_budget::Float64 = Inf)
    pd_func = period_detector()

    # Per-instance memo: recipe key -> live skater. Skaters are pure
    # functions of the recipe (all mutable state lives in the entry), so a
    # fresh instance resuming a saved state rebuilds identical ones.
    memo = Dict{String,Any}()
    function get_skater(recipe::Vector{String}, detected_periods::Vector{Int})
        key = join(recipe, "|")
        if !haskey(memo, key)
            memo[key] = _search_build_from_recipe(recipe, k,
                _search_transforms(detected_periods))
        end
        return memo[key]
    end

    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("pool" => _search_init_pool(k, cost_budget),
                "n_obs" => 0, "buffer" => Float64[], "pd_state" => nothing,
                "detected_periods" => Int[])
        end
        state["n_obs"] += 1
        buffer = state["buffer"]::Vector{Float64}
        push!(buffer, y)
        if length(buffer) > replay_buffer
            deleteat!(buffer, 1:(length(buffer)-replay_buffer))
        end
        pool = state["pool"]
        periods = state["detected_periods"]::Vector{Int}

        # 1. Run all active candidates.
        for e in pool
            f = get_skater(e["recipe"], periods)
            dists, e["s"] = f(y, e["s"])
            e["dists"] = dists
            e["age"] += 1
        end

        # 2+3. Queue current predictions, resolve matured ones. The Dist
        # issued h+1 steps ago targeted the current y. Bounded loss: clamp
        # logpdf to [-20, 20]; the lower clamp also catches NaN.
        for e in pool
            for h in 1:k
                q = e["queues"][h]
                push!(q, e["dists"][h])
                if length(q) > h
                    past = popfirst!(q)
                    if e["warmed"]
                        lp = dist_logpdf(past, y)
                        if lp > 20.0
                            lp = 20.0
                        elseif !(lp >= -20.0)
                            lp = -20.0
                        end
                        e["log_w"][h] += learning_rate * lp -
                            complexity_penalty * e["depth"]
                    end
                end
            end
        end

        # 3b. Run the period detector.
        scores, state["pd_state"] = pd_func(y, state["pd_state"])

        # 4. Periodically expand and prune.
        if state["n_obs"] % expand_interval == 0 && state["n_obs"] > 10
            detected = top_periods(scores; threshold = 0.3, max_periods = 3)
            for period in detected
                if !(period in periods)
                    push!(periods, period)
                end
            end
            transforms = _search_transforms(periods)
            children = _search_expand(pool, k, expand_top_n, max_depth,
                transforms, cost_budget)
            for child in children
                # Replay recent history through the child so it joins warm.
                f = get_skater(child["recipe"], periods)
                for yb in buffer
                    dists, child["s"] = f(yb, child["s"])
                    child["dists"] = dists
                    child["age"] += 1
                end
                if child["dists"] !== nothing
                    for h in 1:k
                        child["queues"][h] = Any[child["dists"][h]]
                    end
                end
                child["warmed"] = true
                push!(pool, child)
            end
            _search_prune!(pool, prune_threshold, max_pool, k)
        end

        # 5. Combine predictions via softmax weights.
        combined = Any[]
        for h in 1:k
            log_ws = Float64[e["log_w"][h] for e in pool]
            max_lw = maximum(log_ws)
            weights = isfinite(max_lw) ? [exp(lw - max_lw) for lw in log_ws] :
                fill(1.0, length(pool))
            d = dist_combine(Any[e["dists"][h] for e in pool], weights)
            if length(d.comps) > max_components
                d = dist_prune(d, max_components)
            end
            push!(combined, d)
        end

        return combined, state
    end
    return _skater
end
