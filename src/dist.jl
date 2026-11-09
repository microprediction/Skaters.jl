# Gaussian mixture distribution: the distributional prediction type.
# Faithful port of skaters/dist.py.
#
# A Dist is a list of components [(weight, mean, std), ...]. Weights are
# positive and normalised to sum to 1 (via fsum, exactly as Python's __init__).

abstract type AbstractDist end

struct Dist <: AbstractDist
    comps::Vector{NTuple{3,Float64}}
    function Dist(comps::Vector{NTuple{3,Float64}})
        # Validation as in the reference (skaters 21f1de5): weights finite and
        # >= 0, means finite, stds finite and >= 0 (0 is a point mass), positive
        # total. Thrown errors, not @assert, so the checks always run.
        isempty(comps) && throw(ArgumentError("Dist needs at least one component"))
        for (w, m, s) in comps
            (isfinite(w) && w >= 0.0) ||
                throw(ArgumentError("component weight must be finite and >= 0, got $w"))
            isfinite(m) || throw(ArgumentError("component mean must be finite, got $m"))
            (isfinite(s) && s >= 0.0) ||
                throw(ArgumentError("component std must be finite and >= 0, got $s"))
        end
        wt = fsum(c[1] for c in comps)
        wt > 0.0 || throw(ArgumentError("total component weight must be positive"))
        new([(c[1] / wt, c[2], c[3]) for c in comps])
    end
end

# --- Constructors ---

dist_gaussian(mean::Float64 = 0.0, std::Float64 = 1.0) = Dist([(1.0, mean, std)])

function dist_combine(dists::Vector, weights::Union{Nothing,Vector{Float64}} = nothing)
    n = length(dists)
    n == 0 && throw(ArgumentError("combine needs at least one distribution"))
    if weights === nothing
        weights = fill(1.0 / n, n)
    end
    length(weights) == n ||
        throw(ArgumentError("combine got $n distributions but $(length(weights)) weights"))
    w_total = fsum(weights)
    (isfinite(w_total) && w_total > 0.0) ||
        throw(ArgumentError("combine weights must be finite with a positive total"))
    comps = NTuple{3,Float64}[]
    for (d, w_outer) in zip(dists, weights)
        for (w_inner, m, s) in d.comps
            push!(comps, (w_outer / w_total * w_inner, m, s))
        end
    end
    return Dist(comps)
end

# --- Queries ---

function dist_pdf(d::Dist, x::Float64)::Float64
    total = 0.0
    for (w, m, s) in d.comps
        w <= 0.0 && continue    # skaters#222: 0 * inf at a discarded (s=0) atom is NaN, not 0
        total += w * gaussian_pdf(x, m, s)
    end
    return total
end

function dist_logpdf(d::Dist, x::Float64)::Float64
    best = -Inf
    terms = Float64[]
    for (w, m, s) in d.comps
        if w <= 0.0
            continue
        end
        if s <= 0.0
            if x == m
                return Inf
            end
            continue
        end
        z = (x - m) / s
        t = log(w) - 0.5 * z * z - log(s) - LOG_SQRT2PI
        push!(terms, t)
        if t > best
            best = t
        end
    end
    if best == -Inf
        return -Inf
    end
    return best + log(fsum(exp(t - best) for t in terms))
end

function dist_cdf(d::Dist, x::Float64)::Float64
    total = 0.0
    for (w, m, s) in d.comps
        total += w * gaussian_cdf(x, m, s)
    end
    return total
end

function dist_crps(d::Dist, x::Float64)::Float64
    t1 = 0.0
    for (w, m, s) in d.comps
        t1 += w * abs_expectation(m - x, s)
    end
    t2 = 0.0
    for (wi, mi, si) in d.comps
        for (wj, mj, sj) in d.comps
            t2 += wi * wj * abs_expectation(mi - mj, sqrt(si * si + sj * sj))
        end
    end
    return t1 - 0.5 * t2
end

function dist_mean(d::Dist)::Float64
    return fsum(w * m for (w, m, _) in d.comps)
end

function dist_var(d::Dist)::Float64
    mu = dist_mean(d)
    return fsum(w * (s * s + (m - mu) * (m - mu)) for (w, m, s) in d.comps)
end

function dist_std(d::Dist)::Float64
    v = dist_var(d)
    return v > 0 ? sqrt(v) : 0.0
end

function dist_quantile(d::Dist, p::Float64; tol::Float64 = 1e-9, max_iter::Int = 100)::Float64
    0 < p < 1 || throw(ArgumentError("quantile needs 0 < p < 1, got $p"))
    mu = dist_mean(d)
    sigma = sqrt(dist_var(d))
    lo = mu - 8 * sigma
    hi = mu + 8 * sigma
    # Expand the bracket geometrically until it contains p (skaters 21f1de5); it
    # never fires when mu +- 8 sigma already sufficed, so results are unchanged.
    width = max(16 * sigma, 1e-12)
    ok = false
    for _ in 1:120
        if dist_cdf(d, lo) <= p
            ok = true
            break
        end
        lo -= width
        width *= 2.0
    end
    ok || error("could not bracket quantile p=$p from below")
    width = max(16 * sigma, 1e-12)
    ok = false
    for _ in 1:120
        if dist_cdf(d, hi) >= p
            ok = true
            break
        end
        hi += width
        width *= 2.0
    end
    ok || error("could not bracket quantile p=$p from above")
    for _ in 1:max_iter
        mid = 0.5 * (lo + hi)
        if dist_cdf(d, mid) < p
            lo = mid
        else
            hi = mid
        end
        if hi - lo < tol
            break
        end
    end
    return 0.5 * (lo + hi)
end

# --- Transform support ---

dist_shift(d::Dist, delta::Float64) = Dist([(w, m + delta, s) for (w, m, s) in d.comps])

function dist_scale(d::Dist, factor::Float64)
    factor == 0 && throw(ArgumentError("scale factor must be nonzero"))
    f = abs(factor)
    return Dist([(w, m * factor, s * f) for (w, m, s) in d.comps])
end

function dist_affine(d::Dist, a::Float64, b::Float64)
    a == 0 && throw(ArgumentError("affine multiplier must be nonzero"))
    return Dist([(w, a * m + b, abs(a) * s) for (w, m, s) in d.comps])
end

# body accessor: plain Dist is its own body (SplicedDist overrides).
dist_body(d::Dist) = d

# --- Pruning (ulp-tolerant closest-pair merge; exact port of Dist.prune) ---

function dist_prune(d::Dist, max_components::Int = 20)
    max_components = max(1, max_components)
    if length(d.comps) <= max_components
        return d
    end
    # Sort so the merge path is order-independent: by (mean, std, weight).
    comps = sort(d.comps, by = c -> (c[2], c[3], c[1]))
    # Translation-invariant (skaters#226): sorted by mean, so this is the span.
    scale = (comps[end][2] - comps[1][2]) + 1e-12
    while length(comps) > max_components
        n = length(comps)
        best_dist = Inf
        for i in 1:n
            for j in (i+1):n
                dd = abs(comps[i][2] - comps[j][2])
                if dd < best_dist
                    best_dist = dd
                end
            end
        end
        thresh = best_dist + 1e-9 * scale
        best_i = 0
        best_j = 0
        found = false
        for i in 1:n
            for j in (i+1):n
                if abs(comps[i][2] - comps[j][2]) <= thresh
                    best_i = i
                    best_j = j
                    found = true
                    break
                end
            end
            if found
                break
            end
        end
        if !found
            best_i = 1
            best_j = 2
        end
        wi, mi, si = comps[best_i]
        wj, mj, sj = comps[best_j]
        w_new = wi + wj
        if w_new < 1e-300
            m_new = 0.5 * (mi + mj)
            s_new = max(si, sj, 1e-12)
        else
            m_new = (wi * mi + wj * mj) / w_new
            v_new = (wi * (si * si + (mi - m_new)^2) + wj * (sj * sj + (mj - m_new)^2)) / w_new
            s_new = sqrt(max(v_new, 0.0))
        end
        comps[best_i] = (w_new, m_new, s_new)
        deleteat!(comps, best_j)
    end
    return Dist(comps)
end
