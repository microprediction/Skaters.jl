# Exact pushforward of a predictive through a monotone coordinate map.
# Port of the parts of skaters/pushforward.py that the transform inverses use:
# a z-space predictive, a strictly increasing (ys, zs) table of the map, and
# the analytic inverse y = inv_fn(z). Queries go through the change of
# variables
#
#     F_Y(y)      = F_Z(z(y))
#     log p_Y(y)  = log p_Z(z(y)) + log dz/dy
#     Q_Y(p)      = y(Q_Z(p))
#
# The reference's analytic-forward options (fwd_fn, log_jacobian_fn,
# component_moments_fn, scaled_inv_fn) are not ported; nothing here builds one.

# 7-point Gauss-Hermite nodes and weights (physicists'), for E[g(Z)], Z ~ N(0,1).
const _GH7 = (
    (-2.6519613568352334, 9.717812450995192e-04),
    (-1.6735516287674714, 5.451558281912703e-02),
    (-0.8162878828589647, 4.256072526101278e-01),
    (0.0, 8.102646175568073e-01),
    (0.8162878828589647, 4.256072526101278e-01),
    (1.6735516287674714, 5.451558281912703e-02),
    (2.6519613568352334, 9.717812450995192e-04),
)
const _SQRTPI = sqrt(pi)

# Fourth-power odds give the tails finer resolution under nonlinear maps.
function _quantile_rule(n::Int)
    edges = Float64[i^4 / (i^4 + (n - i)^4) for i in 0:n]
    return [((edges[j] + edges[j+1]) * 0.5, edges[j+1] - edges[j]) for j in 1:n]
end

const _QUANTILE_RULE = _quantile_rule(257)

# Piecewise-linear with linear extrapolation from the end segments.
function interp_table(x::Float64, xs::Vector{Float64}, fs::Vector{Float64})::Float64
    i = searchsortedfirst(xs, x) - 1        # Python bisect_left, 0-based
    if i <= 0
        i = 1
    elseif i >= length(xs)
        i = length(xs) - 1
    end
    x0, x1 = xs[i], xs[i+1]
    return fs[i] + (fs[i+1] - fs[i]) * (x - x0) / (x1 - x0)
end

# (ys, zs) knots of a monotone map y = inv_fn(z) on [z_lo, z_hi]. Knots where
# inv_fn fails to strictly increase (float ties on flat stretches) are dropped
# so every segment has a positive, finite slope.
function table_from_map(inv_fn, z_lo::Float64, z_hi::Float64, nodes::Int = 400)
    z_hi = max(z_hi, z_lo + 1e-9)
    step = (z_hi - z_lo) / nodes
    ys = Float64[]
    zs = Float64[]
    for j in 0:nodes
        z = z_lo + step * j
        y = inv_fn(z)
        if isempty(ys) || y > ys[end]
            push!(ys, y)
            push!(zs, z)
        end
    end
    if length(ys) < 2
        y0 = inv_fn(z_lo)
        ys, zs = [y0, y0 + 1e-9], [z_lo, z_lo + 1e-9]
    end
    return ys, zs
end

mutable struct PushforwardDist <: AbstractDist
    inner::Any
    ys::Vector{Float64}
    zs::Vector{Float64}
    inv_fn::Any
    _comps::Union{Nothing,Vector{NTuple{3,Float64}}}
    _grid::Union{Nothing,Vector{Float64}}
    PushforwardDist(inner, ys, zs, inv_fn) = new(inner, ys, zs, inv_fn, nothing, nothing)
end

# No clamp (skaters#221): the table extrapolates linearly past its ends.
_pf_z(d::PushforwardDist, y::Float64) = interp_table(y, d.ys, d.zs)
_pf_y(d::PushforwardDist, z::Float64) = d.inv_fn(z)

# dz/dy at y: central difference of the exact inverse, segment secant otherwise.
function _pf_slope(d::PushforwardDist, y::Float64)::Float64
    z = _pf_z(d, y)
    h = 1e-5 * (1.0 + abs(z))
    dy = d.inv_fn(z + h) - d.inv_fn(z - h)
    if dy > 0.0
        return 2.0 * h / dy
    end
    ys, zs = d.ys, d.zs
    i = searchsortedfirst(ys, y) - 1
    i = min(max(i, 1), length(ys) - 1)
    return (zs[i+1] - zs[i]) / (ys[i+1] - ys[i])
end

function dist_logpdf(d::PushforwardDist, x::Float64)::Float64
    z = _pf_z(d, x)
    isinf(z) && return -Inf
    isnan(z) && return NaN
    return dist_logpdf(d.inner, z) + log(_pf_slope(d, x))
end

function dist_pdf(d::PushforwardDist, x::Float64)::Float64
    lp = dist_logpdf(d, x)
    return lp < 700.0 ? exp(lp) : Inf
end

function dist_cdf(d::PushforwardDist, x::Float64)::Float64
    z = _pf_z(d, x)
    isinf(z) && return z < 0 ? 0.0 : 1.0
    isnan(z) && return NaN
    return dist_cdf(d.inner, z)
end

dist_quantile(d::PushforwardDist, p::Float64; tol::Float64 = 1e-9, max_iter::Int = 2100)::Float64 =
    _pf_y(d, dist_quantile(d.inner, p; tol = tol, max_iter = max_iter))

# CRPS by trapezoid quadrature over the table in z-space.
function dist_crps(d::PushforwardDist, x::Float64)::Float64
    isfinite(x) || return abs(x)
    zstar = _pf_z(d, x)
    zs = d.zs
    s_lo = 1.0 / _pf_slope(d, d.ys[1])
    s_hi = 1.0 / _pf_slope(d, d.ys[end])
    base = Float64[zs[1] - 8.0, zs[1] - 6.0, zs[1] - 4.0, zs[1] - 2.0, zs[1] - 1.0]
    append!(base, zs)
    append!(base, [zs[end] + 1.0, zs[end] + 2.0, zs[end] + 4.0, zs[end] + 6.0, zs[end] + 8.0])
    append!(base, [zstar - 1.0, zstar, zstar + 1.0])
    base = sort(unique(base))
    nodes = Float64[base[1]]
    for z in base[2:end]
        lo = nodes[end]
        gap = z - lo
        if gap > 0.1
            pieces = min(trunc(Int, gap / 0.1) + 1, 40)
            step = gap / pieces
            for j in 0:(pieces-2)
                push!(nodes, lo + step * (j + 1))
            end
        end
        push!(nodes, z)
    end
    total = 0.0
    prev_z = nodes[1]
    c_prev = dist_cdf(d.inner, prev_z)
    for z in nodes[2:end]
        c = dist_cdf(d.inner, z)
        ind = prev_z >= zstar ? 1.0 : 0.0
        g0 = (c_prev - ind)^2
        g1 = (c - ind)^2
        if z <= zs[1]
            yp = s_lo
        elseif prev_z >= zs[end]
            yp = s_hi
        else
            yp = (_pf_y(d, z) - _pf_y(d, prev_z)) / (z - prev_z)
        end
        total += 0.5 * (g0 + g1) * (z - prev_z) * yp
        prev_z, c_prev = z, c
    end
    return total
end

# Mapped quantiles on the quantile rule, cached.
function _qgrid(d::PushforwardDist)::Vector{Float64}
    if d._grid === nothing
        d._grid = Float64[dist_quantile(d, p) for (p, _) in _QUANTILE_RULE]
    end
    return d._grid
end

# Gaussian moments of a mapped component, centered Gauss-Hermite (skaters#225).
function _pf_gaussian_moments(d::PushforwardDist, m::Float64, s::Float64)
    node_ys = [_pf_y(d, m + s * SQRT2 * x) for (x, _) in _GH7]
    mu = fsum((w / _SQRTPI) * y for (y, (_, w)) in zip(node_ys, _GH7))
    isfinite(mu) || return mu, Inf
    sd = hypot((sqrt(w / _SQRTPI) * (y - mu) for (y, (_, w)) in zip(node_ys, _GH7))...)
    return mu, max(sd, 1e-12)
end

# Moment-matched mixture components in y-space (lazy, cached). An inner
# without Gaussian components (SplicedDist, PushforwardDist) gets one Gaussian
# fitted to its mapped quantiles.
function _pf_components(d::PushforwardDist)::Vector{NTuple{3,Float64}}
    if d._comps === nothing
        if d.inner isa Dist
            d._comps = NTuple{3,Float64}[(w, _pf_gaussian_moments(d, m, s)...)
                                         for (w, m, s) in d.inner.comps if w > 0.0]
        else
            q = _qgrid(d)
            mu = fsum(w * y for (y, (_, w)) in zip(q, _QUANTILE_RULE))
            sd = !isfinite(mu) ? Inf :
                 hypot((sqrt(w) * (y - mu) for (y, (_, w)) in zip(q, _QUANTILE_RULE))...)
            d._comps = NTuple{3,Float64}[(1.0, mu, sd)]
        end
    end
    return d._comps
end

dist_mean(d::PushforwardDist)::Float64 = fsum(w * m for (w, m, _) in _pf_components(d))

function dist_std(d::PushforwardDist)::Float64
    comps = _pf_components(d)
    length(comps) == 1 && return comps[1][3]
    mu = dist_mean(d)
    isfinite(mu) || return Inf
    return hypot((sqrt(w) * hypot(s, m - mu) for (w, m, s) in comps)...)
end

function dist_var(d::PushforwardDist)::Float64
    sd = dist_std(d)
    return sd * sd
end

# The exact pushforward of a predictive with no mixture to read (laplace's
# tail-spliced SplicedDist) through the monotone map inv_fn. The table window
# pads the extreme quantiles by their own span on each side.
function _pushforward(d, inv_fn)
    q_lo = dist_quantile(d, 1e-6)
    q_hi = dist_quantile(d, 1.0 - 1e-6)
    lo, hi = 2.0 * q_lo - q_hi, 2.0 * q_hi - q_lo
    ys, zs = table_from_map(inv_fn, lo, hi)
    return PushforwardDist(d, ys, zs, inv_fn)
end
