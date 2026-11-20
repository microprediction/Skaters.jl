# Exact pushforward of a predictive through a monotone coordinate map.
# Port of skaters/pushforward.py.
#
# The component-wise delta method used by the nonlinear transform inverses
# (yeo_johnson, power_transform) maps each mixture component to
# N(inv(m), s * dinv(m)). It cannot carry the skew of the coordinate change,
# which lives in the curvature of the map within a component's width.
# PushforwardDist is the exact treatment: keep the inner predictive in the
# transformed z-coordinate and answer queries through the change of variables
#
#     F_Y(y)      = F_Z(z(y))
#     log p_Y(y)  = log p_Z(z(y)) + log dz/dy
#     Q_Y(p)      = y(Q_Z(p))
#
# The map can be supplied analytically or represented by a strictly
# increasing table. Consumers that need mixture components (dist_combine,
# sticky, a further component-reading transform) read _components, which maps
# each inner Gaussian by moment matching. An inner without Gaussian components
# gets a single Gaussian with moments estimated on a quantile grid.

# 7-point Gauss-Hermite nodes and weights (physicists'), for E[g(Z)], Z ~ N(0,1):
# E[g(Z)] ~= sum w_i/sqrt(pi) * g(sqrt(2) x_i).
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

# Python's math.exp raises on overflow; the reference maps that to inf.
_exp_or_inf(x::Float64) = exp(x)

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

# A z-space predictive pushed through a monotone y <-> z map. Supply a table
# (ys, zs) or an analytic map with inv_fn, fwd_fn and log_jacobian_fn
# (log(dz/dy), evaluated at z). With a table and an optional inverse, the
# Jacobian is estimated by finite differences. component_moments_fn
# optionally gives the mapped mean and standard deviation of a Gaussian
# component. scaled_inv_fn(z, weight) evaluates the inverse times a positive
# weight, useful when the inverse overflows.
mutable struct PushforwardDist <: AbstractDist
    inner::Any
    ys::Union{Nothing,Vector{Float64}}
    zs::Union{Nothing,Vector{Float64}}
    inv_fn::Any
    fwd_fn::Any
    log_jacobian_fn::Any
    component_moments_fn::Any
    scaled_inv_fn::Any
    _comps::Union{Nothing,Vector{NTuple{3,Float64}}}
    _grid::Union{Nothing,Vector{Float64}}
end

PushforwardDist(inner, ys = nothing, zs = nothing; inv_fn = nothing, fwd_fn = nothing,
                log_jacobian_fn = nothing, component_moments_fn = nothing,
                scaled_inv_fn = nothing) =
    PushforwardDist(inner, ys, zs, inv_fn, fwd_fn, log_jacobian_fn,
                    component_moments_fn, scaled_inv_fn, nothing, nothing)

# --- the map ---

# No clamp (skaters#221): interp_table already extrapolates linearly past
# the table ends, which is the map's actual, exact behavior out there.
function _pf_z(d::PushforwardDist, y::Float64)::Float64
    d.fwd_fn !== nothing && return d.fwd_fn(y)
    return interp_table(y, d.ys, d.zs)
end

function _pf_y(d::PushforwardDist, z::Float64)::Float64
    d.inv_fn !== nothing && return d.inv_fn(z)
    return interp_table(z, d.zs, d.ys)
end

# dz/dy at y (central difference of the exact map when available, segment
# secant otherwise; end segments extend beyond the table).
function _pf_slope(d::PushforwardDist, y::Float64)::Float64
    if d.fwd_fn !== nothing
        h = 1e-5 * (1.0 + abs(y))
        za, zb = d.fwd_fn(y + h), d.fwd_fn(y - h)
        if isfinite(za) && isfinite(zb) && za > zb
            return (za - zb) / (2.0 * h)
        end
    end
    if d.inv_fn !== nothing
        z = _pf_z(d, y)
        h = 1e-5 * (1.0 + abs(z))
        dy = d.inv_fn(z + h) - d.inv_fn(z - h)
        if dy > 0.0
            return 2.0 * h / dy
        end
    end
    ys, zs = d.ys, d.zs
    i = searchsortedfirst(ys, y) - 1
    i = min(max(i, 1), length(ys) - 1)
    return (zs[i+1] - zs[i]) / (ys[i+1] - ys[i])
end

# --- queries (exact) ---

function dist_logpdf(d::PushforwardDist, x::Float64)::Float64
    z = _pf_z(d, x)
    isinf(z) && return -Inf                 # x on or past a bound of the map
    isnan(z) && return NaN
    if d.log_jacobian_fn !== nothing
        return dist_logpdf(d.inner, z) + d.log_jacobian_fn(z)
    end
    return dist_logpdf(d.inner, z) + log(_pf_slope(d, x))
end

dist_pdf(d::PushforwardDist, x::Float64)::Float64 = _exp_or_inf(dist_logpdf(d, x))

function dist_cdf(d::PushforwardDist, x::Float64)::Float64
    z = _pf_z(d, x)
    isinf(z) && return z < 0 ? 0.0 : 1.0
    isnan(z) && return NaN
    return dist_cdf(d.inner, z)
end

dist_quantile(d::PushforwardDist, p::Float64; tol::Float64 = 1e-9, max_iter::Int = 2100)::Float64 =
    _pf_y(d, dist_quantile(d.inner, p; tol = tol, max_iter = max_iter))

# CRPS by quadrature over quantiles or over the table in z-space.
function dist_crps(d::PushforwardDist, x::Float64)::Float64
    isfinite(x) || return abs(x)
    if d.fwd_fn !== nothing || d.zs === nothing
        return _pf_crps_from_quantiles(d, x)
    end
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

# Integrate twice the pinball loss, including observations outside the
# support. Apply the weights first if the unweighted difference overflows.
function _pf_crps_from_quantiles(d::PushforwardDist, x::Float64)::Float64
    total = 0.0
    for (q, (p, w)) in zip(_qgrid(d), _QUANTILE_RULE)
        weight = 2.0 * w * (q >= x ? (1.0 - p) : p)
        gap = abs(q - x)
        if isfinite(gap)
            total += weight * gap
        else
            weighted_q = weight * q
            if isinf(q) && d.scaled_inv_fn !== nothing
                weighted_q = d.scaled_inv_fn(dist_quantile(d.inner, p), weight)
            end
            total += abs(weighted_q - weight * x)
        end
    end
    return total
end

# --- affine maps ---

# The law of a*Y + b, still a pushforward of (a reflection of) the same inner.
# a > 0 composes the map; a < 0 reflects the inner with dist_scale(inner, -1)
# so the composed map stays increasing. Port of PushforwardDist.affine.
function dist_affine(d::PushforwardDist, a::Float64, b::Float64)
    (a == 0.0 || !isfinite(a) || !isfinite(b)) &&
        throw(ArgumentError("affine needs a finite nonzero a and a finite b"))
    r = a > 0 ? 1.0 : -1.0                 # z' = r * z
    inner = r > 0 ? d.inner : dist_scale(d.inner, -1.0)
    inv, fwd, lj = d.inv_fn, d.fwd_fn, d.log_jacobian_fn
    cmf, sinv = d.component_moments_fn, d.scaled_inv_fn
    log_a = log(abs(a))
    ys = zs = nothing
    if d.ys !== nothing
        ys = Float64[a * y + b for y in d.ys]
        zs = Float64[r * z for z in d.zs]
        if r < 0
            reverse!(ys)
            reverse!(zs)
        end
    end
    return PushforwardDist(inner, ys, zs;
        inv_fn = inv === nothing ? nothing : (z -> a * inv(r * z) + b),
        fwd_fn = fwd === nothing ? nothing : (y -> r * fwd((y - b) / a)),
        log_jacobian_fn = lj === nothing ? nothing : (z -> lj(r * z) - log_a),
        component_moments_fn = cmf === nothing ? nothing :
            ((m, s) -> begin
                mu, sd = cmf(r * m, s)
                (a * mu + b, abs(a) * sd)
            end),
        scaled_inv_fn = sinv === nothing ? nothing : ((z, w) -> a * sinv(r * z, w) + w * b))
end

dist_shift(d::PushforwardDist, delta::Float64) = dist_affine(d, 1.0, delta)

function dist_scale(d::PushforwardDist, factor::Float64)
    factor == 0 && throw(ArgumentError("scale factor must be nonzero"))
    return dist_affine(d, factor, 0.0)
end

# --- Dist-compatibility shim ---

# Moment-matched mixture components in y-space (lazy, cached). For a spliced
# or already transformed inner, one Gaussian fitted to its mapped quantiles.
# This compatibility approximation can lose a support bound when a caller
# uses it to build a plain Dist.
function _pf_components(d::PushforwardDist)::Vector{NTuple{3,Float64}}
    if d._comps === nothing
        inner_comps = d.inner isa PushforwardDist ? nothing : _components(d.inner)
        if inner_comps === nothing
            q = _qgrid(d)
            mu = fsum(w * y for (y, (_, w)) in zip(q, _QUANTILE_RULE))
            sd = !isfinite(mu) ? Inf :
                 hypot((sqrt(w) * (y - mu) for (y, (_, w)) in zip(q, _QUANTILE_RULE))...)
            d._comps = NTuple{3,Float64}[(1.0, mu, sd)]
        else
            comps = NTuple{3,Float64}[]
            for (w, m, s) in inner_comps
                w > 0.0 || continue
                mu, sd = d.component_moments_fn !== nothing ? d.component_moments_fn(m, s) :
                         _pf_gaussian_moments(d, m, s)
                push!(comps, (w, mu, sd))
            end
            d._comps = comps
        end
    end
    return d._comps
end

_components(d::PushforwardDist) = _pf_components(d)

# Centered quadrature (skaters#225). hypot also keeps std representable when
# the corresponding variance is too large or small for a float.
function _pf_gaussian_moments(d::PushforwardDist, m::Float64, s::Float64)
    node_ys = [_pf_y(d, m + s * SQRT2 * x) for (x, _) in _GH7]
    mu = fsum((w / _SQRTPI) * y for (y, (_, w)) in zip(node_ys, _GH7))
    isfinite(mu) || return mu, Inf
    sd = hypot((sqrt(w / _SQRTPI) * (y - mu) for (y, (_, w)) in zip(node_ys, _GH7))...)
    return mu, max(sd, 1e-12)
end

# Mapped quantiles used by numerical moments and CRPS, cached.
function _qgrid(d::PushforwardDist)::Vector{Float64}
    if d._grid === nothing
        d._grid = Float64[dist_quantile(d, p) for (p, _) in _QUANTILE_RULE]
    end
    return d._grid
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

Base.length(d::PushforwardDist) = length(_pf_components(d))
