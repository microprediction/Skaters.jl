# Online residual-transform wrapper: learn to reshape a skater's own errors.
# Port of skaters/residual_transform.py.
#
# Given the base forecast F_t for y_{t+1} and a residual model H_t (a
# predictive for z_{t+1} = Phi^-1(F_t(y_{t+1})), learned from z_1..z_t), the
# corrected forecast is
#
#     F~_t(y) = H_t(Phi^-1(F_t(y)))
#     Q~_t(p) = F_t.quantile(Phi(H_t.quantile(p)))
#
# CorrectedDist implements this exactly for cdf/pdf/logpdf/quantile, and by a
# fixed 32-point quantile-grid quadrature for mean/std/crps. Nested models:
#
#     m0  identity          H_t = N(0, 1)
#     m1  dynamic scale     H_t = N(0, exp(ell_t)),  ell_{t+1} = a ell_t + c (z_t^2 - 1)
#     m2  + leverage        ell_{t+1} = a ell_t + b z_t + c (z_t^2 - 1)
#
# a, b, c are learned online by one RMSProp-normalized gradient step per tick
# through a real-time-recurrent-learning-lite sensitivity filter.

const _RT_STD_NORMAL = Dist(NTuple{3,Float64}[(1.0, 0.0, 1.0)])
const _RT_EPS = 1e-12

# Learner stability bounds
const _RT_A_MIN, _RT_A_MAX = 0.0, 0.995
const _RT_BC_MAX = 3.0
const _RT_ELL_MIN, _RT_ELL_MAX = -20.0, 20.0
const _RT_SENS_MAX = 50.0
const _RT_GRAD_MAX = 3.0

# Quantile-grid quadrature for mean/var/crps (midpoint rule on p in (0,1)).
const _RT_QGRID_N = 32
const _RT_QGRID_P = Float64[(i + 0.5) / _RT_QGRID_N for i in 0:(_RT_QGRID_N-1)]
const _RT_QGRID_W = 1.0 / _RT_QGRID_N

# Standard normal quantile by bisection on the unit Gaussian Dist, as in the
# reference (not the Acklam quantile the tails use).
_rt_phi_inv(p::Float64) = dist_quantile(_RT_STD_NORMAL, p)
_rt_clamp01(u::Float64) = min(max(u, _RT_EPS), 1.0 - _RT_EPS)
_rt_clip(x::Float64, lo::Float64, hi::Float64) = x < lo ? lo : (x > hi ? hi : x)

# --- transformation families: H_t as cdf, quantile, logpdf of (z or p, theta) ---

struct Family
    cdf::Function
    quantile::Function
    logpdf::Function
end

const IDENTITY_FAMILY = Family(
    (z, theta) -> dist_cdf(_RT_STD_NORMAL, z),
    (p, theta) -> _rt_phi_inv(p),
    (z, theta) -> -0.5 * z * z - LOG_SQRT2PI)

function _gss_cdf(z, theta)
    sigma = exp(0.5 * theta["ell"])
    return dist_cdf(_RT_STD_NORMAL, (z - theta["m"]) / sigma)
end

function _gss_quantile(p, theta)
    sigma = exp(0.5 * theta["ell"])
    return theta["m"] + sigma * _rt_phi_inv(p)
end

function _gss_logpdf(z, theta)
    sigma = exp(0.5 * theta["ell"])
    zz = (z - theta["m"]) / sigma
    return -0.5 * zz * zz - log(sigma) - LOG_SQRT2PI
end

const GAUSSIAN_SHIFT_SCALE_FAMILY = Family(_gss_cdf, _gss_quantile, _gss_logpdf)

# --- CorrectedDist: F~_t = H_t[Phi^-1(F_t(.))], exact composition ---

mutable struct CorrectedDist <: AbstractDist
    base::Any
    family::Family
    theta::Dict{String,Any}
    _qgrid::Union{Nothing,Vector{Float64}}
end

CorrectedDist(base, family::Family, theta::Dict) =
    CorrectedDist(base, family, Dict{String,Any}(theta), nothing)

_rt_z(d, y::Float64) = _rt_phi_inv(_rt_clamp01(dist_cdf(d.base, y)))

dist_cdf(d::CorrectedDist, y::Float64)::Float64 = d.family.cdf(_rt_z(d, y), d.theta)

function dist_logpdf(d::CorrectedDist, y::Float64)::Float64
    z = _rt_z(d, y)
    # chain rule: log f~(y) = log f(y) + log h(z) - log phi(z)
    return dist_logpdf(d.base, y) + d.family.logpdf(z, d.theta) + 0.5 * z * z + LOG_SQRT2PI
end

function dist_pdf(d::CorrectedDist, y::Float64)::Float64
    lp = dist_logpdf(d, y)
    return lp > -700.0 ? exp(lp) : 0.0
end

function dist_quantile(d::CorrectedDist, p::Float64; tol::Float64 = 1e-9, max_iter::Int = 2100)::Float64
    @assert 0 < p < 1
    hp = d.family.quantile(p, d.theta)
    u = _rt_clamp01(dist_cdf(_RT_STD_NORMAL, hp))
    return dist_quantile(d.base, u; tol = tol, max_iter = max_iter)
end

# Quantile-grid moments and CRPS, shared with HomogenizedDist.
function _qgrid_moments_mean(q::Vector{Float64})
    return fsum(q) * _RT_QGRID_W
end

function _qgrid_var(q::Vector{Float64})
    mu = _qgrid_moments_mean(q)
    return fsum((qi - mu)^2 for qi in q) * _RT_QGRID_W
end

# CRPS via the quantile-form identity (Gneiting and Raftery 2007).
function _qgrid_crps(q::Vector{Float64}, x::Float64)
    total = 0.0
    for (p, qi) in zip(_RT_QGRID_P, q)
        ind = x <= qi ? 1.0 : 0.0
        total += (qi - x) * (ind - p)
    end
    return 2.0 * total * _RT_QGRID_W
end

function _quantile_grid(d::CorrectedDist)
    if d._qgrid === nothing
        d._qgrid = Float64[dist_quantile(d, p) for p in _RT_QGRID_P]
    end
    return d._qgrid
end

dist_mean(d::CorrectedDist)::Float64 = _qgrid_moments_mean(_quantile_grid(d))
dist_var(d::CorrectedDist)::Float64 = _qgrid_var(_quantile_grid(d))
function dist_std(d::CorrectedDist)::Float64
    v = dist_var(d)
    return v > 0 ? sqrt(v) : 0.0
end
dist_crps(d::CorrectedDist, x::Float64)::Float64 = _qgrid_crps(_quantile_grid(d), x)

# --- RTRL-lite online learners for m1 (scale) and m2 (scale + leverage) ---

const _RT_RMS_RHO = 0.01
const _RT_RMS_EPS = 1e-6

function _init_scale_learner(learn_b::Bool)
    state = Dict{String,Any}("ell" => 0.0, "a" => 0.5, "c" => 0.05, "ea" => 0.0, "ec" => 0.0,
                             "g2a" => 0.0, "g2c" => 0.0)
    if learn_b
        state["b"] = 0.0
        state["eb"] = 0.0
        state["g2b"] = 0.0
    end
    return state
end

# One RMSProp-normalized step.
function _rms_step(lr::Float64, grad::Float64, g2_prev::Float64)
    g2_new = (1.0 - _RT_RMS_RHO) * g2_prev + _RT_RMS_RHO * grad * grad
    step = lr * grad / sqrt(g2_new + _RT_RMS_EPS)
    return step, g2_new
end

function _update_scale_learner(state::Dict, z::Float64, lr::Float64, learn_b::Bool)
    ell_prev = state["ell"]
    a, c = state["a"], state["c"]
    b = get(state, "b", 0.0)

    # d(-log h)/d(ell), h = N(0, exp(ell_prev)):  0.5 - 0.5 z^2 exp(-ell_prev)
    grad_ell = _rt_clip(0.5 - 0.5 * z * z * exp(-ell_prev), -_RT_GRAD_MAX, _RT_GRAD_MAX)

    step_a, g2a_new = _rms_step(lr, grad_ell * state["ea"], state["g2a"])
    step_c, g2c_new = _rms_step(lr, grad_ell * state["ec"], state["g2c"])
    a_new = _rt_clip(a - step_a, _RT_A_MIN, _RT_A_MAX)
    c_new = _rt_clip(c - step_c, -_RT_BC_MAX, _RT_BC_MAX)
    if learn_b
        step_b, g2b_new = _rms_step(lr, grad_ell * state["eb"], state["g2b"])
        b_new = _rt_clip(b - step_b, -_RT_BC_MAX, _RT_BC_MAX)
    else
        b_new = 0.0
    end

    ea_new = _rt_clip(ell_prev + a_new * state["ea"], -_RT_SENS_MAX, _RT_SENS_MAX)
    ec_new = _rt_clip((z * z - 1.0) + a_new * state["ec"], -_RT_SENS_MAX, _RT_SENS_MAX)

    ell_new = _rt_clip(a_new * ell_prev + b_new * z + c_new * (z * z - 1.0), _RT_ELL_MIN, _RT_ELL_MAX)

    new_state = Dict{String,Any}("ell" => ell_new, "a" => a_new, "c" => c_new, "ea" => ea_new,
                                 "ec" => ec_new, "g2a" => g2a_new, "g2c" => g2c_new)
    if learn_b
        new_state["b"] = b_new
        new_state["eb"] = _rt_clip(z + a_new * state["eb"], -_RT_SENS_MAX, _RT_SENS_MAX)
        new_state["g2b"] = g2b_new
    end
    return new_state
end

function _theta_of(model::String, learner)
    model == "m0" && return Dict{String,Any}("m" => 0.0, "ell" => 0.0)
    theta = Dict{String,Any}("m" => 0.0, "ell" => learner["ell"], "a" => learner["a"],
                             "c" => learner["c"])
    if model == "m2"
        theta["b"] = learner["b"]
    end
    return theta
end

const _RT_MODELS = ("m0", "m1", "m2")

# Wrap a k=1 skater with an online residual-transform correction. The state
# also carries state["z"] (the resolved residual, nothing on the first tick),
# state["raw"] (the base skater's own forecast) and state["theta"].
function residual_transform(base; model::String = "m2", lr::Float64 = 0.01, k::Int = 1)
    @assert k == 1 "residual_transform currently supports k=1 only"
    @assert model in _RT_MODELS "model must be one of $(_RT_MODELS), got $model"
    learn_b = model == "m2"
    family = model == "m0" ? IDENTITY_FAMILY : GAUSSIAN_SHIFT_SCALE_FAMILY

    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            learner0 = model == "m0" ? nothing : _init_scale_learner(learn_b)
            state = Dict{String,Any}("base" => nothing, "raw" => nothing, "z" => nothing,
                                     "learner" => learner0,
                                     "theta" => _theta_of(model, learner0 === nothing ?
                                                          Dict{String,Any}() : learner0))
        end
        z = nothing
        if state["raw"] !== nothing
            z = _rt_phi_inv(_rt_clamp01(dist_cdf(state["raw"], y)))
            if model != "m0"
                state["learner"] = _update_scale_learner(state["learner"], z, lr, learn_b)
            end
        end
        state["z"] = z

        dists, state["base"] = base(y, state["base"])
        f_t = dists[1]
        state["raw"] = f_t

        theta = _theta_of(model, model != "m0" ? state["learner"] : Dict{String,Any}())
        state["theta"] = theta
        return Any[CorrectedDist(f_t, family, theta)], state
    end
    return _skater
end

# --- diagnostics: online EWMA moments and the leverage statistic L_k ---

# Online EWMA diagnostics on a z-stream: moments and the leverage statistic
# L_k = E[z_t (z_{t+k}^2 - 1)] for k = 1..K. A diagnostic only; feed it
# state["z"] from residual_transform (or from parade). update!(d, z) and
# summary(d) mirror the reference's update and summary methods.
mutable struct ResidualDiagnostics
    K::Int
    _alpha::Float64
    n::Int
    m1::Float64
    m2::Float64
    m3::Float64
    m4::Float64
    _buf::Vector{Float64}
    _L::Vector{Float64}
    _cross_sq::Vector{Float64}
end

ResidualDiagnostics(; K::Int = 5, halflife::Float64 = 50.0) =
    ResidualDiagnostics(K, 1.0 - 0.5^(1.0 / halflife), 0, 0.0, 0.0, 0.0, 0.0,
                        Float64[], zeros(K), zeros(K))

function update!(d::ResidualDiagnostics, z::Float64)
    d.n += 1
    a = d._alpha > 1.0 / d.n ? d._alpha : 1.0 / d.n
    z2 = z * z
    d.m1 = (1 - a) * d.m1 + a * z
    d.m2 = (1 - a) * d.m2 + a * z2
    d.m3 = (1 - a) * d.m3 + a * z * z2
    d.m4 = (1 - a) * d.m4 + a * z2 * z2
    for lag in 1:d.K
        if length(d._buf) >= lag
            z_lag = d._buf[end-lag+1]
            d._L[lag] = (1 - a) * d._L[lag] + a * z_lag * (z2 - 1.0)
            d._cross_sq[lag] = (1 - a) * d._cross_sq[lag] + a * z_lag * z_lag * z2
        end
    end
    push!(d._buf, z)
    if length(d._buf) > d.K       # deque(maxlen=K)
        popfirst!(d._buf)
    end
    return d
end

Base.summary(d::ResidualDiagnostics) = Dict{String,Any}(
    "n" => d.n, "E[z]" => d.m1, "E[z2]" => d.m2, "E[z3]" => d.m3, "E[z4]" => d.m4,
    "L" => copy(d._L), "cross_sq" => copy(d._cross_sq))
