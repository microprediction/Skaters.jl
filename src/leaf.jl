# Residual distribution estimators: the leaf of every prediction tree.
# Port of skaters/leaf.py. Leaves return (dists::Vector, state::Dict).

# --- plain centered Gaussian leaf ---

function leaf(k::Int = 1)
    function _leaf(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("var" => running_var_init())
        end
        state["var"] = running_var_update(state["var"], y)
        _, var = running_var_get(state["var"])
        if isfinite(var) && var > 0
            std = sqrt(var)
        else
            std = max(abs(y), 1e-8)
        end
        d = dist_gaussian(0.0, std)
        return Any[d for _ in 1:k], state
    end
    return _leaf
end

const _SCALE_BASIS = (0.7, 1.0, 1.6, 3.0, 6.0)

# argmin of |C[i]-1| (first minimum on ties), 1-based.
_one_index(C) = argmin([abs(c - 1.0) for c in C])

function scale_mixture_leaf(k::Int = 1; gamma::Float64 = 0.02,
                            scale_alpha::Float64 = 0.01, scales = _SCALE_BASIS)
    C = Tuple(scales)
    K = length(C)
    one_idx = _one_index(C)

    function _leaf(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            w = fill(1e-6, K)
            w[one_idx] = 1.0
            state = Dict{String,Any}("v" => 0.0, "w" => w, "n" => 0)
        end
        state["n"] += 1
        a = scale_alpha > 1.0 / state["n"] ? scale_alpha : 1.0 / state["n"]
        state["v"] = (1 - a) * state["v"] + a * y * y
        var = state["v"]
        sigma = (isfinite(var) && var > 0) ? sqrt(var) : max(abs(y), 1e-8)
        z = y / sigma
        w = state["w"]
        dens = [w[i] * exp(-0.5 * z * z / (C[i] * C[i])) / C[i] for i in 1:K]
        total = fsum(dens)
        if total > 0
            g = gamma > 1.0 / state["n"] ? gamma : 1.0 / state["n"]
            state["w"] = [(1 - g) * w[i] + g * dens[i] / total for i in 1:K]
        end
        d = Dist([(state["w"][i], 0.0, C[i] * sigma) for i in 1:K])
        return Any[d for _ in 1:k], state
    end
    return _leaf
end

# --- CRPS leaf: scale mixture, weights fit by online CRPS gradient ---

const _S2 = sqrt(2.0)
const _INV = 1.0 / sqrt(2.0 * pi)
const _A0 = 2.0 * _INV

_Phi(x::Float64) = 0.5 * (1.0 + erf(x / _S2))
_phi_small(x::Float64) = exp(-0.5 * x * x) * _INV

function _abs_normal(m::Float64, s::Float64)::Float64
    if s <= 0
        return abs(m)
    end
    z = m / s
    return m * (2.0 * _Phi(z) - 1.0) + 2.0 * s * _phi_small(z)
end

const FINE = Tuple(round(0.4 * 1.28^i, digits = 4) for i in 0:14)

function crps_leaf(k::Int = 1; eta::Float64 = 1.0, scale_alpha::Float64 = 0.01,
                   scales = FINE)
    C = Tuple(scales)
    K = length(C)
    B = [[sqrt(C[a] * C[a] + C[b] * C[b]) * _A0 for b in 1:K] for a in 1:K]
    one_idx = _one_index(C)

    function _leaf(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            w = fill(1e-6, K)
            w[one_idx] = 1.0
            state = Dict{String,Any}("v" => 0.0, "w" => w, "n" => 0)
        end
        state["n"] += 1
        a = scale_alpha > 1.0 / state["n"] ? scale_alpha : 1.0 / state["n"]
        state["v"] = (1 - a) * state["v"] + a * y * y
        v = state["v"]
        sig = (isfinite(v) && v > 0) ? sqrt(v) : max(abs(y), 1e-8)
        z = y / sig
        w = state["w"]
        g = [_abs_normal(-z, C[c]) - fsum(w[j] * B[c][j] for j in 1:K) for c in 1:K]
        gm = fsum(g) / K
        e = [-eta * (g[c] - gm) for c in 1:K]
        emax = maximum(e)
        nw = [w[c] * exp(e[c] - emax) for c in 1:K]
        Z = fsum(nw)
        if !(Z > 0.0 && isfinite(Z))
            nw = copy(w)
            Z = fsum(nw)
        end
        state["w"] = [x / Z for x in nw]
        d = Dist([(state["w"][c], 0.0, C[c] * sig) for c in 1:K])
        return Any[d for _ in 1:k], state
    end
    return _leaf
end

# --- GARCH(1,1)-t leaf ---

const _GARCH_AB_GRID = [(a, b) for a in (0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20)
                        for b in (0.72, 0.78, 0.84, 0.88, 0.92, 0.95, 0.97) if a + b < 0.999]
const _GARCH_OMEGA_MULT = (0.5, 0.7, 1.0, 1.4, 2.0)

function garch_leaf(k::Int = 1; gamma::Float64 = 0.02, refit_every::Int = 40,
                    min_obs::Int = 80, window::Int = 400, scales = _SCALE_BASIS)
    C = Tuple(scales)
    K = length(C)
    one_idx = _one_index(C)

    function _leaf(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            w = fill(1e-6, K)
            w[one_idx] = 1.0
            state = Dict{String,Any}("h" => 0.0, "s2" => 0.0, "n" => 0, "omega" => 0.0,
                "alpha" => 0.05, "beta" => 0.90, "buf" => Float64[], "w" => w, "last_r2" => 0.0)
        end
        s = state
        s["n"] += 1
        a0 = 0.02 > 1.0 / s["n"] ? 0.02 : 1.0 / s["n"]
        s["s2"] = (1 - a0) * s["s2"] + a0 * y * y
        if s["s2"] <= 0
            s["s2"] = max(y * y, 1e-12)
        end
        if s["n"] == 1
            h = s["s2"]
        else
            h = s["omega"] + s["alpha"] * s["last_r2"] + s["beta"] * s["h"]
        end
        if h <= 1e-300
            h = s["s2"]
        end
        s["h"] = h
        s["last_r2"] = y * y
        buf = s["buf"]::Vector{Float64}
        push!(buf, y)
        if length(buf) > window
            popfirst!(buf)
        end

        if s["n"] >= min_obs && s["n"] % refit_every == 0 && length(buf) >= min_obs
            resid = copy(buf)
            s2 = fsum(r * r for r in resid) / length(resid)
            if s2 > 0
                best_om = 0.0; best_al = 0.0; best_be = 0.0
                best_v = Inf
                found = false
                for (al, be) in _GARCH_AB_GRID
                    base = (1.0 - al - be) * s2
                    for c in _GARCH_OMEGA_MULT
                        om = base * c > 1e-12 ? base * c : 1e-12
                        hh = om / (1.0 - al - be)
                        v = 0.0
                        for r in resid
                            hh = om + al * (r * r) + be * hh
                            if hh <= 1e-300
                                hh = 1e-300
                            end
                            v += log(hh) + (r * r) / hh
                        end
                        if v < best_v
                            best_v = v; best_om = om; best_al = al; best_be = be
                            found = true
                        end
                    end
                end
                if found
                    s["omega"] = best_om; s["alpha"] = best_al; s["beta"] = best_be
                end
            end
        end

        sigma = (isfinite(h) && h > 0) ? sqrt(h) : max(abs(y), 1e-8)
        z = y / sigma
        w = s["w"]
        dens = [w[i] * exp(-0.5 * z * z / (C[i] * C[i])) / C[i] for i in 1:K]
        total = fsum(dens)
        if total > 0
            g = gamma > 1.0 / s["n"] ? gamma : 1.0 / s["n"]
            s["w"] = [(1 - g) * w[i] + g * dens[i] / total for i in 1:K]
        end
        d = Dist([(s["w"][i], 0.0, C[i] * sigma) for i in 1:K])
        return Any[d for _ in 1:k], s
    end
    return _leaf
end
