# Portable math helpers.
#
# Two things matter for parity with the Python reference:
#   1. fsum reproduces CPython 3.12+ built-in sum() on floats (Neumaier
#      compensation). The Python source uses sum() for Dist normalisation,
#      moments, log-sum-exp and several dot products; the port must reproduce
#      the exact accumulation or pruning tie-breaks and moments diverge.
#   2. erf/erfc come straight from Julia's bundled openlibm (the same libm
#      Base uses), reached by ccall. openlibm's erf matches CPython's own
#      m_erf to ~1e-16, far inside the 1e-6 gate.

const SQRT2 = sqrt(2.0)
const SQRT2PI = sqrt(2.0 * pi)
const LOG_SQRT2PI = 0.5 * log(2.0 * pi)

# Neumaier-compensated sum, matching CPython 3.12+ sum() on floats.
function fsum(values)
    s = 0.0
    c = 0.0
    for x in values
        xf = Float64(x)
        t = s + xf
        if abs(s) >= abs(xf)
            c += (s - t) + xf
        else
            c += (xf - t) + s
        end
        s = t
    end
    return s + c
end

erf(x::Float64) = ccall((:erf, Base.Math.libm), Cdouble, (Cdouble,), x)
erfc(x::Float64) = ccall((:erfc, Base.Math.libm), Cdouble, (Cdouble,), x)

function gaussian_pdf(x::Float64, mean::Float64, std::Float64)::Float64
    if std <= 0.0
        return x == mean ? Inf : 0.0
    end
    z = (x - mean) / std
    return exp(-0.5 * z * z) / (std * SQRT2PI)
end

function gaussian_cdf(x::Float64, mean::Float64, std::Float64)::Float64
    if std <= 0.0
        return x >= mean ? 1.0 : 0.0
    end
    return 0.5 * (1.0 + erf((x - mean) / (std * SQRT2)))
end

# E|N(m, s^2)| = m(2*Phi(m/s) - 1) + 2s*phi(m/s). (Used by CRPS.)
function abs_expectation(m::Float64, s::Float64)::Float64
    if s <= 0.0
        return abs(m)
    end
    z = m / s
    return m * (2.0 * gaussian_cdf(z, 0.0, 1.0) - 1.0) + 2.0 * s * gaussian_pdf(z, 0.0, 1.0)
end
