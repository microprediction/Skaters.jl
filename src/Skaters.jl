module Skaters

# Ordering matters only for symbol availability at include time; method
# extensions (e.g. SplicedDist) attach to generic functions defined earlier.
include("mathx.jl")
include("dist.jl")
include("runstats.jl")
include("leaf.jl")
include("transform.jl")
include("conjugate.jl")
include("ema.jl")
include("ensemble.jl")
include("bayesian.jl")
include("multiscale.jl")
include("sticky.jl")
include("terminal.jl")
include("tails.jl")
include("parade.jl")
include("api.jl")

export Dist, SplicedDist, dist_gaussian, dist_combine, dist_mean, dist_var, dist_std,
    dist_logpdf, dist_cdf, dist_crps, dist_quantile, dist_pdf, dist_prune,
    dist_shift, dist_scale, dist_affine, dist_body,
    leaf, scale_mixture_leaf, crps_leaf, garch_leaf,
    conjugate, ema,
    difference, fractional_difference, standardize, ema_transform, ou_transform,
    theta, drift, holt_linear, garch, seasonal_difference, power_transform,
    ar, grouped_ar, yeo_johnson,
    precision_weighted_ensemble, bayesian_ensemble, multiscale, sticky,
    terminal_leaf_ensemble, gpdtails, parade, laplace

end # module
