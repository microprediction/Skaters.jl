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
include("pushforward.jl")
include("parade.jl")
include("api.jl")
include("cov.jl")
include("periodicity.jl")
include("spec.jl")
include("search.jl")
include("granger.jl")
include("residual_transform.jl")
include("homogenize.jl")
include("state.jl")

export granger, granger_params, MixtureDist
export Dist, SplicedDist, PushforwardDist, dist_gaussian, dist_combine, dist_mean, dist_var, dist_std,
    dist_logpdf, dist_cdf, dist_crps, dist_quantile, dist_pdf, dist_prune,
    dist_shift, dist_scale, dist_affine, dist_body,
    leaf, scale_mixture_leaf, crps_leaf, garch_leaf,
    conjugate, ema,
    difference, fractional_difference, standardize, ema_transform, ou_transform,
    theta, drift, holt_linear, garch, seasonal_difference, seasonal_anchor, seasonal_scale,
    power_transform, ar, grouped_ar, yeo_johnson, log_transform, gaussianize,
    precision_weighted_ensemble, bayesian_ensemble, multiscale, sticky,
    terminal_leaf_ensemble, gpdtails, parade, laplace, flesaker,
    running_cov, ema_cov, ledoit_wolf_cov,
    period_detector, top_periods,
    spec_build, build, to_json, from_json, spec_name, leaf_spec, ema_spec, ensemble_spec, conjugate_spec,
    diff_spec, frac_spec, std_spec, ema_t_spec,
    adaptive_search, search, dantzig, Skater,
    residual_transform, ResidualDiagnostics, CorrectedDist,
    homogenize, homogenize_candidates, HomogenizedDist,
    state_to_json, state_from_json, assert_plain_state

end # module
