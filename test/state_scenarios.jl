# Scenario registry shared by test/state_resume.jl and
# parity/write_julia_checkpoints.jl. Mirrors parity/gen_state_vectors.py.
using Skaters

_c(t, k = 1) = conjugate(leaf(k), t, k)

function state_transforms()
    return Tuple{String,Any}[
        ("difference", difference()),
        ("fractional_difference", fractional_difference(0.4, 30)),
        ("standardize", standardize()),
        ("ema_transform", ema_transform(0.1)),
        ("ou_transform", ou_transform(0.1)),
        ("theta", theta(0.1)),
        ("drift", drift(0.05, 0.01)),
        ("holt_linear", holt_linear(0.1, 0.05)),
        ("garch", garch()),
        ("seasonal_difference", seasonal_difference(7)),
        ("seasonal_anchor", seasonal_anchor(7)),
        ("seasonal_scale", seasonal_scale(7)),
        ("seasonal_scale_center", seasonal_scale(7; center = true)),
        ("power_transform", power_transform(0.5)),
        ("power_transform_exact", power_transform(0.5; exact = true)),
        ("yeo_johnson", yeo_johnson(0.5)),
        ("yeo_johnson_exact", yeo_johnson(0.5; exact = true)),
        ("ar", ar(2)),
        ("grouped_ar", grouped_ar(8)),
        ("gaussianize", gaussianize()),
        ("gaussianize_smooth", gaussianize(smooth = true)),
    ]
end

# name => (factory, series name). A factory, so each use gets a fresh skater.
function state_scenarios()
    s = Tuple{String,Function,String}[]
    for (name, _) in state_transforms()
        sname = startswith(name, "power") ? "positive" : "series"
        push!(s, ("conjugate_" * name,
                  () -> _c(Dict(state_transforms())[name]), sname))
    end
    push!(s, ("conjugate_log_transform", () -> conjugate(leaf(1), log_transform(), 1), "positive"))
    push!(s, ("laplace", () -> laplace(k = 1), "series"))
    push!(s, ("laplace_k3", () -> laplace(k = 3), "series"))
    push!(s, ("laplace_likelihood", () -> laplace(k = 1, objective = "likelihood"), "series"))
    push!(s, ("laplace_gaussian_tails", () -> laplace(k = 1, tails = "gaussian"), "short"))
    push!(s, ("laplace_repeat", () -> laplace(k = 1), "repeat"))
    push!(s, ("laplace_gaps", () -> laplace(k = 1), "gaps"))
    push!(s, ("laplace_k3_gaps", () -> laplace(k = 3), "gaps"))
    push!(s, ("granger", () -> granger(), "series"))
    push!(s, ("flesaker", () -> flesaker(k = 1, lower = -100.0), "series"))
    push!(s, ("dantzig", () -> dantzig(k = 1, expand_interval = 50), "short"))
    push!(s, ("search_k2", () -> search(k = 2, expand_interval = 40, max_pool = 12), "short"))
    push!(s, ("multiscale_k4", () -> multiscale(kk -> _c(ema_transform(0.1), kk), 4), "short"))
    push!(s, ("parade_spliced", () -> parade(gpdtails(_c(ema_transform(0.1)), 1; nexc = 50, warmup = 100), 1), "series"))
    push!(s, ("parade_laplace_k3", () -> parade(laplace(k = 3), 3), "series"))
    push!(s, ("sticky_ema_repeat", () -> sticky(_c(ema_transform(0.1)); k = 1), "repeat"))
    push!(s, ("homogenize", () -> homogenize(_c(ema_transform(0.1)); k = 1), "short"))
    push!(s, ("homogenize_k2_weights", () -> homogenize(_c(ema_transform(0.1), 2); k = 2, couple = "weights"), "short"))
    push!(s, ("homogenize_k2_shared", () -> homogenize(_c(ema_transform(0.1), 2); k = 2, couple = "shared"), "short"))
    for model in ("m0", "m1", "m2")
        push!(s, ("residual_transform_$model", () -> residual_transform(_c(ema_transform(0.1)); model = model), "short"))
    end
    push!(s, ("pw_ensemble_k2", () -> precision_weighted_ensemble(Any[ema(0.05; k = 2), ema(0.2; k = 2)]; k = 2), "short"))
    push!(s, ("bayes_ensemble_k2", () -> bayesian_ensemble(Any[ema(0.05; k = 2), _c(difference(), 2)]; k = 2), "short"))
    push!(s, ("terminal_adaptive", () -> terminal_leaf_ensemble(Any[ema(0.05; k = 1), _c(difference())];
        k = 1, depths = [0, 1], adaptive_temperature = true), "short"))
    push!(s, ("terminal_k3", () -> terminal_leaf_ensemble(Any[ema(0.05; k = 3), _c(difference(), 3)];
        k = 3, depths = [0, 1], forget = 0.99), "short"))
    push!(s, ("scale_mixture_leaf", () -> scale_mixture_leaf(1), "series"))
    push!(s, ("crps_leaf", () -> crps_leaf(1), "series"))
    push!(s, ("garch_leaf", () -> garch_leaf(1), "series"))
    return s
end
