# Standard Pkg.test entry. Runs the four gates in order: parity (the
# 1e-6 vector gate), quantile contract (the shared cross-language inverse-CDF
# cases), robustness (adversarial streams), contract (determinism and
# checkpoint-resume), then the SplicedDist affine and nonaffine regression
# tests and the transform and wave-2 parity vectors. Each suite also runs
# standalone:
#   julia --project=. test/parity.jl
#   julia --project=. test/quantile_contract.jl
#   julia --project=. test/robustness.jl
#   julia --project=. test/contract.jl
#   julia --project=. test/spliced_affine.jl
#   julia --project=. test/spliced_nonaffine.jl
#   julia --project=. test/transform_parity.jl
#   julia --project=. test/wave2_parity.jl
# Every suite raises (or exits nonzero) on any violation, which fails
# Pkg.test. Suites are wrapped in modules so their helpers do not collide.

module ParitySuite
include("parity.jl")
end

module QuantileContractSuite
include("quantile_contract.jl")
end

module RobustnessSuite
include("robustness.jl")
end

module ContractSuite
include("contract.jl")
end

module SplicedAffineSuite
include("spliced_affine.jl")
end

module SplicedNonaffineSuite
include("spliced_nonaffine.jl")
end

module TransformParitySuite
include("transform_parity.jl")
end

module Wave2ParitySuite
include("wave2_parity.jl")
end

println("ALL SUITES OK (parity, quantile contract, robustness, contract, spliced affine, spliced nonaffine, transform parity, wave2 parity)")
