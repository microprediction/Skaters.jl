# Standard Pkg.test entry. Runs the three gates in order: parity (the
# 1e-6 vector gate), robustness (adversarial streams), contract
# (determinism and checkpoint-resume). Each suite also runs standalone:
#   julia --project=. test/parity.jl
#   julia --project=. test/robustness.jl
#   julia --project=. test/contract.jl
# Every suite raises (or exits nonzero) on any violation, which fails
# Pkg.test. Suites are wrapped in modules so their helpers do not collide.

module ParitySuite
include("parity.jl")
end

module RobustnessSuite
include("robustness.jl")
end

module ContractSuite
include("contract.jl")
end

println("ALL SUITES OK (parity, robustness, contract)")
