# Sticky / lattice projection. Port of skaters/sticky.py.
# The value table is state["counts"], a list of [value, weight] pairs in
# first-seen order (the reference's canonical layout): decay, prune in place
# keeping order, then credit y. An existing entry keeps its slot; a new or
# just-pruned value goes to the end. The order is semantic for the
# tie-broken atom selection.

function sticky(base; k::Int = 1, propensity_alpha::Float64 = 0.05,
                spike_frac::Float64 = 0.005, thresh_mult::Float64 = 1.8,
                max_atoms::Int = 6, prune_eps::Float64 = 1e-6)
    function _skater(y::Float64, state::Union{Nothing,Dict})
        if state === nothing
            state = Dict{String,Any}("base" => nothing, "counts" => Any[])
        elseif haskey(state, "vals")
            # An older Julia checkpoint kept parallel vals/wts vectors.
            state["counts"] = Any[Any[v, w] for (v, w) in zip(state["vals"], state["wts"])]
            delete!(state, "vals")
            delete!(state, "wts")
        end
        dists, state["base"] = base(y, state["base"])

        keep = 1.0 - propensity_alpha
        counts = Any[]
        for pair in state["counts"]
            w = pair[2] * keep
            w < prune_eps && continue
            push!(counts, Any[pair[1], w])
        end
        idx = findfirst(p -> p[1] == y, counts)
        if idx === nothing
            push!(counts, Any[y, propensity_alpha])
        else
            counts[idx][2] += propensity_alpha
        end
        state["counts"] = counts
        vals = Float64[Float64(p[1]) for p in counts]
        wts = Float64[Float64(p[2]) for p in counts]

        # Lattice atoms: revisited values above the floor, top by weight
        # (stable sort so ties break by insertion order, as in Python).
        thr = thresh_mult * propensity_alpha
        sel = [i for i in eachindex(wts) if wts[i] > thr]
        atoms = [(vals[i], wts[i]) for i in sel]
        if !isempty(atoms)
            order = sortperm([-a[2] for a in atoms]; alg = MergeSort)
            atoms = atoms[order]
            if length(atoms) > max_atoms
                atoms = atoms[1:max_atoms]
            end
        end

        out = Any[]
        for d in dists
            if isempty(atoms)
                push!(out, d)
                continue
            end
            sw = fsum(a[2] for a in atoms)
            P = min(sw, 0.999)
            pc = 1.0 - P
            atom_mean = fsum(a[2] * a[1] for a in atoms) / sw
            spike_std = max(spike_frac * dist_std(d), 1e-9)
            if pc <= 1e-9
                comps = NTuple{3,Float64}[(a[2] / sw, a[1], spike_std) for a in atoms]
                push!(out, Dist(comps))
                continue
            end
            mu = dist_mean(d)
            delta = P * (mu - atom_mean) / pc
            comps = NTuple{3,Float64}[(P * (a[2] / sw), a[1], spike_std) for a in atoms]
            for (w, m, s) in _need_components(d)
                push!(comps, (pc * w, m + delta, s))
            end
            push!(out, Dist(comps))
        end
        return out, state
    end
    return _skater
end
