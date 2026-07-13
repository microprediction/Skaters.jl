# EMA skater = conjugate(leaf, ema_transform). Port of skaters/ema.py.

function ema(alpha::Float64 = 0.05; k::Int = 1)
    return conjugate(leaf(k), ema_transform(alpha), k)
end
