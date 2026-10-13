import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn



default_m  = 0.0001
default_p  = 0.1
default_v  = 0.001
default_S0 = 10
default_h0 = 0
default_K = 200.3
default_n = 11
default_b = 0.93

def simulate_asset_price_path(m=default_m, s0=default_S0, h0=default_h0, n=default_n, p=default_p, v=default_v):
    s = np.empty(n+1)
    s[0] = np.log(s0)

    h = h0
    for t in range(n):
        s[t+1] = s[t] + m + np.exp(h) * randn()
        h = p * h + v * randn()

    return np.exp(s)


fig, axes = plt.subplots(2, 1)

titles = 'log paths', 'paths'
transforms = np.log, lambda x: x
for ax, transform, title in zip(axes, transforms, titles):
    for i in range(50):
        path = simulate_asset_price_path()
        ax.plot(transform(path))
    ax.set_title(title)

fig.tight_layout()
plt.show()


def compute_call_price(β=default_b,
                       μ=default_m,
                       S0=default_S0,
                       h0=default_h0,
                       K=default_K,
                       n=default_n,
                       ρ=default_p,
                       ν=default_v,
                       M=10_000):
    current_sum = 0.0
    # For each sample path
    for m in range(M):
        s = np.log(S0)
        h = h0
        # Simulate forward in time
        for t in range(n):
            s = s + μ + np.exp(h) * randn()
            h = ρ * h + ν * randn()
        # And add the value max{S_n - K, 0} to current_sum
        current_sum += np.maximum(np.exp(s) - K, 0)

    return β**n * current_sum / M

print(compute_call_price())



















