"""Debug script for CAF proliferation ODE from Biffi2019 S6K target.

Tests the forward model with Hill-coefficient logistic term.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# --- Data from digitized Figure S6K ---
data = {
    'qPSC': {
        'ic': 32258,
        'means': np.array([36290, 34677, 44355, 46774], dtype=float),
        'sds':   np.array([9677,  11290, 10484, 12903], dtype=float),
    },
    'iCAF': {
        'ic': 44355,
        'means': np.array([61290, 78226, 117742, 178226], dtype=float),
        'sds':   np.array([20968, 29032, 35484,  55645], dtype=float),
    },
    'myCAF': {
        'ic': 46774,
        'means': np.array([116129, 240323, 602632, 800000], dtype=float),
        'sds':   np.array([20968,  35484,  97368,  131579], dtype=float),
    },
}
eval_days = np.array([2, 3, 4, 5], dtype=float)

# --- ODE with Hill-coefficient logistic ---
def ode(t, y, k_q, k_i, k_m, K, n):
    q, i, m = y
    dq = k_q * q * (1 - (q / K) ** n)
    di = k_i * i * (1 - (i / K) ** n)
    dm = k_m * m * (1 - (m / K) ** n)
    return [dq, di, dm]

def solve_model(k_q, k_i, k_m, K, n):
    y0 = [data['qPSC']['ic'], data['iCAF']['ic'], data['myCAF']['ic']]
    sol = solve_ivp(
        lambda t, y: ode(t, y, k_q, k_i, k_m, K, n),
        t_span=[1, 5], y0=y0, t_eval=eval_days,
        method='RK45', rtol=1e-8, atol=1e-8,
    )
    return sol

# --- Test: sweep n to see effect on myCAF fit ---
print("=== Effect of Hill coefficient n on myCAF fit ===\n")
print(f"{'n':>5} {'myCAF_d2':>10} {'myCAF_d3':>10} {'myCAF_d4':>10} {'myCAF_d5':>10} {'SSE':>12}")
print(f"{'data':>5} {116129:>10.0f} {240323:>10.0f} {602632:>10.0f} {800000:>10.0f}")

for n in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0]:
    # Use k_m = 0.71 (from exponential fit), K = 1.2M
    sol = solve_model(0.093, 0.348, 0.71, 1_200_000, n)
    preds = sol.y[2]  # myCAF
    sse = np.sum((preds - data['myCAF']['means'])**2 / data['myCAF']['sds']**2)
    print(f"{n:>5.1f} {preds[0]:>10.0f} {preds[1]:>10.0f} {preds[2]:>10.0f} {preds[3]:>10.0f} {sse:>12.1f}")

# --- Simple MLE fit with n as free parameter ---
print("\n=== MLE fit (all params free) ===\n")

def neg_log_lik(params_vec):
    k_q, k_i, k_m, log_K, log_n = params_vec
    K = np.exp(log_K)
    n = np.exp(log_n)
    if k_q <= 0 or k_i <= 0 or k_m <= 0 or K <= 0 or n <= 0:
        return 1e10
    try:
        sol = solve_model(k_q, k_i, k_m, K, n)
        if not sol.success:
            return 1e10
        nll = 0.0
        for j in range(4):
            for state, key in [(0, 'qPSC'), (1, 'iCAF'), (2, 'myCAF')]:
                pred = sol.y[state, j]
                obs = data[key]['means'][j]
                sd = data[key]['sds'][j]
                nll += 0.5 * ((pred - obs) / sd) ** 2
        return nll
    except:
        return 1e10

x0 = [0.093, 0.348, 0.71, np.log(1_200_000), np.log(3.0)]
res = minimize(neg_log_lik, x0, method='Nelder-Mead',
               options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})

k_q_fit, k_i_fit, k_m_fit, log_K_fit, log_n_fit = res.x
K_fit = np.exp(log_K_fit)
n_fit = np.exp(log_n_fit)

print(f"k_qPSC_prolif = {k_q_fit:.4f} /day")
print(f"k_iCAF_prolif = {k_i_fit:.4f} /day")
print(f"k_myCAF_prolif = {k_m_fit:.4f} /day")
print(f"K_well = {K_fit:.0f}")
print(f"n_logistic = {n_fit:.2f}")
print(f"NLL = {res.fun:.2f}")

sol = solve_model(k_q_fit, k_i_fit, k_m_fit, K_fit, n_fit)
print(f"\nFit vs data:")
for key, idx in [('qPSC', 0), ('iCAF', 1), ('myCAF', 2)]:
    print(f"\n  {key}:")
    for j, day in enumerate([2, 3, 4, 5]):
        pred = sol.y[idx, j]
        obs = data[key]['means'][j]
        sd = data[key]['sds'][j]
        print(f"    Day {day}: pred={pred:>10.0f}  data={obs:>10.0f} +/- {sd:>6.0f}  z={abs(pred-obs)/sd:>5.2f}")

# --- Stability test with new prior for n ---
print(f"\n=== Stability test with n_logistic prior (mu=1.1, sigma=0.5) ===\n")

rng = np.random.default_rng(42)
n_draws = 500
failures = 0
slow = 0

for draw in range(n_draws):
    k_q = rng.lognormal(-2.3, 1.0)
    k_i = rng.lognormal(-2.3, 1.0)
    k_m = rng.lognormal(-1.9, 1.0)
    K = rng.lognormal(14.0, 0.5)
    n = rng.lognormal(1.1, 0.5)  # median ~ 3.0

    y0 = [data['qPSC']['ic'], data['iCAF']['ic'], data['myCAF']['ic']]
    t0 = time.time()
    try:
        sol = solve_ivp(
            lambda t, y, kq=k_q, ki=k_i, km=k_m, Kw=K, nn=n: ode(t, y, kq, ki, km, Kw, nn),
            t_span=[1, 5], y0=y0, t_eval=eval_days,
            method='RK45', rtol=1e-6, atol=1e-6,
        )
        elapsed = time.time() - t0
        if not sol.success:
            failures += 1
        elif elapsed > 0.1:
            slow += 1
            if slow <= 3:
                print(f"  SLOW: {elapsed:.3f}s, k_m={k_m:.3f}, K={K:.0f}, n={n:.2f}")
    except Exception as e:
        failures += 1
        if failures <= 3:
            print(f"  ERROR: {e}, k_m={k_m:.3f}, K={K:.0f}, n={n:.2f}")

print(f"Results: {n_draws-failures-slow} OK, {slow} slow, {failures} fail out of {n_draws}")

# --- Check prior on n: what does lognormal(1.1, 0.5) look like? ---
samples = rng.lognormal(1.1, 0.5, 10000)
print(f"\nn_logistic prior stats: median={np.median(samples):.2f}, "
      f"mean={np.mean(samples):.2f}, "
      f"P5={np.percentile(samples, 5):.2f}, "
      f"P95={np.percentile(samples, 95):.2f}")
