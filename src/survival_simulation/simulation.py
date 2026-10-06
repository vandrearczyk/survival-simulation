import numpy as np
import pandas as pd
from sksurv.util import Surv


def simulate_survival(
    n_samples=1000,
    n_features=50,
    scenario="linear",
    noise=0.0,
    censoring=0.3,
    correlation=0.0,
    betas=None,
    baseline_hazard=None,
    scale=5.0,
    seed=42,
    return_type="sksurv",
):
    """Simulate synthetic survival data across linear, non-linear, and non-PH scenarios."""
    valid_scenarios = {"linear", "nonlinear", "non_ph"}
    if scenario not in valid_scenarios:
        raise ValueError(f"Unknown scenario '{scenario}'. Must be one of {valid_scenarios}")
    if not (0.0 <= censoring < 1.0):
        raise ValueError(f"censoring must be in range [0, 1), got {censoring}")
    if return_type not in {"sksurv", "pandas"}:
        raise ValueError(f"return_type must be 'sksurv' or 'pandas', got '{return_type}'")

    rng = np.random.RandomState(seed)

    # 1. Feature Generation
    if scenario == "linear":
        X = rng.uniform(-1, 1, (n_samples, n_features))
    elif correlation > 0:
        cov = correlation * np.ones((n_features, n_features)) + (1 - correlation) * np.eye(n_features)
        X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    else:
        X = rng.normal(0, 1, (n_samples, n_features))

    if betas is not None:
        betas = np.asarray(betas)
        if len(betas) != n_features:
            raise ValueError(f"Length of betas ({len(betas)}) must match n_features ({n_features}).")

    # 2. Risk & Survival Calculation
    if scenario == "linear":
        risk = X @ betas if betas is not None else X[:, 0] + 2 * X[:, 1]
        risk += noise * rng.randn(n_samples)
        survival_time = rng.exponential(scale, n_samples) / np.exp(risk)

    elif scenario == "non_ph":
        if betas is not None:
            beta1 = beta2 = beta3 = betas
        else:
            beta1 = np.pad([1.5, -1.5, 1, -1, 0.5], (0, max(0, n_features - 5)))[:n_features]
            beta2 = np.pad([1.2, -1.2, -0.8, 0.8, 0.3], (0, max(0, n_features - 5)))[:n_features]
            beta3 = np.pad([0.8, -0.8, -0.4, 0.4, 0.2], (0, max(0, n_features - 5)))[:n_features]

        hazards = [0.05, 0.04, 0.03] if baseline_hazard is None else [baseline_hazard] * 3
        survival_time, remaining = np.zeros(n_samples), np.ones(n_samples, dtype=bool)

        for base, b, end_time in zip(hazards, [beta1, beta2, beta3], [5, 20, np.inf]):
            risk = X @ b + noise * rng.randn(n_samples)
            rate = base * np.exp((risk - risk.mean()) / risk.std())
            sampled = -np.log(rng.uniform(size=n_samples)) / rate

            if np.isinf(end_time):
                survival_time[remaining] += sampled[remaining]
                break

            idx = remaining & (sampled < end_time)
            survival_time[idx] += sampled[idx]
            remaining &= ~idx
            survival_time[remaining] += end_time

    else:  # nonlinear
        risk = X @ betas if betas is not None else np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] * X[:, 3]
        risk += noise * rng.randn(n_samples)
        bh = 0.01 if baseline_hazard is None else baseline_hazard
        hazard = bh * np.exp((risk - risk.mean()) / risk.std())
        survival_time = -np.log(rng.uniform(size=n_samples)) / hazard

    # 3. Censoring via Bisection Search
    low, high = 1e-6, survival_time.max() * 10
    for _ in range(50):
        mid = (low + high) / 2
        if np.mean(rng.exponential(mid, n_samples) < survival_time) > censoring:
            low = mid
        else:
            high = mid

    censor_time = rng.exponential((low + high) / 2, n_samples)
    event = survival_time <= censor_time
    observed_time = np.minimum(survival_time, censor_time)

    # 4. Format Output
    if return_type == "pandas":
        y_censored = pd.DataFrame({"event": event, "time": observed_time})
        y_true = pd.DataFrame({"event": True, "time": survival_time})
    else:
        y_censored = Surv.from_arrays(event, observed_time)
        y_true = Surv.from_arrays(np.ones(n_samples, dtype=bool), survival_time)

    return X, y_censored, y_true