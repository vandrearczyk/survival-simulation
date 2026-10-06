[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vandrearczyk/survival-simulation/blob/main/demo.ipynb)

# survival-simulation

[![PyPI version](https://img.shields.io/pypi/v/survival-simulation.svg)](https://pypi.org/project/survival-simulation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, flexible synthetic survival data generator supporting linear, non-linear, and non-proportional hazards (non-PH) scenarios.

Designed for benchmarking survival analysis algorithms, proportional hazard models, and deep learning architectures (e.g., DeepSurv, CoxPH, Random Survival Forests).

---

## Features

- **Standardized Scenarios:** `linear`, `nonlinear`, and `non_ph`.
- **Customizable Signal Parameters:** Set custom feature coefficients (`betas`), noise levels, baseline hazards, and scale parameters.
- **Controlled Censoring:** Fast bisection search algorithm to match exact target censoring proportions.
- **Flexible Outputs:** Get targets as `scikit-survival` structured arrays or standard `pandas` DataFrames.
- **Multi-Interface Support:** Use via Python API, CLI command, or Docker container.

---

## Installation

### Via `pip`

```bash
pip install survival-simulation
```

### Via Docker

```bash
docker pull yourusername/survival-simulation:latest
```

---

## Python API Usage

### 1. Quick benchmark scenario

By default, the function returns targets in a format compatible with `scikit-survival`.

```python
from survival_simulation import simulate_survival

X, y_censored, y_true = simulate_survival(
    n_samples=1000,
    n_features=50,
    scenario="linear",
    censoring=0.3,
    seed=42
)
```

### 2. Return results as Pandas DataFrames

```python
from survival_simulation import simulate_survival

X, y_censored, y_true = simulate_survival(
    scenario="non_ph",
    return_type="pandas"
)

print(y_censored.head())

#   event       time
# 0   True   4.123456
# 1  False  12.876543
```

### 3. Custom feature coefficients

You can provide custom feature coefficients (`betas`) to control the signal strength of individual features.

```python
import numpy as np

from survival_simulation import simulate_survival

custom_betas = np.zeros(50)
custom_betas[:3] = [2.0, -1.5, 0.8]

X, y_censored, y_true = simulate_survival(
    scenario="linear",
    betas=custom_betas,
    censoring=0.25
)
```

---

## CLI Usage

Generate synthetic survival datasets directly from your command line:

```bash
simulate-survival \
  --scenario non_ph \
  --n-samples 2000 \
  --n-features 30 \
  --censoring 0.25 \
  --out ./data/synthetic_survival.csv
```

---

## Docker Usage

Generate datasets without installing a Python environment locally:

```bash
docker run --rm -v $(pwd)/data:/data yourusername/survival-simulation \
  --scenario nonlinear \
  --n-samples 5000 \
  --censoring 0.3 \
  --out /data/nonlinear_dataset.csv
```

---

## Supported Scenarios

| Scenario | Description | Key Mechanism |
| --- | --- | --- |
| `linear` | Standard linear risk | Uniform features with exponential survival times driven by a linear risk function. |
| `nonlinear` | Non-linear feature interactions | Proportional hazards with a non-linear risk function. |
| `non_ph` | Non-proportional hazards | Piecewise hazard rates across three distinct time intervals. |

---

## License

MIT License. See [LICENSE](LICENSE) for details.