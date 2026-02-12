# Insurance Claim Prediction (French Motor TPL)

Implementation and comparison of **claim frequency** models on the French motor third‑party liability dataset (*freMTPL2freq*), following a Poisson frequency modeling setup with exposure weighting.

Models included:
- Poisson **GLM** (feature engineering + one‑hot encoding)
- Poisson **feedforward neural network** (exponential output)
- Tree‑based methods: **Decision Tree**, **Random Forest**, **Gradient Boosted Trees** (all trained with exposure‑weighted Poisson deviance)

---

## Contents

- `notebooks/insurance_claim_prediction.ipynb`: end‑to‑end workflow (preprocessing → training → evaluation → model comparison)
- `requirements.txt`: minimal dependencies

---

## Problem Setup

We model claim counts `ClaimNb` with policy duration `Exposure` under a Poisson assumption.

- Target frequency: `y_i = ClaimNb_i / Exposure_i`

- Poisson GLM mean: `lambda_i = exp(<theta, x_i> + theta_0)`

- Training objective: Exposure-weighted Poisson deviance (weights = `Exposure`)


### Metrics reported
- MAE  
- MSE  
- Weighted Poisson deviance (loss)

---

## Feature Preprocessing (GLM baseline)

- `VehPower` → `log(VehPower)`
- `VehAge` → categorical bins: `[0,6)`, `[6,13)`, `[13,∞)`
- `DrivAge` → `log(DrivAge)`
- `BonusMalus` → `log(BonusMalus)`
- `Density` → `log(Density)`

Additionally:
- Standardize all continuous/discrete features
- One‑hot encode categorical features

---

## Running

```bash
pip install -r requirements.txt
```

Open the notebook and run all cells.

---

## Notes

- The dataset file `freMTPL2freq.csv` is not included in the repository (large / licensing). Place it locally and update the notebook path accordingly.
- If you use Apple Silicon / CPU‑only PyTorch, ensure your local PyTorch installation matches your machine.

