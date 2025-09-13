# rf_recursive.py

import argparse
import os
import pickle
import tqdm
import shap
import numpy as np
import pandas as pd
from math import sqrt, log2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# -----------------------------
# Data prep
# -----------------------------
inflation_df = pd.read_csv("Inflation.csv", index_col=0, header=[0, 1])
inflation_df.columns = inflation_df.columns.droplevel(1)
cols = inflation_df.columns.values
cols[-12] = "Global"
inflation_df.columns = cols
inflation_df.index = pd.to_datetime(inflation_df.index.astype(str), format="%Y%m")
inflation_df = inflation_df.asfreq("MS")

country_names = inflation_df.columns[:-12]


def select_covariates(inflation_df, country, p=4):
    """Construct covariates: AR lags, other countries, global factor, monthly dummies."""
    selected_covariates = pd.DataFrame()

    # AR lags
    for i in range(1, p + 1):
        selected_covariates[f"{country}_lag_{i}"] = inflation_df[country].shift(i)

    # Other countries (lagged)
    other_countries_inflation = inflation_df[country_names].drop(columns=[country]).shift(1)

    # Global factor (lagged)
    global_factor = inflation_df["Global"].shift(1)

    # Monthly dummies
    monthly_dummies = inflation_df[inflation_df.columns[-11:]]

    return pd.concat(
        [selected_covariates, other_countries_inflation, global_factor, monthly_dummies],
        axis=1,
    )


# -----------------------------
# Helpers
# -----------------------------
def get_predictor_grid(p):
    return list(set([p, int(2 / 3 * p), int(1 / 3 * p), int(sqrt(p)), int(log2(p))]))


from sklearn.metrics import mean_squared_error
import numpy as np

def recursive_cv_score(rf_model, X_train, y_train, country_name, p, max_horizon=12, n_splits=3):
    """
    Evaluate RF in recursive mode inside CV.
    Returns mean RMSE across folds.
    Optimized with NumPy instead of pandas row slicing.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []

    # Convert once to NumPy arrays
    X_full = X_train.to_numpy(copy=True)
    y_full = y_train.to_numpy(copy=True)

    # Map AR lag column indices for quick access
    lag_cols = [X_train.columns.get_loc(f"{country_name}_lag_{j}") for j in range(1, p + 1)]

    for train_idx, val_idx in tscv.split(X_full):
        X_tr, y_tr = X_full[train_idx], y_full[train_idx]
        X_val, y_val = X_full[val_idx], y_full[val_idx]

        rf_model.fit(X_tr, y_tr)

        preds = np.zeros_like(y_val)
        history = list(y_tr[-p:])  # last p observed values

        # Recursive loop
        for t in range(len(y_val)):
            x_t = X_val[t].copy()

            # Overwrite AR lags with recursive history
            for j, col_idx in enumerate(lag_cols, start=1):
                x_t[col_idx] = history[-j]

            pred = rf_model.predict(x_t.reshape(1, -1))[0]
            preds[t] = pred
            history.append(pred)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        errors.append(rmse)

    return np.mean(errors)


# -----------------------------
# RF recursive forecasting
# -----------------------------
def run_rf_recursive(inflation_df, country_names, select_covariates,
                     train_length, start_date, max_horizon,
                     get_predictor_grid, p_values):
    """Recursive Random Forest forecasting with SHAP explanations."""
    for i, country_name in enumerate(country_names):
        print(f"{i}/{len(country_names)}: {country_name}", flush=True)

        target_series = inflation_df[country_name]
        start_idx = target_series.index.get_loc(start_date)

        historical_forecasts = []  # (timestamp, forecast_value, horizon)
        best_params_list = []
        shap_values_list = []
        tune_time = ((len(target_series) - max_horizon) - start_idx) // 2 + start_idx

        for t in tqdm.tqdm(range(start_idx, len(target_series) - max_horizon)):
            train_end_idx = t
            train_start_idx = max(0, train_end_idx - train_length)
            train_data = inflation_df.iloc[train_start_idx:train_end_idx + 1]

            # 1-step target
            target_shifted = train_data[country_name].shift(-1)

            # --- tuning (once, at first step) ---
            if t == start_idx or t == tune_time:
                best_score = float("inf")
                best_model, best_p = None, None

                for p in p_values:
                    covariates = select_covariates(train_data, country_name, p=p).iloc[p:-1]
                    target = target_shifted.iloc[p:-1]
                    if len(target) < 20:
                        continue

                    for mf in get_predictor_grid(len(covariates.columns)):
                        candidate = RandomForestRegressor(
                            random_state=42, n_estimators=500,
                            min_samples_leaf=5, n_jobs=-1,
                            max_features=mf
                        )
                        score = recursive_cv_score(candidate, covariates, target, country_name, p,
                                                   max_horizon=max_horizon, n_splits=5)

                        if score < best_score:
                            best_score = score
                            best_model = candidate
                            best_p = p

                print(f"Best params for {country_name}: p={best_p}, max_features={best_model.max_features}")

            else:
                covariates = select_covariates(train_data, country_name, p=best_p).iloc[best_p:-1]
                target = target_shifted.iloc[best_p:-1]
                best_model.fit(covariates, target)

            best_params_list.append({"best_p": best_p, "max_features": best_model.max_features})

            # --- recursive forecast loop ---
            history = target_series.iloc[train_end_idx - best_p + 1:train_end_idx + 1].tolist()
            forecast_index = target_series.index[train_end_idx + 1: train_end_idx + max_horizon + 1]

            for h in range(1, max_horizon + 1):
                X_forecast = select_covariates(
                    inflation_df.iloc[train_start_idx:train_end_idx + h],
                    country_name, p=best_p
                ).iloc[[-1]]

                # overwrite AR lags with recursive history
                for j in range(1, best_p + 1):
                    X_forecast[f"{country_name}_lag_{j}"] = history[-j]

                forecast_val = best_model.predict(X_forecast)[0]
                history.append(forecast_val)

                # record
                historical_forecasts.append((forecast_index[h - 1], forecast_val, h))

                # SHAP explanation (can be heavy!)
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer(X_forecast)
                shap_values_list.append(shap_values)

        # save
        out = {
            "forecast": historical_forecasts,
            "best_params": best_params_list,
            "shap_explanation": shap_values_list,
        }
        if not os.path.exists("RF_forecasts_recursive"):
            os.makedirs("RF_forecasts_recursive")
        with open(f"RF_forecasts_recursive/RF_forecast_{country_name}.pkl", "wb") as f:
            pickle.dump(out, f)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Random Forest Recursive Forecasting")
    parser.add_argument("--max_horizon", type=int, default=12, help="Forecast horizon")
    args = parser.parse_args()

    train_length = 360
    start_date = inflation_df.index[train_length - 1]
    p_values = [4, 12, 24]

    run_rf_recursive(
        inflation_df, country_names, select_covariates,
        train_length, start_date, args.max_horizon,
        get_predictor_grid, p_values
    )
