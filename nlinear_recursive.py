import os
import pickle
import darts
import tqdm
import logging
import numpy as np
import pandas as pd
import torch
import shap
import torch.nn as nn

from math import sqrt, log2
from sklearn.model_selection import ParameterGrid
from darts import TimeSeries
from darts.models import NLinearModel
from darts.metrics import rmse

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
inflation_series = TimeSeries.from_dataframe(inflation_df)

train_length = 360
start_date = inflation_df.index[train_length - 1]

# -----------------------------
# Model wrapper for SHAP
# -----------------------------
class NLinearWrapper(nn.Module):
    def __init__(self, model, input_chunk_length, n_targets, n_covariates, horizon=12):
        super().__init__()
        self.model = model
        self.L = input_chunk_length
        self.n_targets = n_targets
        self.n_covariates = n_covariates
        self.horizon = horizon

    def forward(self, x):
        B = x.shape[0]
        if x.dim() == 2:
            x = x.view(B, self.L, self.n_targets + self.n_covariates)

        cov_last = x[:, -1, self.n_targets:].clone()
        current = x.clone()
        preds = []

        for _ in range(self.horizon):
            y_pred = self.model((current, None, None))  # can be weird shapes

            # --- normalize ---
            # remove all singleton dims except batch
            y_pred = y_pred.view(B, -1)

            if y_pred.shape[1] != self.n_targets:
                raise ValueError(f"Unexpected y_pred shape after reshape: {y_pred.shape}, expected (B, {self.n_targets})")

            preds.append(y_pred.unsqueeze(1))  # (B, 1, n_targets)

            # roll and update
            current = torch.roll(current, shifts=-1, dims=1)
            current[:, -1, :self.n_targets] = y_pred
            current[:, -1, self.n_targets:] = cov_last

        preds = torch.cat(preds, dim=1)  # (B, horizon, n_targets)
        return preds.reshape(B, -1)


def shap_values_nlinear_recursive(
    best_model,
    pred_loader,
    train_loader,
    n_targets: int,
    n_covariates: int,
    horizon: int = 12,
    n_background: int = 100,
):
    """
    SHAP with recursive NLinearWrapper.
    - Builds background and explain tensors by CONCAT(target, past_covariates) along features (last dim).
    - Uses the recursive wrapper so SHAP "sees" the full h-step recursive mapping.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.model = best_model.model.float().to(device).eval()

    def concat_tp(batch):
        tgt, pc = batch[:2]
        return torch.cat([tgt.float(), pc.float()], dim=-1)  # (B, L, n_targets+n_cov)

    # --- background ---
    bg_list = [concat_tp(b) for b in train_loader]
    background = torch.cat(bg_list, dim=0)
    if background.size(0) > n_background:
        idx = torch.randperm(background.size(0))[:n_background]
        background = background[idx]
    background = background.to(device)

    # --- batch to explain (same shape as background) ---
    batch = next(iter(pred_loader))
    x_to_explain = concat_tp(batch).to(device)

    # wrapper with recursion
    wrapped = NLinearWrapper(
        model=best_model.model,
        input_chunk_length=best_model.input_chunk_length,
        n_targets=n_targets,
        n_covariates=n_covariates,
        horizon=horizon,
    ).to(device)

    explainer = shap.GradientExplainer(wrapped, background)
    shap_expl = explainer(x_to_explain)   # explains (B, L, n_targets+n_cov) -> (B, horizon*n_targets)

    return shap_expl


# -----------------------------
# Recursive validation scoring
# ----------------------------
def recursive_forecast(model, train_series, horizon, past_covariates):
    preds = []
    current_series = train_series
    current_covs = past_covariates.drop_after(current_series.end_time() + past_covariates.freq)

    for _ in range(horizon):
        # one-step forecast
        pred = model.predict(
            1, 
            series=current_series, 
            past_covariates=current_covs, 
            show_warnings=False
        )
        preds.append(pred)

        # extend the series with prediction
        current_series = current_series.append(pred)
        try:
            current_covs = past_covariates.drop_after(current_series.end_time() + past_covariates.freq)
        except Exception as e:
            print(f"Error occurred while dropping after: {e}")
            continue

        pred_df = pred.pd_dataframe()
        global_val = float(pred_df.iloc[0].mean())

        # overwrite "Global" at the last timestamp
        covs_df = current_covs.pd_dataframe()
        if "Global" not in covs_df.columns:
            raise ValueError("past_covariates must include a 'Global' component/column.")
        last_ts = covs_df.index[-1]
        covs_df.loc[last_ts, "Global"] = global_val

        # rebuild the TimeSeries (preserve freq)
        current_covs = TimeSeries.from_dataframe(
            covs_df,
            freq=current_covs.freq
        )

    # return a single stacked series instead of list
    return darts.concatenate(preds)

def recursive_validation_score(model_class, params, train_series, val_series, past_covariates, horizon):
    """
    Fit NLinear (or similar) with output_chunk_length=1 and evaluate recursive forecasts.
    Uses a rolling-origin setup: train on train_series, validate on val_series.
    """
    # enforce recursive setup
    params = {**params, "output_chunk_length": 1}

    # fit model
    model = model_class(**params)
    model.fit(series=train_series, past_covariates=past_covariates)

    # predict horizon steps ahead in one call (internally recursive)
    forecast = model.predict(horizon, series=train_series, past_covariates=past_covariates, show_warnings=False)

    # align with val_series
    val_series = val_series.slice_intersect(forecast)

    return rmse(val_series, forecast)


def recursive_gridsearch(model_class, param_grid, train_series, val_series, past_covariates, horizon):
    """
    Grid search over params with recursive forecasting.
    Returns (best_params, best_score).
    """
    from itertools import product

    best_score = float("inf")
    best_params = None

    # expand param grid manually
    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        score = recursive_validation_score(model_class, params, train_series, val_series, past_covariates, horizon)
        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score

# -----------------------------
# Main recursive run
# -----------------------------
if __name__ == "__main__":
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    param_grid = {
        "input_chunk_length": [4, 8, 12, 24],
        "output_chunk_length": [1],  # recursive → always 1-step
        "pl_trainer_kwargs": [{"enable_progress_bar": False, "enable_model_summary": False}],
        "batch_size": [8],
        "n_epochs": [20],
    }

    for h in [12]:
        print(f"Recursive NLinear forecasting, horizon={h}")
        historical_forecasts = []
        all_shap_explanations = []

        start_idx = inflation_df.index.get_loc(start_date)

        for t in tqdm.tqdm(range(start_idx, len(inflation_df) - h)):
            train_end_idx = t
            train_start_idx = max(0, train_end_idx - train_length)

            train_data = inflation_series[train_start_idx:train_end_idx + 1]
            val_data = inflation_series[train_end_idx + 1: train_end_idx + h + 1]
            past_covariates = inflation_series.drop_columns(list(country_names))

            # tuning once every 12 months
            if inflation_series[t].time_index.month == 12 or t == start_idx:
                best_params, best_score = recursive_gridsearch(
                    NLinearModel, param_grid,
                    train_data, val_data, past_covariates, h
                )
                print(f"Best params at {inflation_series[t].time_index}: {best_params} (score={best_score:.4f})")
            else:
                best_params = best_params

            best_model = NLinearModel(**best_params)
            best_model.fit(series=train_data[list(country_names)], past_covariates=past_covariates)

            forecast = recursive_forecast(
                best_model,
                train_data[list(country_names)],
                h,
                past_covariates
            )
            historical_forecasts.append(forecast)

            shap_explanation = shap_values_nlinear_recursive(
                best_model,
                best_model.pred_loader_out,
                best_model.train_loader_out,
                n_targets=91,
                n_covariates=12,
                n_background=50
            )
            print(shap_explanation.shape)
            all_shap_explanations.append(shap_explanation)

        out_dict = {"forecast": historical_forecasts, "shap_explanation": all_shap_explanations}
        if not os.path.exists("nlinear_forecasts_recursive"):
            os.makedirs("nlinear_forecasts_recursive")
        with open(f"nlinear_forecasts_recursive/nlinear_forecast_h{h}.pkl", "wb") as f:
            pickle.dump(out_dict, f)
