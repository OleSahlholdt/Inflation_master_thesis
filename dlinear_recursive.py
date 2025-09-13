# dlinear_recursive.py

import os
import pickle
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
from darts.models import DLinearModel
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
class DLinearWrapper(nn.Module):
    def __init__(self, model, input_chunk_length, n_features):
        super().__init__()
        self.model = model
        self.input_chunk_length = input_chunk_length
        self.n_features = n_features

    def forward(self, x):
        # x: (batch, input_chunk_length * n_features)
        B = x.shape[0]
        x = x.view(-1, self.input_chunk_length, self.n_features)
        y_pred = self.model((x, None, None))
        return y_pred.view(B, -1)

def shap_values_dlinear(best_model, pred_loader, train_loader, n_background=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.model = best_model.model.float().to(device)
    best_model.model.eval()

    # background
    bg_inputs = []
    for batch in train_loader:
        target, past_covariates = batch[:2]
        arr = torch.cat([target.float(), past_covariates.float()], dim=-1)
        bg_inputs.append(arr)
    bg_inputs = torch.cat(bg_inputs, dim=0)

    if bg_inputs.size(0) > n_background:
        idx = torch.randperm(bg_inputs.size(0))[:n_background]
        background = bg_inputs[idx].to(device)
    else:
        background = bg_inputs.to(device)

    # one batch to explain
    batch = next(iter(pred_loader))
    target, past_covariates = batch[:2]
    x_to_explain = torch.cat([target.float(), past_covariates.float()], dim=-1).to(device)

    n_features = bg_inputs.shape[-1]
    input_chunk_length = best_model.input_chunk_length
    wrapped_model = DLinearWrapper(best_model.model, input_chunk_length, n_features).to(device)

    explainer = shap.GradientExplainer(wrapped_model, background)
    shap_values = explainer(x_to_explain)
    return shap_values

# -----------------------------
# Recursive validation scoring
# -----------------------------
def recursive_validation_score(model_class, params, train_series, val_series, past_covariates, horizon, country_names):
    """Fit model with given params and score it recursively on val_series."""
    model = model_class(**params)
    model.fit(train_series[list(country_names)], past_covariates=past_covariates)

    input_series = train_series
    preds = []
    for _ in range(len(val_series)):
        forecast = model.predict(1, series=input_series, past_covariates=past_covariates)
        preds.append(forecast)
        input_series = input_series.append(forecast)

    preds = preds[0].stack(preds[1:]) if len(preds) > 1 else preds[0]
    return rmse(val_series[list(country_names)], preds)

def recursive_gridsearch(model_class, param_grid, train_series, val_series, past_covariates, horizon, country_names):
    """Grid search with recursive evaluation."""
    best_score = float("inf")
    best_params = None
    for params in ParameterGrid(param_grid):
        score = recursive_validation_score(model_class, params, train_series, val_series, past_covariates, horizon, country_names)
        if score < best_score:
            best_score, best_params = score, params
    return best_params, best_score

# -----------------------------
# Main recursive run
# -----------------------------
if __name__ == "__main__":
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    param_grid = {
        "kernel_size": [5, 9, 13, 25],
        "input_chunk_length": [4, 8, 12, 24],
        "output_chunk_length": [1],  # recursive → always 1-step
        "pl_trainer_kwargs": [{"enable_progress_bar": False, "enable_model_summary": False}],
        "batch_size": [8],
        "n_epochs": [20],
    }

    for h in [1, 6, 12]:
        print(f"Recursive DLinear forecasting, horizon={h}")
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
                    DLinearModel, param_grid,
                    train_data, val_data, past_covariates, h, country_names
                )
                print(f"Best params at {inflation_series[t].time_index}: {best_params} (score={best_score:.4f})")
            else:
                best_params = best_params

            best_model = DLinearModel(**best_params)
            best_model.fit(series=train_data[list(country_names)], past_covariates=past_covariates)

            forecast = best_model.predict(h, series=train_data, past_covariates=past_covariates)
            historical_forecasts.append(forecast)

            shap_explanation = shap_values_dlinear(best_model, best_model.pred_loader_out,
                                                   train_loader=best_model.train_loader_out,
                                                   n_background=100)
            all_shap_explanations.append(shap_explanation)

        out_dict = {"forecast": historical_forecasts, "shap_explanation": all_shap_explanations}
        if not os.path.exists("dlinear_forecasts_recursive"):
            os.makedirs("dlinear_forecasts_recursive")
        with open(f"dlinear_forecasts_recursive/dlinear_forecast_h{h}.pkl", "wb") as f:
            pickle.dump(out_dict, f)
