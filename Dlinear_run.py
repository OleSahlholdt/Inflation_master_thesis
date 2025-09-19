import pandas as pd
from darts.models import AutoARIMA
from darts import TimeSeries
import pickle
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


inflation_df = pd.read_csv("Inflation.csv", index_col=0, header = [0,1])
CPI_df = pd.read_csv("CPI.csv", index_col=0, header = [0,1])


inflation_df.columns = inflation_df.columns.droplevel(1)
cols = inflation_df.columns.values  # Get column names as a NumPy array

# Rename only specific indexed columns
cols[-12] = "Global"

inflation_df.columns = cols


inflation_df.index = pd.to_datetime(inflation_df.index.astype(str), format='%Y%m')

inflation_df = inflation_df.asfreq("MS")

inflation_df_train = inflation_df[inflation_df.index < pd.Timestamp('2000-03-01')]
inflation_df_test = inflation_df[inflation_df.index >= pd.Timestamp('2000-03-01')]

inflation_series = TimeSeries.from_dataframe(inflation_df)

country_names = inflation_df.columns[:-12]

import numpy as np
import pandas as pd
from darts.models import DLinearModel
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from math import sqrt, log2
import tqdm
import logging
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
from darts.metrics import rmse
import shap
import torch
import torch.nn as nn

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

def shap_values_dlinear(best_model, pred_loader, country_names, covariate_names, 
                        train_loader, n_background):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.model = best_model.model.float().to(device)
    best_model.model.eval()

    # --- prepare background ---
    bg_inputs = []
    for batch in train_loader:
        target, past_covariates = batch[:2]
        target = target.float()
        past_covariates = past_covariates.float()
        arr = torch.cat([target, past_covariates], dim=-1)
        bg_inputs.append(arr)
    bg_inputs = torch.cat(bg_inputs, dim=0)
    if bg_inputs.size(0) > n_background:
        idx = torch.randperm(bg_inputs.size(0))[:n_background]
        background = bg_inputs[idx].to(device)
    else:
        background = bg_inputs.to(device)

    # --- point to explain ---
    batch = next(iter(pred_loader))
    target, past_covariates = batch[:2]
    x_to_explain = torch.cat([target, past_covariates], dim=-1).float().to(device)

    n_features = bg_inputs.shape[-1]
    input_chunk_length = best_model.input_chunk_length

    # wrap model
    wrapped_model = DLinearWrapper(best_model.model, input_chunk_length, n_features).to(device)
    
    # --- SHAP explainer ---
    explainer = shap.GradientExplainer(wrapped_model, background)
    shap_values = explainer(x_to_explain)


    return shap_values

dlinear = DLinearModel(

    input_chunk_length=4,

    output_chunk_length=1,

    kernel_size=12,

    batch_size = 8,

    n_epochs=20,
)

param_grid = {'kernel_size': [5, 9, 13, 25],
              'input_chunk_length': [4, 8, 12, 24],
              "output_chunk_length": [1],
              "pl_trainer_kwargs": [{"enable_progress_bar": False, "enable_model_summary": False}]
              }

# Assuming `inflation_series` is a DataFrame with countries as columns
forecasts_country = {}
train_length = 360  # Rolling window size
start_date = inflation_df.index[train_length-1]
print(f"start_date: {start_date}")
for h in [12]:
    print(f"h: {h}")
    dlinear = DLinearModel(

    input_chunk_length=4,

    output_chunk_length=h,

    kernel_size=12,

    batch_size = 8,

    n_epochs=20)
    param_grid = {'kernel_size': [5, 9, 13, 25],
              'input_chunk_length': [4, 8, 12, 24],
              "output_chunk_length": [h],
              "pl_trainer_kwargs": [{"enable_progress_bar": False, "enable_model_summary": False}]
              }

    # Extract the target time series
    historical_forecasts = []
    all_shap_explanations = []
    # Define the starting index
    start_idx = inflation_df.index.get_loc(start_date)
    for t in tqdm.tqdm(range(start_idx, len(inflation_df) - 1 - h)):
        train_end_idx = t
        train_start_idx = max(0, train_end_idx - train_length)
        train_data = inflation_series[train_start_idx:train_end_idx + 1]
        # Extract training data
        train_series = train_data[4:]
        if inflation_series[t].time_index.month == 12 or t == start_idx:
            # Fit dlinear model with grid search
            # Use TimeSeriesSplit for time series cross-validation
            grid_search = dlinear.gridsearch(param_grid, 
                                            train_series[list(country_names)], 
                                            inflation_series.drop_columns(list(country_names)), 
                                            forecast_horizon = len(train_series)//5, stride = len(train_series)//5, 
                                            verbose=False, metric=rmse, show_warnings = False)
            # Get the best model from grid search
            best_model = grid_search[0]
            best_params = best_model.model_params
            print(f"Best params at time {t}: {best_params}")
        else:
            best_model = DLinearModel(**{**best_params, "batch_size": 8, "n_epochs": 20, "output_chunk_length": h})
        best_model.fit(series=train_series[list(country_names)], past_covariates=inflation_series.drop_columns(list(country_names)))
        # Forecast h steps ahead
        forecast = best_model.predict(h)
        covariate_names = inflation_series.drop_columns(list(country_names)).columns

        shap_explanation = shap_values_dlinear(best_model, best_model.pred_loader_out, 
                                      country_names=country_names, 
                                      covariate_names=covariate_names,
                                      train_loader=best_model.train_loader_out, n_background=100)
        print(shap_explanation.shape)
        historical_forecasts.append(forecast)
        all_shap_explanations.append(shap_explanation)
    forecasts = pd.Series(historical_forecasts)
    out_dict = {"forecast": forecasts, "shap_explanation": all_shap_explanations}

    with open(f'dlinear_forecasts/dlinear_forecast_h{h}.pkl', 'wb') as f:
        pickle.dump(out_dict, f)