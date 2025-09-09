import argparse
import os
import pandas as pd
import pickle


inflation_df = pd.read_csv("Inflation.csv", index_col=0, header = [0,1])
#CPI_df = pd.read_csv("CPI.csv", index_col=0, header = [0,1])


inflation_df.columns = inflation_df.columns.droplevel(1)
cols = inflation_df.columns.values  # Get column names as a NumPy array

# Rename only specific indexed columns
cols[-12] = "Global"

inflation_df.columns = cols


inflation_df.index = pd.to_datetime(inflation_df.index.astype(str), format='%Y%m')

inflation_df = inflation_df.iloc.asfreq("MS")

country_names = inflation_df.columns[:-12]

def select_covariates(inflation_df, country, p = 4):
    # Step 1: Select the target variable (inflation) for the given country
    target = inflation_df[country]

    # Step 2: Create a copy of inflation_df for manipulation (we won't modify the original)
    selected_covariates = pd.DataFrame()

    # Step 3: Generate autoregressive lags (p lags) for the selected country without modifying the original dataframe
    for i in range(1, p + 1):
        selected_covariates[f'{country}_lag_{i}'] = inflation_df[country].shift(i)

    # Step 4: Select inflation data for all other countries (excluding the selected country)
    other_countries_inflation = inflation_df[country_names].drop(columns=[country]).shift(1)  # Drop the selected country

    # Step 5: Select the regional inflation factor for the selected country
    # Assuming that you have a column with regional factors, e.g., 'region' column in the dataset
    global_factor = inflation_df["Global"].shift(1)

    # Step 6: Add monthly dummy variables for seasonality
    monthly_dummies = inflation_df[inflation_df.columns[-11:]]

    # Step 7: Combine all the covariates into the selected_covariates DataFrame
    selected_covariates = pd.concat([selected_covariates,
                                     other_countries_inflation,
                                     global_factor,
                                     monthly_dummies], axis=1)

    return selected_covariates

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from math import sqrt, log2
import tqdm
import shap

# Set parameters
train_length = 360  # Rolling window size
start_date = inflation_df.index[train_length-1]
print(f"start_date: {start_date}")
max_horizon = 12

# Helper function for feature grid
def get_predictor_grid(p):
    return list(set([p, int(2/3 * p), int(1/3 * p), int(sqrt(p)), int(log2(p))]))

p_values = [4, 12, 24]  # Values of lags to try

def run_rf_direct(inflation_df, country_names, select_covariates, train_length, start_date, max_horizon, get_predictor_grid, p_values, start_horizon, end_horizon):
    for i, country_name in enumerate(country_names):
        print(f"{i}/{len(country_names)}: {country_name}", flush=True)

        target_series = inflation_df[country_name]
        start_idx = target_series.index.get_loc(start_date)

        for h in range(start_horizon, end_horizon + 1):
            print(h)
            historical_forecasts = []
            best_params_list = []
            shap_values = []
            tune_time = ((len(target_series) - h) - start_idx) // 2 + start_idx
            for t in tqdm.tqdm(range(start_idx, len(target_series) - h)):
                train_end_idx = t
                train_start_idx = max(0, train_end_idx - train_length)

                train_data = inflation_df.iloc[train_start_idx:train_end_idx + 1]
                target_shifted = train_data[country_name].shift(-h)

            # Forecast covariates for time t+h
                full_forecast_data = inflation_df.iloc[train_start_idx:train_end_idx + h + 1]

                if t == tune_time or t == start_idx:
                    best_score = float('inf')
                    for p in p_values:
                        covariates = select_covariates(train_data, country_name, p=p).iloc[p:-h]
                        target = target_shifted.iloc[p:-h]
                        X_tmp = select_covariates(full_forecast_data, country_name, p=p).iloc[[-1]]

                        if len(target) < 10:
                            continue  # skip too-short windows

                        param_grid = {
                        'max_features': get_predictor_grid(len(covariates.columns))
                    }

                        rf_model = RandomForestRegressor(
                        random_state=42, n_estimators=500, min_samples_leaf=5, n_jobs=-1
                    )
                        tscv = TimeSeriesSplit(n_splits=5)
                        grid_search = GridSearchCV(rf_model, param_grid, cv=tscv,
                                               scoring='neg_mean_squared_error', verbose=0)
                        grid_search.fit(covariates, target)
                        score = -grid_search.best_score_

                        if score < best_score:
                            best_score = score
                            best_model = grid_search.best_estimator_
                            best_p = p
                            X_forecast = X_tmp
                        best_params_list.append({'best_p': best_p,
                                             'max_features': best_model.max_features})
                else:
                    covariates = select_covariates(train_data, country_name, p=best_p).iloc[best_p:-h]
                    target = target_shifted.iloc[best_p:-h]
                    X_forecast = select_covariates(full_forecast_data, country_name, p=best_p).iloc[[-1]]
                    best_model.fit(covariates, target)
                print(f"predicting with best_p: {best_p}, max_features: {best_model.max_features}")
                forecast = best_model.predict(X_forecast)
            # Initialize explainer
                explainer = shap.TreeExplainer(best_model)
                shap_values.append(explainer(X_forecast))
            # Calculate SHAP values on training covariates
                forecast_index = target_series.index[t + h]
                historical_forecasts.append((forecast_index, forecast[0]))
            out = {"forecast": historical_forecasts,
               "best_params": best_params_list,
               "shap_explanation": shap_values}
            if os.path.exists('RF_forecasts') is False:
                os.makedirs('RF_forecasts')
            with open(f'RF_forecasts/RF_forecast_h{h}_{country_name}.pkl', 'wb') as f:
                pickle.dump(out, f)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Random Forest Direct Forecasting")
    parser.add_argument("--start_horizon", type=int, default=1, help="Start of the horizon range")
    parser.add_argument("--end_horizon", type=int, default=12, help="End of the horizon range")
    args = parser.parse_args()
    start_horizon = args.start_horizon
    end_horizon = args.end_horizon

    run_rf_direct(inflation_df, country_names, select_covariates, train_length, start_date, max_horizon, get_predictor_grid, p_values,
                  start_horizon, end_horizon)

