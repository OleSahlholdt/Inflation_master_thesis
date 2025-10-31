# Inflation Master Thesis Repository

This repository contains the code, data, and experiments for a master's thesis focused on inflation forecasting using advanced machine learning models, including transformers and SHAP-based interpretability techniques.

## Repository Structure

### 1. **FEDFormer**
   - Contains a forked repo from the FEDFormer article with several changes for time series forecasting.
   - **Key Files**:
     - `Autoformer_run.py`, `FEDFormer_run.py`, `Informer_run.py`: Scripts for running experiments with different transformer models.
     - `default_args.py`: Default arguments for configuring the models.
     - `run.py`: Main script for training, testing, and predicting with FEDformer.
     - `README.md`: Documentation for the FEDformer implementation.
     - `requirements.txt`: Dependencies required to run the FEDformer experiments.
   - **Folders**:
     - `data_provider/`: Data loading and preprocessing utilities.
     - `dataset/`: Contains the dataset files, such as `Inflation_transformer.csv`.
     - `exp/`: Experiment setup and execution scripts.
     - `layers/`: Implementation of model layers.
     - `models/`: Transformer model implementations.
     - `results/`: Stores the results of the experiments.
     - `scripts/`: Shell scripts for running experiments.
     - `utils/`: Utility functions for the project.

### 2. **Data Wrangling**
   - **File**: `Data_Wrang.ipynb`
   - Jupyter notebook for preprocessing and preparing the inflation data. Includes steps for:
     - Calculating inflation rates.
     - Generating dummy variables for monthly seasonality.
     - Exporting processed data to CSV files (`Inflation.csv`, `CPI.csv`).

### 3. **SHAP Analysis**
   - **File**: `SHAP_inference.ipynb`
   - Notebook for interpreting model predictions using SHAP (SHapley Additive exPlanations). Includes:
     - Generating SHAP values for feature importance.
     - Visualizing the impact of predictors on model outputs.

### 4. **Results Analysis**
   - **File**: `get_results.ipynb`
   - Notebook for analyzing and visualizing the results of the forecasting models. Includes:
     - Performance metrics for different models.
     - Comparison of forecasting accuracy across horizons.

### 5. **NLinear Recursive**
   - **File**: `nlinear_recursive.py`
   - Implementation of the NLinear model for recursive forecasting. Includes:
     - Data preparation for recursive forecasting.
     - Model training and evaluation.
     - SHAP integration for interpretability.

### 6. **Base Models**
   - This section contains implementations of baseline models for inflation forecasting.
   - **Key Files**:
     - `dlinear.py`:
       - Implements the DLinear (Decomposition Linear) model for time series forecasting.
       - Decomposes the input series into trend and seasonal components for better forecasting accuracy.
     - `nlinear.py`:
       - Implements the NLinear (Nonlinear Linear) model for time series forecasting.
       - Focuses on capturing nonlinear relationships in the data.
     - `dlinear_recursive.py`:
       - A recursive version of the DLinear model.
       - Uses recursive forecasting to predict multiple steps ahead by feeding predictions back into the model.
     - `RF_direct.py`:
       - Implements a Random Forest (RF) model for direct multi-step forecasting.
       - Trains separate models for each forecasting horizon to improve accuracy.

### 6. **plots**
   - all plots and tables can be found in the plot folder

### 7. **Licenses and References**
   - **File**: `FEDFormer/LICENSE`
   - The FEDformer implementation is licensed under the MIT License.
   - **File**: `FEDFormer/README.md`
   - Documentation for the FEDformer model, including citations and acknowledgments.

### 8. **The Actual Thesis**
   - The thesis can be found in Master_Thesis.pdf

---

## Acknowledgments

This repository builds on the FEDformer implementation and incorporates additional modifications for inflation forecasting. Key references include:
- [FEDformer Paper](https://arxiv.org/abs/2201.12740)

For further details, refer to the individual README files in the `FEDFormer` folder.