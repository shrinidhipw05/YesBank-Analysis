# 📈 Yes Bank Stock Price Prediction

## 📝 Project Overview
Yes Bank is a prominent private sector bank in India whose stock performance was historically robust and stable. However, following a major corporate governance crisis in 2018 involving alleged financial mismanagement, the bank's valuation experienced a catastrophic collapse. This severe structural break created extreme market volatility and drastically shifted the baseline valuation of the stock.

The primary objective of this capstone project is to develop a highly accurate machine learning regression model capable of predicting the monthly closing price of Yes Bank's stock, navigating significant data challenges such as extreme multicollinearity and a non-stationary target distribution.

## 📂 Repository Contents
*   **`data_YesBank_StockPrices.csv`**: The raw historical monthly stock dataset containing Open, High, Low, and Close prices.
*   **`EDA_YesBank_StockPrices.ipynb`**: Exploratory Data Analysis notebook detailing data distributions, trends, structural breaks, and correlation heatmaps.
*   **`Sample_ML_Submission_Template.ipynb`**: The primary machine learning pipeline covering data preprocessing, feature engineering, model training, evaluation, and explainability.
*   **`YesBank_GradientBoosting_Model.pkl`**: The serialized, hyperparameter-tuned Gradient Boosting Regressor model, exported and ready for production deployment.

## ⚙️ Methodology & Approach

### 1. Data Preprocessing & Feature Engineering
*   **Target Normalization:** The target variable (`Close` price) originally exhibited severe right-skewness. A logarithmic mathematical transformation was applied to stabilize variance and normalize the distribution.
*   **Handling Multicollinearity:** The independent variables (`Open`, `High`, and `Low`) moved in near-perfect lockstep. Feeding these raw variables into a model would cause unstable coefficient weights. To resolve this, two distinct features were engineered:
    *   **`Average_Price`**: Represents the baseline midpoint trading value.
    *   **`Volatility`**: Captures the extreme monthly trading spread.

### 2. Machine Learning Modeling
Several regression algorithms were evaluated to find the optimal fit for the time-series data:
*   **Linear Regression:** Established a baseline reference but struggled to minimize prediction errors during sudden market shifts.
*   **Random Forest Regressor:** A robust ensemble method that drastically reduced error rates by averaging multiple decision trees to prevent overfitting.
*   **Gradient Boosting Regressor (Final Selection):** Emerged as the top-performing model. By building sequential decision trees designed to correct the residual errors of preceding trees, it achieved the lowest Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### 3. Model Explainability
*   **SHAP (SHapley Additive exPlanations)** was utilized to validate the model's decision-making process. The SHAP summary plot confirmed that the engineered `Average_Price` feature is the primary fundamental driver for the predictions.

## 🚀 Deployment & Usage
The optimized model is saved as a lightweight `.pkl` file. It can be directly integrated into an operational financial dashboard or live trading API. 

*Note: The upstream data ingestion pipeline must be configured to calculate `Average_Price` and `Volatility` dynamically before passing inputs to the model.*

## 👨‍💻 Author
**Shrinidhi Walvekar**
