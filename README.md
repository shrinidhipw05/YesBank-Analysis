# 📈 Yes Bank Stock Price Prediction

## 📝 Project Overview
The primary objective of this capstone project is to develop a highly accurate machine learning regression model capable of predicting the monthly closing price of Yes Bank's stock. This project tackles significant data challenges—specifically, navigating extreme multicollinearity and a non-stationary target distribution caused by a severe structural break in the stock's history.

## 🏦 Background: The Rise and Fall of Yes Bank
To understand the unique challenges of this dataset, it is essential to understand the bank's history:
*   **Rapid Rise:** Founded in 2004, Yes Bank grew meteorically to become one of India's largest private-sector lenders. It was highly favored by investors and known for aggressive corporate lending.
*   **The 2018 Crisis:** The bank's rapid growth masked severe underlying issues, including reckless lending to distressed companies (such as IL&FS and DHFL) and the under-reporting of Non-Performing Assets (NPAs). When these massive bad loans came to light in 2018, it triggered a major corporate governance crisis. 
*   **The Collapse:** The crisis caused depositors to withdraw funds rapidly, and the bank was unable to raise capital. Yes Bank's stock, which traded at nearly ₹400 in 2018, plummeted by roughly 85% to single digits.
*   **RBI Intervention:** To prevent a systemic banking collapse, the Reserve Bank of India (RBI) intervened in 2020. The RBI took over the bank's management, imposed a 30-day moratorium capping customer withdrawals, and orchestrated a bailout led by the State Bank of India (SBI). 

*This history explains the "structural break" in the data—the stock's behavior before the 2018 crisis was fundamentally different from its behavior during and after the crash, requiring advanced feature engineering to build a reliable predictive model.*

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
