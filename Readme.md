# Vomitoxin Level Prediction from Hyperspectral Imaging Data

## Project Overview
This project focuses on predicting vomitoxin (DON) levels in corn samples using hyperspectral imaging data. The goal is to develop a machine learning model that can accurately predict vomitoxin contamination levels from spectral signatures, providing a non-destructive testing method.

## Dataset Description
- **Input Features**: 448 spectral bands from hyperspectral imaging
- **Target Variable**: Vomitoxin concentration (ppb - parts per billion)
- **Dataset Size**: 500 samples
- **Data Format**: CSV file with spectral readings and corresponding vomitoxin levels

## Data Preprocessing
1. **Feature Scaling**: StandardScaler applied to normalize spectral data
2. **Dimensionality Reduction**: PCA (Principal Component Analysis) used to reduce feature dimensions while preserving data variance
3. **Train-Test Split**: Data split into training (80%) and testing (20%) sets

## Models Implemented

### 1. Random Forest Regressor
- **Best Performance**:
  - R² Score: 0.959
  - Adjusted R² Score: 0.957
  - RMSE: 3,399.22 ppb

- **Model Configuration**:
  - Algorithm: Random Forest Regression
  - Number of Trees: 600
  - Random State: 42
  - Cross-validation: 5-fold

### 2. Convolutional Neural Network (CNN)
- **Performance Metrics**:
  - Training Loss: 98,098,928.0
  - Training MAE: 2,685.71
  - Validation Loss: 298,972,704.0
  - Validation MAE: 4,417.25
  - Test MAE: 3,552.24

## Key Findings
1. **Model Comparison**:
   - Random Forest outperformed CNN significantly
   - RF showed excellent predictive capability with R² > 0.95
   - CNN showed higher error rates and instability

2. **Feature Importance**:
   - Random Forest identified key spectral bands
   - PCA helped reduce dimensionality while maintaining predictive power

3. **Prediction Accuracy**:
   - RF model achieved high accuracy in vomitoxin prediction
   - Error rates well within acceptable range for practical applications

## Conclusions
- Random Forest proves to be more suitable for this specific problem
- The model successfully predicts vomitoxin levels with high accuracy
- Non-destructive testing using hyperspectral imaging is viable

## Future Improvements
1. Feature Engineering:
   - Investigate domain-specific spectral band combinations
   - Explore additional preprocessing techniques

2. Model Optimization:
   - Hyperparameter tuning using techniques like Optuna
   - Ensemble methods combining multiple models
   - Deep learning architecture optimization

3. Validation:
   - Additional cross-validation strategies
   - External validation datasets
   - Robustness testing

## Requirements
- Python 3.9+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - tensorflow, keras (for CNN)

## Usage
1. Data Preparation:
   ```python
   df = pd.read_csv('TASK-ML-INTERN.csv')
   X = df.drop(['hsi_id', 'vomitoxin_ppb'], axis=1)
   y = df['vomitoxin_ppb']
   ```

2. Model Training:
   ```python
   rf_model = RandomForestRegressor(n_estimators=600, random_state=42)
   rf_model.fit(X_train, y_train)
   ```

3. Prediction:
   ```python
   y_pred = rf_model.predict(X_test)
   ```

## Project Structure
```
.
├── main.ipynb           # Main notebook with analysis and models
├── TASK-ML-INTERN.csv  # Dataset file
└── README.md           # Project documentation
```

## Author
Uday om srivastava
ML Intern Candidate

*Note: This project was completed as part of the ML Internship assignment at ImagoAI.*