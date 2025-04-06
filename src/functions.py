# ./src/functions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

def load_data(data_path):
    """Load and preprocess the metagenomic data"""
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Unnamed: 0', 'Project ID', 'Experiment type', 'Sex', 'Host age', 'BMI', 'Disease MESH ID'])  # Assuming BMI is the target column 'Unnamed: 0', 'Project ID', 'Experiment type', 'Sex', 'Host age', 
    y = data['BMI']


    return X, y

def scale(X_train, X_test):

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
    

def baseline_model_fit_save(model, model_name, X_train, y_train):
    """
    Establish baseline model with all features and default parameters
    Returns RMSE score and the model
    """

    
    # Train baseline model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f'./models/baseline_{model_name}.joblib')
    
    return model


def bootstrap_evaluate(model, X_test, y_test, n_bootstraps=100, random_state=42):
    """Perform bootstrapping on the evaluation dataset to get repeated performance estimates."""

    np.random.seed(random_state)
    n_samples = X_test.shape[0]
    
    rmse_scores, mae_scores, r2_scores = [], [], []
    
    for i in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        
        y_pred = model.predict(X_sample)
        rmse_scores.append(np.sqrt(mean_squared_error(y_sample, y_pred)))
        mae_scores.append(mean_absolute_error(y_sample, y_pred))
        r2_scores.append(r2_score(y_sample, y_pred))
    
    return pd.DataFrame({
        'RMSE': rmse_scores,
        'MAE': mae_scores,
        'R2': r2_scores
    })

def plot_metrics_boxplot(results_df, save_path=None):
    """Create boxplots of metrics for comparison across models."""
    melted = results_df.melt(id_vars='model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x='Metric', y='Score', hue='model')
    plt.title('Model Performance Metrics Comparison')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def compute_confidence_interval(metric_scores, alpha=0.05):
    lower = np.percentile(metric_scores, 100 * (alpha/2))
    upper = np.percentile(metric_scores, 100 * (1 - alpha/2))
    return lower, upper

def stability_selection(X, y, n_iterations=100, threshold=0.5, alpha=0.1):
    """
    Perform stability selection using ElasticNet.

    Parameters:
        X (np.array): Feature matrix for the development set.
        y (np.array): Target vector for the development set.
        n_iterations (int): Number of bootstrap iterations.
        threshold (float): Minimum selection frequency required for a feature to be selected.
        alpha (float): Regularization strength for Lasso.

    Returns:
        stable_features (np.array): Indices of features with selection frequency above threshold.
        selection_frequency (np.array): Frequency of selection for each feature.
    """
    n_samples, n_features = X.shape
    selected_counts = np.zeros(n_features)

    for i in range(n_iterations):
        # Bootstrap sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        model = Lasso(alpha=alpha, random_state=i)
        model.fit(X_boot, y_boot)
        
        # Increment count for features with non-zero coefficients
        selected_counts += (np.abs(model.coef_) > 1e-5)

    # Calculate the selection frequency for each feature
    selection_frequency = selected_counts / n_iterations
    stable_features = np.where(selection_frequency >= threshold)[0]

    return stable_features, selection_frequency







##########################
def feature_selection(model, X_train, y_train, X_test, y_test):
    """
    Perform feature selection using ElasticNet's built-in feature importance
    Returns selected features and RMSE score
    """
    
    # First train model to get feature importances
    #trained_model = model.fit(X_train, y_train)
    
    # Select features using SelectFromModel
    selector = SelectFromModel(model, prefit=False)
    X_train_selected = selector.transform(X_train)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()]
    
    # Train model with selected features
    model_selected = model.fit(X_train_selected, y_train)
    
    return selected_features, model_selected



def tune_model(X_train, y_train, selected_features):
    """
    Perform hyperparameter tuning with cross-validation
    Returns best model and best parameters
    """
    # Filter data to only selected features
    X_train_selected = X_train[selected_features]
    
    # Create pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__l1_ratio': np.linspace(0.1, 0.9, 9)
    }
    
    # Perform grid search with 5-fold CV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_selected, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

