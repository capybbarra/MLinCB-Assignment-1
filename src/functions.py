# ./src/functions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(data_path):
    """Load and preprocess the metagenomic data"""
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Unnamed: 0', 'Project ID', 'Experiment type', 'Sex', 'Host age', 'BMI', 'Disease MESH ID'])  # Assuming BMI is the target column 'Unnamed: 0', 'Project ID', 'Experiment type', 'Sex', 'Host age', 
    y = data['BMI']


    return X, y

    

def baseline_model_fit_save(model, model_name, X_train, y_train, path = "/models", prefix='baseline'):
    """
    Establish baseline model with all features and default parameters.
    Returns the fitted pipeline with RMSE score and the model.
    """
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), X_train.columns)],
        remainder='drop'
    )

    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)  # Add model as a step in the pipeline
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Save the entire pipeline (model + preprocessing)
    joblib.dump(pipeline, f'.{path}/{prefix}_{model_name}.joblib')
    
    return pipeline


def bootstrap_evaluate(model, X_test, y_test, n_bootstraps=100, random_state=42):
    """Perform bootstrapping on the evaluation dataset to get repeated performance estimates."""

    np.random.seed(random_state)
    n_samples = X_test.shape[0]
    
    rmse_scores, mae_scores, r2_scores = [], [], []
    
    for i in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample = X_test.iloc[indices]
        y_sample = y_test.iloc[indices]
        
        y_pred = model.predict(X_sample)
        rmse_scores.append(np.sqrt(mean_squared_error(y_sample, y_pred)))
        mae_scores.append(mean_absolute_error(y_sample, y_pred))
        r2_scores.append(r2_score(y_sample, y_pred))
    
    return pd.DataFrame({
        'RMSE': rmse_scores,
        'MAE': mae_scores,
        'R2': r2_scores
    })

def plot_metrics_boxplot(results_df, name='Model Performance Metrics Comparison', save_path=None):
    """Create boxplots of metrics for comparison across models."""
    melted = results_df.melt(id_vars='model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x='Metric', y='Score', hue='model')
    plt.title(name)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def compute_confidence_interval(metric_scores, alpha=0.05):
    lower = np.percentile(metric_scores, 100 * (alpha/2))
    upper = np.percentile(metric_scores, 100 * (1 - alpha/2))
    return lower, upper



def elasticnet_feature_selection(X, y, alpha=1.0, l1_ratio=0.5, threshold=None, plot=True):
    """
    Perform feature selection using ElasticNet with visualization.
    
    Parameters:
    X (pd.DataFrame): Features DataFrame
    y (pd.Series): Target variable
    alpha: Constant that multiplies the penalty terms
    l1_ratio: Mixing parameter (0 for L2, 1 for L1)
    threshold: The threshold for feature selection
    plot: Whether to plot feature coefficients
    
    Returns:
    pd.DataFrame: DataFrame with selected features
    pd.Series: Feature coefficients
    """
    # Initialize ElasticNet
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), X.columns)
        ],
        remainder='drop'
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectFromModel(enet, threshold=threshold))
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Get the fitted ElasticNet model from the pipeline
    fitted_enet = pipeline.named_steps['selector'].estimator_
    
    # Get selected feature mask and names
    selected_mask = pipeline.named_steps['selector'].get_support()
    selected_features = X.columns[selected_mask]
    
    # Transform the data
    transformed_data = pipeline.transform(X)

    
    # Get the coefficients (absolute values for importance)
    coefficients = pd.Series(np.abs(fitted_enet.coef_), index=X.columns)
    
    # Plot coefficients if requested
    if plot:
        plt.figure(figsize=(12, 8))
        coefficients.sort_values(ascending=False).plot(kind='bar')
        plt.title(f'ElasticNet Coefficients (α={alpha}, L1 ratio={l1_ratio})')
        plt.ylabel('Absolute Coefficient Value')
        if threshold:
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/feature_importances_{l1_ratio}.png")
        plt.show()
    
    return selected_features, coefficients


def hyperparameter_tuning_save(model, model_name, param_grid, X_train, y_train, 
                              cv=5, scoring='neg_root_mean_squared_error',
                              n_jobs=-1, random_state=42):
    """
    Perform hyperparameter tuning with grid search and cross-validation.
    Saves best model and returns it with validation results.
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=n_jobs
    )
    grid_search.fit(X_train, y_train)
    
    # Save best model
    joblib.dump(grid_search.best_estimator_, f'./final_models/tuned_{model_name}.joblib')
    
    return grid_search.best_estimator_, grid_search.cv_results_


