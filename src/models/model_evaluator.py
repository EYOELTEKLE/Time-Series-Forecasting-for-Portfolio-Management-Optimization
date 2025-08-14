"""
Model evaluation utilities for time series forecasting models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluator for time series forecasting models.
    
    This class provides various evaluation metrics and visualization tools
    for comparing different forecasting models.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.results = {}
        self.models = {}
        
    def calculate_metrics(self, 
                         y_true: pd.Series, 
                         y_pred: pd.Series, 
                         model_name: str = "model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model for identification
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            raise ValueError("No valid data points for evaluation")
        
        # Basic metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        
        # Percentage errors
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        mpe = np.mean((y_true_clean - y_pred_clean) / y_true_clean) * 100
        
        # Additional metrics
        mae_percent = mae / np.mean(np.abs(y_true_clean)) * 100
        rmse_percent = rmse / np.mean(y_true_clean) * 100
        
        # Directional accuracy
        direction_accuracy = np.mean(
            np.sign(y_true_clean.diff().dropna()) == np.sign(y_pred_clean.diff().dropna())
        ) * 100
        
        # Theil's U statistic (values < 1 indicate better than naive forecast)
        naive_forecast = y_true_clean.shift(1).dropna()
        y_true_naive = y_true_clean.iloc[1:]
        y_pred_naive = y_pred_clean.iloc[1:]
        
        if len(y_true_naive) > 0:
            theil_u = np.sqrt(
                np.mean((y_pred_naive - y_true_naive) ** 2)
            ) / np.sqrt(
                np.mean((naive_forecast - y_true_naive) ** 2)
            )
        else:
            theil_u = np.nan
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'MPE': mpe,
            'MAE_Percent': mae_percent,
            'RMSE_Percent': rmse_percent,
            'Directional_Accuracy': direction_accuracy,
            'Theil_U': theil_u
        }
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_model(self, 
                      model: Any, 
                      train_data: pd.Series, 
                      test_data: pd.Series, 
                      model_name: str,
                      is_multivariate: bool = False) -> Dict[str, float]:
        """
        Evaluate a fitted model on test data.
        
        Args:
            model: Fitted model object
            train_data: Training data
            test_data: Test data
            model_name: Name of the model
            is_multivariate: Whether the model is multivariate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            if is_multivariate:
                # For multivariate models (like VAR)
                forecast = model.forecast(steps=len(test_data))
                if isinstance(forecast, tuple):
                    forecast = forecast[0]  # Extract forecast values
                
                # For multivariate, we need to handle each variable separately
                if isinstance(forecast, np.ndarray) and forecast.ndim > 1:
                    metrics = {}
                    for i, col in enumerate(test_data.columns):
                        col_metrics = self.calculate_metrics(
                            test_data[col], 
                            pd.Series(forecast[:, i], index=test_data.index),
                            f"{model_name}_{col}"
                        )
                        metrics[col] = col_metrics
                    return metrics
                else:
                    return self.calculate_metrics(test_data, forecast, model_name)
            else:
                # For univariate models (like ARIMA)
                forecast = model.forecast(steps=len(test_data))
                if isinstance(forecast, tuple):
                    forecast = forecast[0]  # Extract forecast values
                
                return self.calculate_metrics(test_data, forecast, model_name)
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            return {}
    
    def compare_models(self, 
                      models: Dict[str, Any], 
                      train_data: pd.Series, 
                      test_data: pd.Series,
                      is_multivariate: bool = False) -> pd.DataFrame:
        """
        Compare multiple models and return a comparison DataFrame.
        
        Args:
            models: Dictionary of {model_name: model_object}
            train_data: Training data
            test_data: Test data
            is_multivariate: Whether the models are multivariate
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, train_data, test_data, model_name, is_multivariate)
            
            if metrics:
                if isinstance(metrics, dict) and any(isinstance(v, dict) for v in metrics.values()):
                    # Multivariate results
                    for var_name, var_metrics in metrics.items():
                        var_metrics['Model'] = model_name
                        var_metrics['Variable'] = var_name
                        comparison_results.append(var_metrics)
                else:
                    # Univariate results
                    metrics['Model'] = model_name
                    comparison_results.append(metrics)
        
        return pd.DataFrame(comparison_results)
    
    def plot_forecast_comparison(self, 
                               models: Dict[str, Any], 
                               train_data: pd.Series, 
                               test_data: pd.Series,
                               figsize: Tuple[int, int] = (15, 10),
                               is_multivariate: bool = False):
        """
        Plot forecast comparison for multiple models.
        
        Args:
            models: Dictionary of {model_name: model_object}
            train_data: Training data
            test_data: Test data
            figsize: Figure size
            is_multivariate: Whether the models are multivariate
        """
        plt.figure(figsize=figsize)
        
        # Plot training data
        plt.plot(train_data.index, train_data, label='Training Data', color='blue', linewidth=2)
        
        # Plot test data
        plt.plot(test_data.index, test_data, label='Actual Test Data', color='black', linewidth=2)
        
        # Plot forecasts for each model
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        for i, (model_name, model) in enumerate(models.items()):
            try:
                if is_multivariate:
                    forecast = model.forecast(steps=len(test_data))
                    if isinstance(forecast, tuple):
                        forecast = forecast[0]
                    
                    # For multivariate, plot first variable or mean
                    if isinstance(forecast, np.ndarray) and forecast.ndim > 1:
                        forecast_series = pd.Series(forecast[:, 0], index=test_data.index)
                    else:
                        forecast_series = pd.Series(forecast, index=test_data.index)
                else:
                    forecast = model.forecast(steps=len(test_data))
                    if isinstance(forecast, tuple):
                        forecast = forecast[0]
                    forecast_series = pd.Series(forecast, index=test_data.index)
                
                color = colors[i % len(colors)]
                plt.plot(test_data.index, forecast_series, 
                        label=f'{model_name} Forecast', 
                        color=color, linestyle='--', linewidth=1.5)
                
            except Exception as e:
                print(f"Error plotting {model_name}: {str(e)}")
        
        plt.title('Forecast Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, 
                              metrics_df: pd.DataFrame, 
                              metrics: List[str] = ['RMSE', 'MAPE', 'MAE'],
                              figsize: Tuple[int, int] = (15, 5)):
        """
        Plot comparison of metrics across models.
        
        Args:
            metrics_df: DataFrame with model comparison results
            metrics: List of metrics to plot
            figsize: Figure size
        """
        if 'Variable' in metrics_df.columns:
            # Multivariate case
            fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                if metric in metrics_df.columns:
                    pivot_df = metrics_df.pivot(index='Model', columns='Variable', values=metric)
                    pivot_df.plot(kind='bar', ax=axes[i], title=f'{metric} Comparison')
                    axes[i].set_ylabel(metric)
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend(title='Variable')
        else:
            # Univariate case
            fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                if metric in metrics_df.columns:
                    metrics_df.plot(x='Model', y=metric, kind='bar', ax=axes[i], title=f'{metric} Comparison')
                    axes[i].set_ylabel(metric)
                    axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals_analysis(self, 
                               model: Any, 
                               test_data: pd.Series, 
                               model_name: str = "Model",
                               figsize: Tuple[int, int] = (15, 10)):
        """
        Plot residual analysis for a model.
        
        Args:
            model: Fitted model
            test_data: Test data
            model_name: Name of the model
            figsize: Figure size
        """
        try:
            # Get forecasts
            forecast = model.forecast(steps=len(test_data))
            if isinstance(forecast, tuple):
                forecast = forecast[0]
            
            forecast_series = pd.Series(forecast, index=test_data.index)
            
            # Calculate residuals
            residuals = test_data - forecast_series
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Residuals over time
            axes[0, 0].plot(residuals.index, residuals, color='blue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_title(f'{model_name} - Residuals Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals histogram
            axes[0, 1].hist(residuals.dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 1].set_title(f'{model_name} - Residuals Distribution')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title(f'{model_name} - Q-Q Plot')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Residuals vs Fitted
            axes[1, 1].scatter(forecast_series, residuals, alpha=0.6, color='blue')
            axes[1, 1].axhline(y=0, color='red', linestyle='--')
            axes[1, 1].set_title(f'{model_name} - Residuals vs Fitted')
            axes[1, 1].set_xlabel('Fitted Values')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting residuals for {model_name}: {str(e)}")
    
    def generate_report(self, metrics_df: pd.DataFrame) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics_df: DataFrame with model comparison results
            
        Returns:
            Formatted report string
        """
        report = "=" * 80 + "\n"
        report += "TIME SERIES FORECASTING MODEL EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Best model for each metric
        metrics_to_check = ['RMSE', 'MAPE', 'MAE', 'Directional_Accuracy']
        
        for metric in metrics_to_check:
            if metric in metrics_df.columns:
                if 'Variable' in metrics_df.columns:
                    # Multivariate case
                    report += f"\n{metric} Analysis (by Variable):\n"
                    report += "-" * 40 + "\n"
                    for var in metrics_df['Variable'].unique():
                        var_data = metrics_df[metrics_df['Variable'] == var]
                        best_model = var_data.loc[var_data[metric].idxmin() if metric != 'Directional_Accuracy' else var_data[metric].idxmax()]
                        report += f"{var}: {best_model['Model']} ({best_model[metric]:.4f})\n"
                else:
                    # Univariate case
                    best_model = metrics_df.loc[metrics_df[metric].idxmin() if metric != 'Directional_Accuracy' else metrics_df[metric].idxmax()]
                    report += f"\nBest {metric}: {best_model['Model']} ({best_model[metric]:.4f})\n"
        
        # Summary statistics
        report += "\n" + "=" * 40 + "\n"
        report += "SUMMARY STATISTICS\n"
        report += "=" * 40 + "\n"
        
        if 'Variable' in metrics_df.columns:
            for var in metrics_df['Variable'].unique():
                report += f"\n{var}:\n"
                var_data = metrics_df[metrics_df['Variable'] == var]
                for metric in metrics_to_check:
                    if metric in var_data.columns:
                        report += f"  {metric}: Mean={var_data[metric].mean():.4f}, Std={var_data[metric].std():.4f}\n"
        else:
            for metric in metrics_to_check:
                if metric in metrics_df.columns:
                    report += f"{metric}: Mean={metrics_df[metric].mean():.4f}, Std={metrics_df[metric].std():.4f}\n"
        
        return report
