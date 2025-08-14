"""
VAR (Vector Autoregression) model implementation for multivariate time series.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.select_order import select_order
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class VARModel:
    """
    VAR model for multivariate time series forecasting.
    
    This class provides a comprehensive implementation of VAR modeling
    with automatic lag selection, stationarity testing, and forecasting.
    """
    
    def __init__(self, 
                 maxlags: int = 15,
                 auto_select_lags: bool = True,
                 ic: str = 'aic',
                 trend: str = 'c'):
        """
        Initialize VAR model.
        
        Args:
            maxlags: Maximum number of lags to consider
            auto_select_lags: Whether to automatically select best lag order
            ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
            trend: Trend specification ('n', 'c', 'ct', 'ctt')
        """
        self.maxlags = maxlags
        self.auto_select_lags = auto_select_lags
        self.ic = ic
        self.trend = trend
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.best_lag_order = None
        self.aic = None
        self.bic = None
        self.hqic = None
        self.fpe = None
        
    def check_stationarity(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Check if all time series in the dataframe are stationary using ADF test.
        
        Args:
            data: Multivariate time series data
            
        Returns:
            Dictionary containing test results for each variable
        """
        results = {}
        
        for column in data.columns:
            result = adfuller(data[column].dropna())
            results[column] = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        
        return results
    
    def make_stationary(self, data: pd.DataFrame, max_diff: int = 2) -> Tuple[pd.DataFrame, List[int]]:
        """
        Make time series stationary by differencing.
        
        Args:
            data: Multivariate time series data
            max_diff: Maximum number of differences to apply
            
        Returns:
            Tuple of (stationary_data, differences_applied)
        """
        stationary_data = data.copy()
        differences = [0] * len(data.columns)
        
        for i, column in enumerate(data.columns):
            diff_count = 0
            current_data = data[column]
            
            while diff_count < max_diff:
                adf_result = adfuller(current_data.dropna())
                if adf_result[1] < 0.05:  # Stationary
                    break
                current_data = current_data.diff().dropna()
                diff_count += 1
            
            if diff_count > 0:
                stationary_data[column] = data[column].diff(diff_count)
                differences[i] = diff_count
        
        return stationary_data.dropna(), differences
    
    def find_best_lag_order(self, data: pd.DataFrame) -> int:
        """
        Automatically find the best lag order using information criteria.
        
        Args:
            data: Multivariate time series data
            
        Returns:
            Best lag order
        """
        print(f"Finding best lag order using {self.ic.upper()}...")
        
        lag_order_results = select_order(data, maxlags=self.maxlags, trend=self.trend)
        
        self.best_lag_order = lag_order_results.selected_orders[self.ic]
        self.aic = lag_order_results.ic_table['aic'].min()
        self.bic = lag_order_results.ic_table['bic'].min()
        self.hqic = lag_order_results.ic_table['hqic'].min()
        self.fpe = lag_order_results.ic_table['fpe'].min()
        
        print(f"Best lag order: {self.best_lag_order}")
        print(f"AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")
        print(f"HQIC: {self.hqic:.2f}, FPE: {self.fpe:.2f}")
        
        return self.best_lag_order
    
    def fit(self, data: pd.DataFrame, make_stationary: bool = True) -> 'VARModel':
        """
        Fit the VAR model to the data.
        
        Args:
            data: Multivariate time series data to fit
            make_stationary: Whether to make data stationary before fitting
            
        Returns:
            Self for method chaining
        """
        # Check stationarity and make stationary if needed
        if make_stationary:
            stationarity_results = self.check_stationarity(data)
            non_stationary = [col for col, result in stationarity_results.items() 
                            if not result['is_stationary']]
            
            if non_stationary:
                print(f"Making non-stationary series stationary: {non_stationary}")
                data, differences = self.make_stationary(data)
                print(f"Applied differences: {dict(zip(data.columns, differences))}")
        
        # Find best lag order if auto-selection is enabled
        if self.auto_select_lags:
            self.best_lag_order = self.find_best_lag_order(data)
        else:
            self.best_lag_order = self.maxlags
        
        # Create and fit the model
        self.model = VAR(data)
        self.fitted_model = self.model.fit(maxlags=self.best_lag_order, trend=self.trend)
        self.is_fitted = True
        
        print(f"VAR model fitted successfully with lag order {self.best_lag_order}")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
        
        return self
    
    def forecast(self, steps: int, return_conf_int: bool = True) -> pd.DataFrame:
        """
        Generate forecasts for the specified number of steps.
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(self.fitted_model.y, steps=steps)
        
        if return_conf_int:
            conf_int = self.fitted_model.forecast_interval(self.fitted_model.y, steps=steps)
            return forecast_result, conf_int
        
        return forecast_result
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make in-sample predictions.
        
        Args:
            data: Data to make predictions on
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.fitted_model.fittedvalues
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the fitted model.
        
        Returns:
            Model summary as string
        """
        if not self.is_fitted:
            return "Model not fitted yet"
        
        return str(self.fitted_model.summary())
    
    def get_residuals(self) -> pd.DataFrame:
        """
        Get model residuals.
        
        Returns:
            Residuals from the fitted model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting residuals")
        
        return self.fitted_model.resid
    
    def get_granger_causality(self) -> pd.DataFrame:
        """
        Perform Granger causality tests between all variables.
        
        Returns:
            DataFrame with Granger causality test results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before testing Granger causality")
        
        return self.fitted_model.test_causality()
    
    def get_impulse_response(self, periods: int = 10, orth: bool = True) -> Any:
        """
        Get impulse response functions.
        
        Args:
            periods: Number of periods for impulse response
            orth: Whether to use orthogonalized impulse responses
            
        Returns:
            Impulse response results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing impulse responses")
        
        return self.fitted_model.irf(periods=periods, orth=orth)
    
    def plot_forecast(self, steps: int = 10, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot forecasts for all variables.
        
        Args:
            steps: Number of steps to forecast
            figsize: Figure size for the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting forecasts")
        
        forecast_result = self.forecast(steps)
        
        import matplotlib.pyplot as plt
        
        n_vars = len(self.fitted_model.names)
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize)
        
        if n_vars == 1:
            axes = [axes]
        
        for i, var_name in enumerate(self.fitted_model.names):
            # Plot historical data
            axes[i].plot(self.fitted_model.data.index, self.fitted_model.data[var_name], 
                        label='Historical', color='blue')
            
            # Plot forecast
            forecast_index = pd.date_range(
                start=self.fitted_model.data.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
            axes[i].plot(forecast_index, forecast_result[:, i], 
                        label='Forecast', color='red', linestyle='--')
            
            axes[i].set_title(f'VAR Forecast: {var_name}')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_impulse_response(self, periods: int = 10, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot impulse response functions.
        
        Args:
            periods: Number of periods for impulse response
            figsize: Figure size for the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting impulse responses")
        
        irf_result = self.get_impulse_response(periods=periods)
        irf_result.plot(figsize=figsize)
        plt.tight_layout()
        plt.show()
