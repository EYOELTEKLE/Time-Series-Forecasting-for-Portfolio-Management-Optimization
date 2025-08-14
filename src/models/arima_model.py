"""
ARIMA (AutoRegressive Integrated Moving Average) model implementation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """
    ARIMA model for univariate time series forecasting.
    
    This class provides a comprehensive implementation of ARIMA modeling
    with automatic parameter selection, stationarity testing, and forecasting.
    """
    
    def __init__(self, 
                 auto_select_params: bool = True,
                 order: Optional[Tuple[int, int, int]] = None,
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 max_p: int = 5,
                 max_d: int = 2,
                 max_q: int = 5,
                 seasonal: bool = False,
                 stepwise: bool = True,
                 suppress_warnings: bool = True,
                 trace: bool = False):
        """
        Initialize ARIMA model.
        
        Args:
            auto_select_params: Whether to automatically select best parameters
            order: Manual (p, d, q) order specification
            seasonal_order: Manual seasonal (P, D, Q, s) order specification
            max_p: Maximum AR order for auto-selection
            max_d: Maximum differencing order for auto-selection
            max_q: Maximum MA order for auto-selection
            seasonal: Whether to use seasonal ARIMA
            stepwise: Whether to use stepwise parameter selection
            suppress_warnings: Whether to suppress warnings
            trace: Whether to trace parameter selection process
        """
        self.auto_select_params = auto_select_params
        self.order = order
        self.seasonal_order = seasonal_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        self.trace = trace
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.best_order = None
        self.best_seasonal_order = None
        self.aic = None
        self.bic = None
        
    def check_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """
        Check if the time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary containing test results
        """
        result = adfuller(data.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def find_best_order(self, data: pd.Series) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
        """
        Automatically find the best ARIMA order using pmdarima.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (order, seasonal_order)
        """
        print("Finding best ARIMA parameters...")
        
        auto_model = pm.auto_arima(
            data,
            start_p=0, start_q=0,
            max_p=self.max_p, max_d=self.max_d, max_q=self.max_q,
            seasonal=self.seasonal,
            stepwise=self.stepwise,
            suppress_warnings=self.suppress_warnings,
            trace=self.trace,
            error_action='ignore',
            random_state=42
        )
        
        self.best_order = auto_model.order
        self.best_seasonal_order = auto_model.seasonal_order
        self.aic = auto_model.aic()
        self.bic = auto_model.bic()
        
        print(f"Best order: {self.best_order}")
        if self.seasonal and self.best_seasonal_order:
            print(f"Best seasonal order: {self.best_seasonal_order}")
        print(f"AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")
        
        return self.best_order, self.best_seasonal_order
    
    def fit(self, data: pd.Series) -> 'ARIMAModel':
        """
        Fit the ARIMA model to the data.
        
        Args:
            data: Time series data to fit
            
        Returns:
            Self for method chaining
        """
        if self.auto_select_params and self.order is None:
            self.order, self.seasonal_order = self.find_best_order(data)
        
        # Create and fit the model
        if self.seasonal and self.seasonal_order:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.model = SARIMAX(
                data, 
                order=self.order, 
                seasonal_order=self.seasonal_order
            )
        else:
            self.model = ARIMA(data, order=self.order)
        
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        
        print(f"Model fitted successfully with order {self.order}")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
        
        return self
    
    def forecast(self, steps: int, return_conf_int: bool = True) -> pd.Series:
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
        
        forecast_result = self.fitted_model.forecast(steps=steps)
        
        if return_conf_int:
            conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
            return forecast_result, conf_int
        
        return forecast_result
    
    def predict(self, data: pd.Series) -> pd.Series:
        """
        Make in-sample predictions.
        
        Args:
            data: Data to make predictions on
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.fitted_model.predict(start=0, end=len(data)-1)
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the fitted model.
        
        Returns:
            Model summary as string
        """
        if not self.is_fitted:
            return "Model not fitted yet"
        
        return str(self.fitted_model.summary())
    
    def get_residuals(self) -> pd.Series:
        """
        Get model residuals.
        
        Returns:
            Residuals from the fitted model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting residuals")
        
        return self.fitted_model.resid
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot model diagnostics.
        
        Args:
            figsize: Figure size for the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        import matplotlib.pyplot as plt
        self.fitted_model.plot_diagnostics(figsize=figsize)
        plt.tight_layout()
        plt.show()
