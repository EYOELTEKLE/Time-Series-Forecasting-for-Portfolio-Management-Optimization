"""
Data loader and preprocessing utilities for financial time series data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class FinancialDataLoader:
    """
    Comprehensive data loader for financial time series with preprocessing capabilities.
    """
    
    def __init__(self, tickers: List[str], start_date: str = '2018-01-01', end_date: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date for data collection
            end_date: End date for data collection (None for current date)
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None
        self.processed_data = None
        self.close_data = None
        self.returns_data = None
        
    def download_data(self) -> pd.DataFrame:
        """
        Download financial data from Yahoo Finance.
        
        Returns:
            Raw financial data
        """
        print(f"Downloading data for {self.tickers} from {self.start_date} to {self.end_date or 'present'}")
        
        try:
            self.raw_data = yf.download(
                self.tickers, 
                start=self.start_date, 
                end=self.end_date, 
                group_by='ticker', 
                auto_adjust=True,
                progress=False
            )
            
            print(f"Successfully downloaded {len(self.raw_data)} data points")
            return self.raw_data
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            raise
    
    def extract_close_prices(self) -> pd.DataFrame:
        """
        Extract closing prices for all tickers.
        
        Returns:
            DataFrame with closing prices
        """
        if self.raw_data is None:
            self.download_data()
        
        close_data = {}
        for ticker in self.tickers:
            if len(self.tickers) == 1:
                close_data[ticker] = self.raw_data['Close']
            else:
                close_data[ticker] = self.raw_data[ticker]['Close']
        
        self.close_data = pd.DataFrame(close_data)
        print(f"Extracted closing prices for {len(self.tickers)} assets")
        return self.close_data
    
    def calculate_returns(self, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from closing prices.
        
        Args:
            method: 'log' for log returns, 'simple' for simple returns
            
        Returns:
            DataFrame with returns
        """
        if self.close_data is None:
            self.extract_close_prices()
        
        if method == 'log':
            self.returns_data = np.log(self.close_data / self.close_data.shift(1))
        else:
            self.returns_data = self.close_data.pct_change()
        
        # Remove first row (NaN)
        self.returns_data = self.returns_data.dropna()
        
        print(f"Calculated {method} returns for {len(self.tickers)} assets")
        return self.returns_data
    
    def handle_missing_data(self, method: str = 'ffill', max_gap: int = 5) -> pd.DataFrame:
        """
        Handle missing data with comprehensive strategies.
        
        Args:
            method: 'ffill' (forward fill), 'bfill' (backward fill), 'interpolate'
            max_gap: Maximum gap to fill before dropping
            
        Returns:
            Cleaned data
        """
        if self.close_data is None:
            self.extract_close_prices()
        
        cleaned_data = self.close_data.copy()
        
        # Check for missing data
        missing_summary = cleaned_data.isnull().sum()
        print(f"Missing data summary:\n{missing_summary}")
        
        # Handle missing data based on method
        if method == 'ffill':
            cleaned_data = cleaned_data.fillna(method='ffill')
        elif method == 'bfill':
            cleaned_data = cleaned_data.fillna(method='bfill')
        elif method == 'interpolate':
            cleaned_data = cleaned_data.interpolate(method='linear')
        
        # Drop rows with too many consecutive missing values
        for col in cleaned_data.columns:
            # Find consecutive missing values
            missing_mask = cleaned_data[col].isnull()
            missing_groups = missing_mask.groupby((~missing_mask).cumsum()).sum()
            
            # Drop if gap is too large
            if missing_groups.max() > max_gap:
                print(f"Warning: Large gap in {col}, dropping affected periods")
        
        # Final cleanup
        cleaned_data = cleaned_data.dropna()
        
        print(f"Data cleaned: {len(cleaned_data)} rows remaining")
        return cleaned_data
    
    def create_train_test_split(self, test_size: float = 0.2, method: str = 'chronological') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split for time series data.
        
        Args:
            test_size: Proportion of data for testing
            method: 'chronological' for time-based split
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.close_data is None:
            self.extract_close_prices()
        
        if method == 'chronological':
            split_idx = int(len(self.close_data) * (1 - test_size))
            train_data = self.close_data.iloc[:split_idx]
            test_data = self.close_data.iloc[split_idx:]
            
            print(f"Train set: {len(train_data)} rows ({train_data.index[0]} to {train_data.index[-1]})")
            print(f"Test set: {len(test_data)} rows ({test_data.index[0]} to {test_data.index[-1]})")
            
            return train_data, test_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive data summary.
        
        Returns:
            Dictionary with data statistics
        """
        if self.close_data is None:
            self.extract_close_prices()
        
        summary = {}
        
        for ticker in self.tickers:
            ticker_data = self.close_data[ticker]
            
            summary[ticker] = {
                'start_date': ticker_data.index[0],
                'end_date': ticker_data.index[-1],
                'total_observations': len(ticker_data),
                'missing_values': ticker_data.isnull().sum(),
                'mean_price': ticker_data.mean(),
                'std_price': ticker_data.std(),
                'min_price': ticker_data.min(),
                'max_price': ticker_data.max(),
                'current_price': ticker_data.iloc[-1],
                'total_return': (ticker_data.iloc[-1] / ticker_data.iloc[0] - 1) * 100
            }
        
        return summary
    
    def plot_data_overview(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive data visualization.
        
        Args:
            figsize: Figure size
        """
        if self.close_data is None:
            self.extract_close_prices()
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Price evolution
        for ticker in self.tickers:
            axes[0, 0].plot(self.close_data.index, self.close_data[ticker], label=ticker)
        axes[0, 0].set_title('Price Evolution')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        if self.returns_data is not None:
            for ticker in self.tickers:
                axes[0, 1].hist(self.returns_data[ticker].dropna(), bins=50, alpha=0.7, label=ticker)
            axes[0, 1].set_title('Returns Distribution')
            axes[0, 1].set_xlabel('Returns')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation matrix
        if len(self.tickers) > 1:
            corr_matrix = self.close_data.corr()
            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 0].set_title('Correlation Matrix')
            axes[1, 0].set_xticks(range(len(self.tickers)))
            axes[1, 0].set_yticks(range(len(self.tickers)))
            axes[1, 0].set_xticklabels(self.tickers)
            axes[1, 0].set_yticklabels(self.tickers)
            plt.colorbar(im, ax=axes[1, 0])
        
        # Volatility over time
        if self.returns_data is not None:
            for ticker in self.tickers:
                rolling_vol = self.returns_data[ticker].rolling(window=30).std() * np.sqrt(252)
                axes[1, 1].plot(rolling_vol.index, rolling_vol, label=ticker)
            axes[1, 1].set_title('30-Day Rolling Volatility')
            axes[1, 1].set_ylabel('Annualized Volatility')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class DataPreprocessor:
    """
    Advanced data preprocessing utilities for financial time series.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize preprocessor with data.
        
        Args:
            data: Input data for preprocessing
        """
        self.data = data.copy()
        self.original_data = data.copy()
        
    def check_stationarity(self) -> Dict[str, Dict[str, Any]]:
        """
        Check stationarity of all time series.
        
        Returns:
            Dictionary with stationarity test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        results = {}
        
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            results[column] = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        
        return results
    
    def make_stationary(self, max_diff: int = 2) -> pd.DataFrame:
        """
        Make time series stationary by differencing.
        
        Args:
            max_diff: Maximum number of differences to apply
            
        Returns:
            Stationary data
        """
        stationary_data = self.data.copy()
        differences_applied = {}
        
        for column in self.data.columns:
            diff_count = 0
            current_data = self.data[column]
            
            while diff_count < max_diff:
                adf_result = adfuller(current_data.dropna())
                if adf_result[1] < 0.05:  # Stationary
                    break
                current_data = current_data.diff()
                diff_count += 1
            
            if diff_count > 0:
                stationary_data[column] = self.data[column].diff(diff_count)
                differences_applied[column] = diff_count
        
        # Remove NaN values from differencing
        stationary_data = stationary_data.dropna()
        
        print(f"Applied differences: {differences_applied}")
        return stationary_data
    
    def normalize_data(self, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize data using various methods.
        
        Args:
            method: 'zscore', 'minmax', 'robust'
            
        Returns:
            Normalized data
        """
        normalized_data = self.data.copy()
        
        if method == 'zscore':
            normalized_data = (normalized_data - normalized_data.mean()) / normalized_data.std()
        elif method == 'minmax':
            normalized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())
        elif method == 'robust':
            normalized_data = (normalized_data - normalized_data.median()) / normalized_data.mad()
        
        return normalized_data
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Returns:
            Data with technical indicators
        """
        enhanced_data = self.data.copy()
        
        for column in self.data.columns:
            # Moving averages
            enhanced_data[f'{column}_MA_20'] = self.data[column].rolling(window=20).mean()
            enhanced_data[f'{column}_MA_50'] = self.data[column].rolling(window=50).mean()
            
            # Volatility
            enhanced_data[f'{column}_Volatility_20'] = self.data[column].rolling(window=20).std()
            
            # Momentum
            enhanced_data[f'{column}_Momentum_10'] = self.data[column] / self.data[column].shift(10) - 1
        
        return enhanced_data.dropna()
