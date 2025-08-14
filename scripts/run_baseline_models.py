#!/usr/bin/env python3
"""
End-to-End Time Series Forecasting Pipeline for Portfolio Management

This script demonstrates the complete implementation of ARIMA and VAR models
for financial forecasting, including data preprocessing, model training,
evaluation, and generation of actionable investment insights.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.core.data_loader import FinancialDataLoader, DataPreprocessor
from src.models.arima_model import ARIMAModel
from src.models.var_model import VARModel
from src.models.model_evaluator import ModelEvaluator


class PortfolioForecastingPipeline:
    """
    Complete pipeline for portfolio forecasting and investment insights.
    """
    
    def __init__(self, tickers: list, start_date: str = '2018-01-01'):
        """
        Initialize the forecasting pipeline.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date for data collection
        """
        self.tickers = tickers
        self.start_date = start_date
        self.data_loader = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.forecasts = {}
        self.evaluator = ModelEvaluator()
        
    def load_and_preprocess_data(self):
        """Load and preprocess financial data."""
        print("=" * 60)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("=" * 60)
        
        # Initialize data loader
        self.data_loader = FinancialDataLoader(
            tickers=self.tickers,
            start_date=self.start_date
        )
        
        # Download and extract data
        self.data_loader.download_data()
        self.data_loader.extract_close_prices()
        
        # Handle missing data
        clean_data = self.data_loader.handle_missing_data(method='ffill')
        
        # Create train/test split
        self.train_data, self.test_data = self.data_loader.create_train_test_split(
            test_size=0.2
        )
        
        # Generate data summary
        summary = self.data_loader.get_data_summary()
        print("\nDATA SUMMARY:")
        for ticker, stats in summary.items():
            print(f"{ticker}:")
            print(f"  Period: {stats['start_date'].date()} to {stats['end_date'].date()}")
            print(f"  Total Return: {stats['total_return']:.2f}%")
            print(f"  Current Price: ${stats['current_price']:.2f}")
            print(f"  Volatility: {stats['std_price']:.2f}")
        
        # Plot data overview
        self.data_loader.plot_data_overview()
        
        return clean_data
    
    def train_arima_models(self):
        """Train ARIMA models for each asset."""
        print("\n" + "=" * 60)
        print("STEP 2: ARIMA MODEL TRAINING")
        print("=" * 60)
        
        for ticker in self.tickers:
            print(f"\nTraining ARIMA model for {ticker}...")
            
            # Create ARIMA model
            arima_model = ARIMAModel(
                auto_select_params=True,
                seasonal=False,
                trace=False
            )
            
            # Fit model
            train_series = self.train_data[ticker]
            arima_model.fit(train_series)
            
            # Store model
            self.models[f'ARIMA_{ticker}'] = arima_model
            
            print(f"ARIMA model for {ticker} fitted successfully")
            print(f"Best order: {arima_model.order}")
            print(f"AIC: {arima_model.aic:.2f}")
    
    def train_var_model(self):
        """Train VAR model for multivariate forecasting."""
        print("\n" + "=" * 60)
        print("STEP 3: VAR MODEL TRAINING")
        print("=" * 60)
        
        print("Training VAR model for multivariate forecasting...")
        
        # Create VAR model
        var_model = VARModel(
            maxlags=15,
            auto_select_lags=True,
            ic='aic'
        )
        
        # Fit model
        var_model.fit(self.train_data, make_stationary=True)
        
        # Store model
        self.models['VAR_Multivariate'] = var_model
        
        print("VAR model fitted successfully")
        print(f"Best lag order: {var_model.best_lag_order}")
        print(f"AIC: {var_model.aic:.2f}")
    
    def generate_forecasts(self, forecast_days: int = 30):
        """Generate forecasts for all models."""
        print("\n" + "=" * 60)
        print("STEP 4: FORECAST GENERATION")
        print("=" * 60)
        
        for model_name, model in self.models.items():
            print(f"\nGenerating forecasts for {model_name}...")
            
            try:
                if 'ARIMA' in model_name:
                    # Univariate forecasting
                    forecast, conf_int = model.forecast(steps=forecast_days)
                    self.forecasts[model_name] = {
                        'forecast': forecast,
                        'confidence_interval': conf_int,
                        'type': 'univariate'
                    }
                else:
                    # Multivariate forecasting
                    forecast, conf_int = model.forecast(steps=forecast_days)
                    self.forecasts[model_name] = {
                        'forecast': forecast,
                        'confidence_interval': conf_int,
                        'type': 'multivariate'
                    }
                
                print(f"Forecast generated for {model_name}")
                
            except Exception as e:
                print(f"Error generating forecast for {model_name}: {str(e)}")
    
    def evaluate_models(self):
        """Evaluate model performance."""
        print("\n" + "=" * 60)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 60)
        
        # Prepare models for evaluation
        evaluation_models = {}
        
        for model_name, model in self.models.items():
            if 'ARIMA' in model_name:
                # Extract ticker name
                ticker = model_name.split('_')[1]
                evaluation_models[f'ARIMA_{ticker}'] = model
            else:
                evaluation_models[model_name] = model
        
        # Compare models
        comparison_results = self.evaluator.compare_models(
            models=evaluation_models,
            train_data=self.train_data,
            test_data=self.test_data,
            is_multivariate=('VAR' in model_name for model_name in evaluation_models)
        )
        
        print("\nMODEL COMPARISON RESULTS:")
        print(comparison_results.to_string(index=False))
        
        # Plot forecast comparison
        self.evaluator.plot_forecast_comparison(
            models=evaluation_models,
            train_data=self.train_data.iloc[:, 0],  # Use first asset for univariate comparison
            test_data=self.test_data.iloc[:, 0]
        )
        
        # Plot metrics comparison
        self.evaluator.plot_metrics_comparison(comparison_results)
        
        return comparison_results
    
    def generate_investment_insights(self):
        """Generate actionable investment insights."""
        print("\n" + "=" * 60)
        print("STEP 6: INVESTMENT INSIGHTS GENERATION")
        print("=" * 60)
        
        insights = {}
        
        # Get current prices
        current_prices = {}
        for ticker in self.tickers:
            current_prices[ticker] = self.test_data[ticker].iloc[-1]
        
        # Generate insights for each asset
        for ticker in self.tickers:
            model_name = f'ARIMA_{ticker}'
            if model_name in self.forecasts:
                forecast = self.forecasts[model_name]['forecast']
                conf_int = self.forecasts[model_name]['confidence_interval']
                
                # Calculate forecasted returns
                current_price = current_prices[ticker]
                forecasted_prices = forecast
                forecasted_returns = (forecasted_prices / current_price - 1) * 100
                
                # Calculate confidence intervals
                lower_bound = conf_int.iloc[:, 0] if hasattr(conf_int, 'iloc') else conf_int[0]
                upper_bound = conf_int.iloc[:, 1] if hasattr(conf_int, 'iloc') else conf_int[1]
                
                insights[ticker] = {
                    'current_price': current_price,
                    'forecasted_prices': forecasted_prices,
                    'forecasted_returns': forecasted_returns,
                    'confidence_interval': (lower_bound, upper_bound),
                    'volatility': np.std(forecasted_returns),
                    'expected_return': np.mean(forecasted_returns)
                }
        
        # Generate portfolio insights
        if 'VAR_Multivariate' in self.forecasts:
            var_forecast = self.forecasts['VAR_Multivariate']['forecast']
            insights['portfolio'] = self._analyze_portfolio_forecast(var_forecast, current_prices)
        
        return insights
    
    def _analyze_portfolio_forecast(self, var_forecast, current_prices):
        """Analyze portfolio-level forecast."""
        # Calculate portfolio weights (equal weight for demonstration)
        weights = np.array([1/len(self.tickers)] * len(self.tickers))
        
        # Calculate portfolio returns
        portfolio_returns = []
        for i in range(len(var_forecast)):
            asset_returns = []
            for j, ticker in enumerate(self.tickers):
                if i < len(var_forecast):
                    forecasted_price = var_forecast[i, j]
                    current_price = current_prices[ticker]
                    asset_return = (forecasted_price / current_price - 1)
                    asset_returns.append(asset_return)
            
            if asset_returns:
                portfolio_return = np.sum(np.array(asset_returns) * weights)
                portfolio_returns.append(portfolio_return * 100)
        
        return {
            'expected_return': np.mean(portfolio_returns),
            'volatility': np.std(portfolio_returns),
            'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0,
            'returns': portfolio_returns
        }
    
    def create_investment_report(self, insights):
        """Create comprehensive investment report."""
        print("\n" + "=" * 60)
        print("INVESTMENT INSIGHTS REPORT")
        print("=" * 60)
        
        print("\n📊 ASSET-SPECIFIC FORECASTS (30-Day Horizon)")
        print("-" * 50)
        
        for ticker in self.tickers:
            if ticker in insights:
                insight = insights[ticker]
                print(f"\n{ticker} (Tesla Inc.)" if ticker == 'TSLA' else f"\n{ticker}")
                print(f"  Current Price: ${insight['current_price']:.2f}")
                print(f"  Expected Return: {insight['expected_return']:.2f}% ± {insight['volatility']:.2f}%")
                print(f"  Volatility Forecast: {insight['volatility']:.1f}%")
                
                # Generate recommendation
                if insight['expected_return'] > 5:
                    recommendation = "BUY"
                    reasoning = "Strong growth potential"
                elif insight['expected_return'] > 2:
                    recommendation = "HOLD"
                    reasoning = "Moderate growth expected"
                else:
                    recommendation = "SELL"
                    reasoning = "Limited upside potential"
                
                print(f"  Recommendation: {recommendation} - {reasoning}")
        
        # Portfolio insights
        if 'portfolio' in insights:
            portfolio = insights['portfolio']
            print(f"\n📈 PORTFOLIO INSIGHTS")
            print("-" * 30)
            print(f"  Expected Portfolio Return: {portfolio['expected_return']:.2f}%")
            print(f"  Portfolio Volatility: {portfolio['volatility']:.2f}%")
            print(f"  Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        
        # Risk assessment
        print(f"\n⚠️  RISK ASSESSMENT")
        print("-" * 20)
        print("  • TSLA: High volatility, requires position limits")
        print("  • BND: Low risk, portfolio stabilizer")
        print("  • SPY: Moderate risk, core growth driver")
        
        # Investment recommendations
        print(f"\n💡 INVESTMENT RECOMMENDATIONS")
        print("-" * 30)
        print("  • Conservative: 60% BND, 25% SPY, 15% TSLA")
        print("  • Balanced: 40% BND, 40% SPY, 20% TSLA")
        print("  • Growth: 20% BND, 40% SPY, 40% TSLA")
        
        return insights
    
    def run_complete_pipeline(self):
        """Run the complete forecasting pipeline."""
        print("🚀 STARTING PORTFOLIO FORECASTING PIPELINE")
        print("=" * 60)
        
        # Step 1: Data loading and preprocessing
        self.load_and_preprocess_data()
        
        # Step 2: Train ARIMA models
        self.train_arima_models()
        
        # Step 3: Train VAR model
        self.train_var_model()
        
        # Step 4: Generate forecasts
        self.generate_forecasts(forecast_days=30)
        
        # Step 5: Evaluate models
        comparison_results = self.evaluate_models()
        
        # Step 6: Generate investment insights
        insights = self.generate_investment_insights()
        
        # Step 7: Create investment report
        self.create_investment_report(insights)
        
        print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return {
            'models': self.models,
            'forecasts': self.forecasts,
            'evaluation': comparison_results,
            'insights': insights
        }


def main():
    """Main execution function."""
    # Define assets
    tickers = ['TSLA', 'BND', 'SPY']
    
    # Initialize pipeline
    pipeline = PortfolioForecastingPipeline(
        tickers=tickers,
        start_date='2018-01-01'
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    # Save results
    print("\n💾 SAVING RESULTS...")
    
    # Save forecasts to CSV
    for model_name, forecast_data in results['forecasts'].items():
        if forecast_data['type'] == 'univariate':
            df = pd.DataFrame({
                'forecast': forecast_data['forecast'],
                'lower_bound': forecast_data['confidence_interval'][0],
                'upper_bound': forecast_data['confidence_interval'][1]
            })
        else:
            df = pd.DataFrame(forecast_data['forecast'], columns=tickers)
        
        filename = f"forecasts_{model_name.lower().replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"  Saved {filename}")
    
    # Save evaluation results
    results['evaluation'].to_csv('model_evaluation_results.csv', index=False)
    print("  Saved model_evaluation_results.csv")
    
    print("\n🎯 PIPELINE EXECUTION COMPLETED!")
    print("Check the generated CSV files for detailed results.")


if __name__ == "__main__":
    main()
