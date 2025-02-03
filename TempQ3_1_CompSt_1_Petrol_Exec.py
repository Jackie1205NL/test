import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mysql.connector
from pathlib import Path
import json

# Get the absolute path to your project directory
PROJECT_DIR = Path('/Users/shushengye/PycharmProjects/pythonProject/量化策略/Quant3_1_CompleteStrategy/Q3_1_CompSt_1_Petrol')

# Import the modules using full path
import importlib.util

def import_from_file(file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(
        f"module_{Path(file_path).stem}", 
        file_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import your existing code files
trade_module = import_from_file(PROJECT_DIR / 'TempQ3_1_CompSt_1_Petrol_Trade.py')
model_module = import_from_file(PROJECT_DIR / 'TempQ3_1_CompSt_1_Petrol_Comb.py')

# Import specific classes and functions
SignalGenerator = trade_module.SignalGenerator
run_full_analysis = model_module.run_full_analysis
DataPreparation = model_module.DataPreparation
DimensionReductionAnalysis = model_module.DimensionReductionAnalysis
AdvancedTSNAnalysis = model_module.AdvancedTSNAnalysis

from global_config import initialize_globals, EXECUTION_TIMESTAMP

class EnhancedPCAUpdater:
    """
    A class to handle efficient PCA updates using existing parameters.
    This avoids full recalculation when only an update is needed.
    """
    def __init__(self, engine, target_asset):
        self.engine = engine
        self.target_asset = target_asset
        
    def get_latest_parameters(self):
        """Fetch the most recent PCA parameters from the database."""
        query = """
        SELECT variable_group, columns, imputer_mean, scaler_mean, scaler_std,
               n_components, explained_variance_ratio, components, pca_mean
        FROM quant2_3_MuAs_StReg_PcaParams
        WHERE target = %s
        AND timestamp = (
            SELECT MAX(timestamp)
            FROM quant2_3_MuAs_StReg_PcaParams
            WHERE target = %s
        )
        """
        return pd.read_sql(query, self.engine, params=(self.target_asset, self.target_asset))

    def update_pca_with_new_data(self, new_data, existing_params):
        """
        Update PCA transformation using existing parameters and new data.
        This is more efficient than recalculating everything from scratch.
        """
        # Ensure the index is datetime
        if not isinstance(new_data.index, pd.DatetimeIndex):
            new_data.index = pd.to_datetime(new_data.index)
            
        updated_components = {}
        
        for _, row in existing_params.iterrows():
            group_name = row['variable_group']
            
            # Parse stored parameters
            columns = json.loads(row['columns'])
            imputer_mean = np.array(json.loads(row['imputer_mean']))
            scaler_mean = np.array(json.loads(row['scaler_mean']))
            scaler_std = np.array(json.loads(row['scaler_std']))
            components_matrix = np.array(json.loads(row['components']))
            
            # Extract relevant features from new data
            if all(col in new_data.columns for col in columns):
                X = new_data[columns].copy()
                
                # Apply existing preprocessing
                X_imputed = X.fillna(dict(zip(columns, imputer_mean)))
                X_scaled = (X_imputed - scaler_mean) / scaler_std
                
                # Transform using existing components
                X_transformed = np.dot(X_scaled, components_matrix.T)
                
                updated_components[group_name] = pd.DataFrame(
                    X_transformed,
                    index=X.index,
                    columns=[f"{group_name}_PC{i+1}" 
                            for i in range(components_matrix.shape[0])]
                )
            
        return pd.concat(updated_components.values(), axis=1)

class StrategyExecutor:
    """
    Enhanced executor class that coordinates execution of trading signals
    and model updates with improved data sharing and efficiency.
    """
    def __init__(self, target_asset):
        # Initialize global variables first
        initialize_globals(target_asset=target_asset, start_date=None, end_date=None)
        self.target_asset = target_asset
        self.engine = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345678",
            database="StockAnalysis"
        )
        
        # Initialize components
        self.signal_generator = SignalGenerator(target_asset=target_asset)
        self.pca_updater = EnhancedPCAUpdater(self.engine, target_asset)
        
        # Track update timestamps
        self.last_model_rebuild = self._get_last_model_rebuild()
        self.last_metric_update = self._get_last_metric_update()
        
        # Initialize data storage
        self.reduced_data = None
        self.current_regimes = None
        self.execution_timestamp = datetime.now()  # Store as datetime object

    def _get_last_model_rebuild(self):
        """Retrieve the timestamp of the most recent complete model rebuild."""
        query = """
        SELECT MAX(timestamp) FROM quant2_3_MuAs_StReg_PcaParams
        WHERE target = %s
        """
        with self.engine.cursor() as cursor:
            cursor.execute(query, (self.target_asset,))
            return cursor.fetchone()[0]

    def _get_last_metric_update(self):
        """Retrieve the timestamp of the most recent rolling metric update."""
        query = """
        SELECT MAX(timestamp) FROM quant2_3_MuAs_StReg_RollMetrics
        WHERE target = %s
        """
        with self.engine.cursor() as cursor:
            cursor.execute(query, (self.target_asset,))
            return cursor.fetchone()[0]

    def update_rolling_metrics(self, current_date):
        """
        Efficiently update rolling metrics using existing PCA parameters.
        """
        try:
            # Get latest PCA parameters
            existing_params = self.pca_updater.get_latest_parameters()
            
            # Prepare new data using DataPreparation with extended date range
            training_start = pd.to_datetime('2021-01-01')  # Keep original start
            data_prep = DataPreparation(
                start_date=training_start,
                end_date=current_date
            )
            self.data = data_prep.prepare_merged_data()
            
            # Ensure date column is datetime if it exists and index is unique
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data = self.data.set_index('date')
            
            # Ensure index is unique by keeping last value for duplicates
            self.data = self.data[~self.data.index.duplicated(keep='last')]
            
            # Update PCA components efficiently
            self.reduced_data = self.pca_updater.update_pca_with_new_data(
                self.data, existing_params
            )
            
            # Ensure reduced_data has datetime index and is unique
            if not isinstance(self.reduced_data.index, pd.DatetimeIndex):
                self.reduced_data.index = pd.to_datetime(self.reduced_data.index)
            self.reduced_data = self.reduced_data[~self.reduced_data.index.duplicated(keep='last')]
            
            # Create a temporary DimensionReductionAnalysis instance
            temp_analysis = type('TempAnalysis', (), {
                'data': self.data,
                'reduced_data': self.reduced_data
            })()
            
            # Initialize analysis components
            advanced = AdvancedTSNAnalysis(temp_analysis)
            
            # Update rolling metrics for all horizons
            horizons = [
                f'{self.target_asset}_r_3d',
                f'{self.target_asset}_r_5d',
                f'{self.target_asset}_r_10d',
                f'{self.target_asset}_r_20d'
            ]
            
            for horizon in horizons:
                print(f"Updating rolling metrics for {horizon} through {current_date.date()}")
                advanced.analyze_rolling_windows(
                    target_col=horizon,
                    timestamp=self.execution_timestamp,
                    end_date=current_date
                )
                
            self.last_metric_update = self.execution_timestamp
            
        except Exception as e:
            print(f"Error updating rolling metrics: {str(e)}")
            raise

    def _get_lookback_dates(self, current_date):
        """Calculate lookback dates for different data types."""
        return {
            'price_lookback': current_date - timedelta(days=60),
            'economic_lookback': current_date - timedelta(days=365),
            'current': current_date
        }

    def _validate_data_coverage(self, current_date):
        """Validate data coverage for both price and KPI data."""
        dates = self._get_lookback_dates(current_date)
        
        # Check price data coverage
        price_query = """
        WITH date_coverage AS (
            SELECT date, COUNT(DISTINCT ticker) as ticker_count
            FROM quant1_1_daily_prices
            WHERE date BETWEEN %s AND %s
            GROUP BY date
        )
        SELECT COUNT(*) as dates_with_data, MIN(ticker_count) as min_coverage
        FROM date_coverage
        """
        
        # Check KPI data coverage with longer lookback
        kpi_query = """
        SELECT COUNT(DISTINCT date) as dates_with_data
        FROM quant1_1_economic_kpi
        WHERE date BETWEEN %s AND %s
        """
        
        with self.engine.cursor() as cursor:
            # Check price data with 60-day lookback
            cursor.execute(price_query, (dates['price_lookback'], dates['current']))
            price_dates, min_coverage = cursor.fetchone()
            
            # Check KPI data with 365-day lookback
            cursor.execute(kpi_query, (dates['economic_lookback'], dates['current']))
            kpi_dates = cursor.fetchone()[0]
            
            # Calculate expected business days for each period
            expected_price_days = len(pd.date_range(dates['price_lookback'], dates['current'], freq='B'))
            expected_kpi_days = len(pd.date_range(dates['economic_lookback'], dates['current'], freq='B'))
            
            coverage_report = {
                'price_coverage': min_coverage if min_coverage else 0,
                'price_dates': price_dates,
                'kpi_dates': kpi_dates,
                'expected_price_days': expected_price_days,
                'expected_kpi_days': expected_kpi_days,
                'is_valid': (
                    min_coverage is not None 
                    and min_coverage >= 10 
                    and price_dates >= expected_price_days * 0.9
                    and kpi_dates >= expected_kpi_days * 0.8
                )
            }
            
            if not coverage_report['is_valid']:
                print(f"Data coverage warning for {current_date.date()}:")
                print(f"- Price data: {coverage_report['price_coverage']} tickers minimum")
                print(f"- Price dates covered: {coverage_report['price_dates']}/{coverage_report['expected_price_days']}")
                print(f"- KPI dates covered: {coverage_report['kpi_dates']}/{coverage_report['expected_kpi_days']}")
            
            return coverage_report

    def execute_strategy(self, start_date: str, end_date: str):
        # Update global start and end dates
        initialize_globals(self.target_asset, start_date, end_date)
        print(f"Strategy execution started at: {self.execution_timestamp}")
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        simulation_dates = pd.date_range(
            start=start_dt,
            end=end_dt,
            freq='B'  # Business days only
        )

        for current_date in simulation_dates:
            print(f"\nProcessing date: {current_date.date()}")
            
            # Get lookback dates
            dates = self._get_lookback_dates(current_date)
            
            # Validate data coverage with lookback periods
            coverage = self._validate_data_coverage(current_date)
            if not coverage['is_valid']:
                print("Insufficient data coverage. Skipping date.")
                continue

            if self._is_quarter_end(current_date):
                print(f"Quarter-end detected. Initiating complete model rebuild...")
                try:
                    # Run complete modeling pipeline with extended lookback
                    results = run_full_analysis(
                        start_date=dates['economic_lookback'],  # Use full lookback for training
                        end_date=current_date,
                        target_asset=self.target_asset,
                        timestamp=self.execution_timestamp
                    )
                    
                    # Verify and update data
                    if not results or 'reduced_data' not in results:
                        raise ValueError("Model rebuild failed to produce required data")
                    
                    self.reduced_data = results['reduced_data']
                    
                    # Reinitialize with updated model and extended data range
                    self.signal_generator = SignalGenerator(
                        target_asset=self.target_asset,
                        model_timestamp=self.execution_timestamp,
                        lookback_days=60  # Explicitly set lookback for signals
                    )
                    
                    # Force data preparation refresh with extended range
                    data_prep = DataPreparation(
                        start_date=dates['economic_lookback'],
                        end_date=current_date
                    )
                    self.data = data_prep.prepare_merged_data()
                    
                    self.last_model_rebuild = self.execution_timestamp
                    print(f"Model rebuild completed successfully through {current_date.date()}")
                    
                except Exception as e:
                    print(f"Error during model rebuild: {str(e)}")
                    print("Attempting to continue with previous model...")
                    continue

            # Check if we need rolling metric update (month-end)
            elif self._is_month_end(current_date):
                print(f"Month-end detected. Updating rolling metrics...")
                # Calculate the new extended period for metrics
                metrics_end = current_date  # Use current date as end
                self.update_rolling_metrics(metrics_end)
                print(f"Rolling metrics update completed successfully through {metrics_end.date()}")

            # Generate daily trading signal
            print(f"Generating trading signal for {current_date.date()}")
            signal = self.signal_generator.generate_signal(current_date)
            
            if signal:
                self._save_signal_to_database(signal, current_date)
                print(f"Signal saved: {signal['final_signal']}")
            else:
                print("No signal generated for this date")

    def _is_month_end(self, date):
        """
        Check if a date is the last business day of the month.
        This improved version handles cases where month-end falls on weekend.
        
        The logic works by:
        1. Looking forward one business day
        2. If that next business day is in a different month, then current day is last business day
        
        Args:
            date (datetime): The date to check
            
        Returns:
            bool: True if date is the last business day of the month
        """
        # Get the next business day
        next_business_day = date
        while True:
            next_business_day += timedelta(days=1)
            if next_business_day.weekday() < 5:  # 5 is Saturday, 6 is Sunday
                break
                
        # If next business day is in a different month, this is the last business day
        return date.month != next_business_day.month

    def _is_quarter_end(self, date):
        """Check if date is the last business day of the quarter."""
        return self._is_month_end(date) and date.month in [3, 6, 9, 12]

    def _save_signal_to_database(self, signal, current_date):
        """Save generated signal to the database with proper error handling and NaN handling."""
        try:
            insert_query = """
            INSERT INTO quant2_3_MuAs_StReg_TradeSignal 
            (target, timestamp, date, regime, trade_signal, final_signal, 
             3d_signal, 5d_signal, 10d_signal, 20d_signal,
             3d_confidence, 5d_confidence, 10d_confidence, 20d_confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            regime = VALUES(regime),
            trade_signal = VALUES(trade_signal),
            final_signal = VALUES(final_signal),
            3d_signal = VALUES(3d_signal),
            5d_signal = VALUES(5d_signal),
            10d_signal = VALUES(10d_signal),
            20d_signal = VALUES(20d_signal),
            3d_confidence = VALUES(3d_confidence),
            5d_confidence = VALUES(5d_confidence),
            10d_confidence = VALUES(10d_confidence),
            20d_confidence = VALUES(20d_confidence)
            """
            
            # Create execution timestamp if not available
            execution_timestamp = getattr(self, 'execution_timestamp', None)
            if execution_timestamp is None:
                execution_timestamp = datetime.now()
                self.execution_timestamp = execution_timestamp

            # Helper function to safely get signal values
            def safe_get(d, *keys, default=None):
                try:
                    current = d
                    for key in keys:
                        if isinstance(current, dict):
                            current = current.get(key, default)
                        else:
                            return default
                    return None if pd.isna(current) else current
                except (KeyError, TypeError, AttributeError):
                    return default

            # Prepare signal data with NaN handling
            signal_data = (
                self.target_asset,
                self.execution_timestamp,
                current_date.date(),
                safe_get(signal, 'regime', default='unknown'),
                safe_get(signal, 'trade_signal', default=0),
                safe_get(signal, 'final_signal', default=0),
                safe_get(signal, 'horizon_signals', '3d', 'signal_strength', default=0),
                safe_get(signal, 'horizon_signals', '5d', 'signal_strength', default=0),
                safe_get(signal, 'horizon_signals', '10d', 'signal_strength', default=0),
                safe_get(signal, 'horizon_signals', '20d', 'signal_strength', default=0),
                safe_get(signal, 'horizon_signals', '3d', 'confidence', default=0),
                safe_get(signal, 'horizon_signals', '5d', 'confidence', default=0),
                safe_get(signal, 'horizon_signals', '10d', 'confidence', default=0),
                safe_get(signal, 'horizon_signals', '20d', 'confidence', default=0)
            )

            # Validate all values are not None/NaN before executing SQL
            if any(pd.isna(x) for x in signal_data):
                logging.warning("Signal contains NaN values, replacing with defaults")
                signal_data = tuple(0 if pd.isna(x) else x for x in signal_data)
            
            with self.engine.cursor() as cursor:
                cursor.execute(insert_query, signal_data)
            self.engine.commit()
            
        except Exception as e:
            logging.error(f"Error saving signal to database: {str(e)}")
            self.engine.rollback()
            raise

if __name__ == "__main__":
    executor = StrategyExecutor(target_asset='XOM')
    executor.execute_strategy(
        start_date='2024-07-01',
        end_date='2024-09-29'
    )
