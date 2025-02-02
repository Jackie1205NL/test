import pandas as pd
import numpy as np
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
from datetime import datetime, timedelta
import warnings
import os
import logging
warnings.filterwarnings('ignore')

# Default configuration
DEFAULT_TARGET_ASSET = 'KCUSX'
DEFAULT_START_DATE = '2024-07-01'
DEFAULT_END_DATE = '2024-12-31'
DEFAULT_TIMESTAMP = None  # Will use latest available if None

# Global configuration (can be overridden by environment variables)
TARGET_ASSET = os.getenv('TARGET_ASSET', DEFAULT_TARGET_ASSET)
SIMULATION_START_DATE = os.getenv('SIMULATION_START_DATE', DEFAULT_START_DATE)
SIMULATION_END_DATE = os.getenv('SIMULATION_END_DATE', DEFAULT_END_DATE)
MODEL_TIMESTAMP = os.getenv('MODEL_TIMESTAMP', DEFAULT_TIMESTAMP)

class SignalGenerator:
    def __init__(self, target_asset=None, model_timestamp=None):
        """Initialize the signal generator with database connection and target asset."""
        # Use parameters if provided, otherwise use global config
        self.target_asset = target_asset or TARGET_ASSET
        self.model_timestamp = model_timestamp or MODEL_TIMESTAMP
        self.engine = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345678",
            database="StockAnalysis"
        )
        
        # Load the appropriate model timestamp
        self.model_timestamp = (
            self.model_timestamp if self.model_timestamp 
            else self._get_latest_model_timestamp()
        )
        
        # Initialize PCA models and other components
        self.initialize_pca_models()
        self.load_regime_performance()
        self.load_rolling_metrics()
        
        # Add logging configuration
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_required_features(self):
        """Get the actual list of features used in PCA training."""
        required_features = set()
        for group_info in self.pca_models.values():
            required_features.update(group_info['feature_order'])
        return list(required_features)

    def fetch_price_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch price data for only the tickers we need."""
        features = self.get_required_features()
        tickers = set()
        for feature in features:
            if '_r_' in feature:
                ticker = feature.split('_r_')[0]
                tickers.add(ticker)
        
        tickers_str = "', '".join(tickers)
        
        # Query that gets actual data range in the database
        range_query = """
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM quant1_1_daily_prices 
        WHERE ticker IN ('{tickers_str}')
        """
        
        with self.engine.cursor() as cursor:
            cursor.execute(range_query.format(tickers_str=tickers_str))
            db_min_date, db_max_date = cursor.fetchone()
            
            if db_min_date is None or db_max_date is None:
                print(f"Warning: No data available for the specified tickers")
                return pd.DataFrame()
            
            # Convert to pandas datetime for comparison
            db_min_date = pd.to_datetime(db_min_date)
            db_max_date = pd.to_datetime(db_max_date)
            
            # Ensure requested dates are within available data range
            # For end date, use the latest available if requested is future
            effective_end_date = min(end_date, db_max_date)
            
            # For start date, ensure we have enough lookback for calculations
            # but don't go before available data
            desired_start = effective_end_date - timedelta(days=90)  # Increased buffer
            effective_start_date = max(desired_start, db_min_date)
            
            print(f"Fetching price data from {effective_start_date.date()} to {effective_end_date.date()}")
            
            # Main query to fetch data
            query = f"""
            SELECT date, ticker, close_price
            FROM quant1_1_daily_prices
            WHERE ticker IN ('{tickers_str}')
            AND date BETWEEN %s AND %s
            ORDER BY date, ticker
            """
            
            df_prices = pd.read_sql(
                query, self.engine,
                params=(effective_start_date, effective_end_date)
            )
            
            if df_prices.empty:
                print(f"Warning: No price data available in specified date range")
                return pd.DataFrame()
            
            # Create a complete date range with all business days
            all_dates = pd.date_range(start=effective_start_date, 
                                    end=effective_end_date, 
                                    freq='B')
            
            # For each ticker, ensure we have all dates by forward filling
            result_dfs = []
            for ticker in tickers:
                ticker_data = df_prices[df_prices['ticker'] == ticker].copy()
                if not ticker_data.empty:
                    date_df = pd.DataFrame(index=all_dates)
                    ticker_data.set_index('date', inplace=True)
                    merged = date_df.join(ticker_data)
                    merged['ticker'] = ticker
                    merged['close_price'] = merged['close_price'].ffill()
                    result_dfs.append(merged.reset_index().rename(columns={'index': 'date'}))
            
            if result_dfs:
                final_df = pd.concat(result_dfs, ignore_index=True)
                return final_df
            
            return pd.DataFrame()

    def fetch_economic_data(self, current_date: datetime) -> pd.DataFrame:
        """Fetch and process economic indicators with improved date handling."""
        # Query to get the actual data range
        range_query = """
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM quant1_1_economic_kpi
        WHERE status = 'align_freq'
        AND symbol IN ('GDP', 'CPI', 'unemploymentRate', 'federalFunds')
        """
        
        with self.engine.cursor() as cursor:
            cursor.execute(range_query)
            db_min_date, db_max_date = cursor.fetchone()
            
            if db_min_date is None or db_max_date is None:
                return pd.DataFrame()
                
            # Convert to pandas datetime for comparison
            db_min_date = pd.to_datetime(db_min_date)
            db_max_date = pd.to_datetime(db_max_date)
            
            # Ensure we're using data that exists
            effective_end_date = min(current_date, db_max_date)
            # Need enough lookback for 12-month changes
            desired_start = effective_end_date - timedelta(days=365)
            effective_start_date = max(desired_start, db_min_date)
            
            print(f"Fetching economic data from {effective_start_date.date()} to {effective_end_date.date()}")
        
        query = """
        SELECT symbol, date, value
        FROM quant1_1_economic_kpi
        WHERE date BETWEEN %s AND %s
        AND status = 'align_freq'
        AND symbol IN ('GDP', 'CPI', 'unemploymentRate', 'federalFunds')
        ORDER BY date, symbol
        """
        
        df_kpi = pd.read_sql(
            query, self.engine,
            params=(effective_start_date, effective_end_date)
        )
        
        if df_kpi.empty:
            return pd.DataFrame()
        
        # Process the KPIs
        df_kpi['date'] = pd.to_datetime(df_kpi['date'])
        df_pivot = df_kpi.pivot(index='date', columns='symbol', values='value')
        
        # Calculate 12-month changes with proper handling of missing data
        changes = {}
        for col in df_pivot.columns:
            shifted = df_pivot[col].shift(12)
            mask = ~shifted.isna()  # Only calculate where we have 12-month lookback
            changes[f'{col}_change'] = pd.Series(index=df_pivot.index)
            changes[f'{col}_change'][mask] = (
                (df_pivot[col][mask] - shifted[mask]) / shifted[mask] * 100
            )
        
        result = pd.DataFrame(changes, index=df_pivot.index)
        result.fillna(method='ffill', inplace=True)  # Forward fill any gaps
        return result

    def _get_latest_model_timestamp(self) -> datetime:
        """Get the most recent model timestamp from PCA parameters table."""
        query = """
        SELECT MAX(timestamp) as latest_timestamp 
        FROM quant2_3_MuAs_StReg_PcaParams 
        WHERE target = %s
        """
        with self.engine.cursor() as cursor:
            cursor.execute(query, (self.target_asset,))
            result = cursor.fetchone()
            if not result or not result[0]:
                raise ValueError("No model timestamps found in database")
            return result[0]

    def initialize_pca_models(self):
        """Load PCA models and preprocessing parameters from the database."""
        query = """
        SELECT variable_group, columns, imputer_mean, scaler_mean, scaler_std,
               n_components, explained_variance_ratio, components, pca_mean
        FROM quant2_3_MuAs_StReg_PcaParams
        WHERE target = %s AND timestamp = %s
        """
        
        df_pca = pd.read_sql(query, self.engine, params=(self.target_asset, self.model_timestamp))
        
        # Initialize PCA models dictionary
        self.pca_models = {}
        
        for _, row in df_pca.iterrows():
            group_name = row['variable_group']
            
            # Parse JSON strings into Python objects
            columns = json.loads(row['columns'])
            imputer_mean = json.loads(row['imputer_mean'])
            scaler_mean = json.loads(row['scaler_mean'])
            scaler_std = json.loads(row['scaler_std'])
            components_matrix = json.loads(row['components'])
            explained_variance = json.loads(row['explained_variance_ratio'])
            pca_mean = json.loads(row['pca_mean']) if row['pca_mean'] else None
            
            # Store all parameters for this group
            self.pca_models[group_name] = {
                'feature_order': columns,
                'preprocessing': {
                    'imputer_mean': np.array(imputer_mean),
                    'scaler_mean': np.array(scaler_mean),
                    'scaler_std': np.array(scaler_std)
                },
                'pca': {
                    'n_components': row['n_components'],
                    'components': np.array(components_matrix),
                    'explained_variance_ratio': np.array(explained_variance),
                    'mean': np.array(pca_mean) if pca_mean else None
                }
            }

    def load_regime_performance(self):
        """Load regime performance metrics from database."""
        query = """
        SELECT horizon, regime, r2, mse, importance_features
        FROM quant2_3_MuAs_StReg_RegimePerf
        WHERE target = %s AND timestamp = %s
        """
        
        self.regime_performance = pd.read_sql(
            query, self.engine, 
            params=(self.target_asset, self.model_timestamp)
        ).set_index(['horizon', 'regime'])

    def load_rolling_metrics(self):
        """Load rolling window metrics from database."""
        query = """
        SELECT horizon, window_start, window_end, r2_score, mse,
               top_features, feature_coefficients, avg_target_volatility
        FROM quant2_3_MuAs_StReg_RollMetrics
        WHERE target = %s AND timestamp = %s
        """
        
        self.rolling_metrics = pd.read_sql(
            query, self.engine,
            params=(self.target_asset, self.model_timestamp)
        )

    def calculate_regimes(self, current_date: datetime) -> dict:
        """Calculate current market regimes based on recent data."""
        window = 20
        lookback = window * 2  # Double the window size to ensure enough data
        
        # First, find the latest available date in the database
        query = """
        SELECT MAX(date) as last_date
        FROM quant1_1_daily_prices 
        WHERE ticker = %s
        """
        
        with self.engine.cursor() as cursor:
            cursor.execute(query, (self.target_asset,))
            latest_available = cursor.fetchone()[0]
            
            if latest_available is None:
                self.logger.warning(f"No data available for {self.target_asset}")
                return self._get_default_regime()
                
            latest_available = pd.to_datetime(latest_available)
            
            # If current_date is beyond our data, use the latest available
            effective_date = min(current_date, latest_available)
            start_date = effective_date - timedelta(days=lookback)
            
            # Log the date adjustment if any
            if effective_date != current_date:
                self.logger.info(f"Adjusting analysis date from {current_date.date()} to {effective_date.date()}")
        
        # Fetch price data
        query = """
        SELECT date, close_price 
        FROM quant1_1_daily_prices 
        WHERE ticker = %s AND date BETWEEN %s AND %s
        ORDER BY date
        """
        
        df_prices = pd.read_sql(
            query, self.engine,
            params=(self.target_asset, start_date, effective_date)
        )
        
        if df_prices.empty:
            self.logger.warning("No price data available for regime calculation")
            return self._get_default_regime()
        
        # Calculate returns
        df_prices['returns'] = df_prices['close_price'].pct_change()
        
        # Ensure we have enough data for reliable regime calculation
        if len(df_prices) < window:
            self.logger.warning(f"Insufficient data for regime calculation. Required: {window}, Got: {len(df_prices)}")
            return self._get_default_regime()
        
        # Calculate regimes
        try:
            # Volatility regime
            rolling_vol = df_prices['returns'].rolling(window).std()
            vol_quantiles = rolling_vol.quantile([0.33, 0.67])
            last_vol = rolling_vol.iloc[-1]
            
            if pd.isna(last_vol):
                volatility_regime = 'Medium'  # Default if can't calculate
            elif last_vol <= vol_quantiles.iloc[0]:
                volatility_regime = 'Low'
            elif last_vol <= vol_quantiles.iloc[1]:
                volatility_regime = 'Medium'
            else:
                volatility_regime = 'High'
            
            # Trend regime
            rolling_mean = df_prices['returns'].rolling(window).mean()
            last_mean = rolling_mean.iloc[-1]
            
            if pd.isna(last_mean):
                trend_regime = 'Up'  # Default if can't calculate
            else:
                trend_regime = 'Up' if last_mean > 0 else 'Down'
                
            # Default to medium correlation if sector data unavailable
            correlation_regime = 'Medium'
            
            # Attempt to get sector correlation if possible
            sector_query = """
            SELECT date, close_price 
            FROM quant1_1_daily_prices 
            WHERE ticker = 'PAUSD' AND date BETWEEN %s AND %s
            ORDER BY date
            """
            
            df_sector = pd.read_sql(
                sector_query, self.engine,
                params=(start_date, effective_date)
            )
            
            if not df_sector.empty and len(df_sector) >= window:
                df_sector['returns'] = df_sector['close_price'].pct_change()
                df_merged = pd.merge(
                    df_prices['returns'],
                    df_sector['returns'],
                    left_index=True,
                    right_index=True,
                    suffixes=('_target', '_sector')
                )
                
                if not df_merged.empty:
                    corr = df_merged['returns_target'].corr(df_merged['returns_sector'])
                    if not pd.isna(corr):
                        if corr <= -0.33:
                            correlation_regime = 'Low'
                        elif corr <= 0.33:
                            correlation_regime = 'Medium'
                        else:
                            correlation_regime = 'High'
            
            return {
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'correlation_regime': correlation_regime,
                'effective_date': effective_date.date(),
                'data_points': len(df_prices)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating regimes: {str(e)}")
            return self._get_default_regime()

    def _get_default_regime(self) -> dict:
        """Return default regime settings when calculation isn't possible."""
        return {
            'volatility_regime': 'Medium',
            'trend_regime': 'Up',
            'correlation_regime': 'Medium',
            'effective_date': None,
            'data_points': 0
        }

    def calculate_pca_features(self, current_returns: pd.DataFrame) -> pd.DataFrame:
        """Transform current market data into PCA components using stored parameters."""
        pca_features = {}
        
        for group_name, group_info in self.pca_models.items():
            # Get required features for this group
            required_features = group_info['feature_order']
            
            # Check available features and ensure correct ordering
            available_features = [f for f in required_features if f in current_returns.columns]
            feature_indices = [required_features.index(f) for f in available_features]
            missing_ratio = 1 - (len(available_features) / len(required_features))
            
            if len(available_features) > 0 and missing_ratio < 0.5:  # At least 50% features available
                # Extract and order features
                X = current_returns[available_features].copy()
                
                # Apply preprocessing
                preproc = group_info['preprocessing']
                
                # Get the corresponding imputer means and scaling parameters for available features
                imputer_means = preproc['imputer_mean'][feature_indices]
                scaler_means = preproc['scaler_mean'][feature_indices]
                scaler_stds = preproc['scaler_std'][feature_indices]
                
                # Impute missing values
                X_imputed = X.fillna({feat: mean for feat, mean in zip(available_features, imputer_means)})
                
                # Standardize
                X_scaled = pd.DataFrame(
                    (X_imputed - scaler_means) / scaler_stds,
                    columns=available_features,
                    index=X.index
                )
                
                # Get latest values
                latest_values = X_scaled.iloc[-1].values
                
                # Apply PCA transformation
                pca_info = group_info['pca']
                components = pca_info['components']
                
                # Calculate PCA scores using available features only
                for i in range(pca_info['n_components']):
                    component_name = f"{group_name}_PC{i+1}"
                    component_weights = components[i, feature_indices]
                    
                    # Adjust for missing features
                    component_value = np.dot(latest_values, component_weights)
                    component_value *= (1 / (1 - missing_ratio))  # Scale up to account for missing features
                    
                    pca_features[component_name] = component_value
                    
                # print(f"Successfully calculated PCA components for {group_name}")
            else:
                print(f"Warning: Insufficient features for {group_name}")
                # Initialize components to 0
                for i in range(group_info['pca']['n_components']):
                    pca_features[f"{group_name}_PC{i+1}"] = 0
        
        result = pd.DataFrame([pca_features])
        
        # Debug information
        print("\nPCA Components Summary:")
        print(f"Total components calculated: {len(pca_features)}")
        non_zero = sum(1 for v in pca_features.values() if abs(v) > 0)
        print(f"Non-zero components: {non_zero}")
        
        return result

    def get_model_confidence(self, regime: str, horizon: str) -> float:
        """Calculate model confidence based on regime performance."""
        try:
            performance = self.regime_performance.loc[(horizon, regime)]
            confidence = performance['r2'] * (1 / np.sqrt(performance['mse']))
            return max(0, min(1, confidence))  # Normalize to [0,1]
        except KeyError:
            return 0.0

    # def validate_feature_coverage(self, features):
    #     """
    #     Validate feature coverage and return True if sufficient features are available.
    #     """
    #     required_groups = ['currencies', 'macro', 'sector_financial', 'commodity_precious_metals']
    #     available_features = set(features.columns)
        
    #     coverage = {}
    #     for group in required_groups:
    #         group_features = [col for col in available_features if col.startswith(f"{group}_")]
    #         coverage[group] = len(group_features)
            
    #     # Log coverage details
    #     self.logger.info("Feature coverage:")
    #     for group, count in coverage.items():
    #         self.logger.info(f"{group}: {count} features")
            
    #     # Define minimum requirements
    #     min_requirements = {
    #         'currencies': 2,
    #         'macro': 2,
    #         'sector_financial': 1,
    #         'commodity_precious_metals': 1
    #     }
        
    #     # Check if requirements are met
    #     requirements_met = all(
    #         coverage.get(group, 0) >= min_count 
    #         for group, min_count in min_requirements.items()
    #     )
        
    #     if not requirements_met:
    #         self.logger.warning("Insufficient feature coverage for reliable signal generation")
            
    #     return requirements_met

    def fetch_and_prepare_data(self, current_date: datetime) -> pd.DataFrame:
        """
        Fetch and prepare all required data for signal generation.
        This method combines price data fetching and return calculations.
        """
        lookback_days = 60
        start_date = current_date - timedelta(days=lookback_days)
        
        # Get required features and tickers
        features = self.get_required_features()
        tickers = set()
        for feature in features:
            if '_r_' in feature:
                ticker = feature.split('_r_')[0]
                tickers.add(ticker)
        
        tickers_str = "', '".join(tickers)
        
        # Query that gets actual data range and verifies data availability
        range_query = """
        SELECT MIN(date) as min_date, MAX(date) as max_date,
               COUNT(DISTINCT ticker) as ticker_count
        FROM quant1_1_daily_prices 
        WHERE ticker IN ('{tickers_str}')
        AND date BETWEEN %s AND %s
        """
        
        with self.engine.cursor() as cursor:
            cursor.execute(range_query.format(tickers_str=tickers_str), 
                          (start_date, current_date))
            result = cursor.fetchone()
            db_min_date, db_max_date, ticker_count = result
            
            if not ticker_count or ticker_count < len(tickers) * 0.5:  # Require at least 50% ticker coverage
                self.logger.warning(f"Insufficient ticker coverage: {ticker_count}/{len(tickers)}")
                return pd.DataFrame()
            
            # Get the price data
            price_query = f"""
            SELECT date, ticker, close_price
            FROM quant1_1_daily_prices
            WHERE ticker IN ('{tickers_str}')
            AND date BETWEEN %s AND %s
            ORDER BY ticker, date
            """
            
            df_prices = pd.read_sql(price_query, self.engine, 
                                  params=(start_date, current_date))
            
            if df_prices.empty:
                return pd.DataFrame()
            
            # Convert to datetime
            df_prices['date'] = pd.to_datetime(df_prices['date'])
            
            # Create return data for each window size
            returns_data = {}
            windows = [3, 5, 10, 20]  # Return windows
            
            for ticker in tickers:
                ticker_data = df_prices[df_prices['ticker'] == ticker].copy()
                if not ticker_data.empty:
                    ticker_data.sort_values('date', inplace=True)
                    ticker_data.set_index('date', inplace=True)
                    
                    # Calculate returns for each window
                    for w in windows:
                        col_name = f'{ticker}_r_{w}d'
                        returns = ticker_data['close_price'].pct_change(w) * 100
                        returns_data[col_name] = returns
            
            if not returns_data:
                return pd.DataFrame()
            
            # Combine all return series
            returns_df = pd.DataFrame(returns_data)
            
            # Add economic data
            economic_data = self.fetch_economic_data(current_date)
            if not economic_data.empty:
                returns_df = pd.merge(
                    returns_df,
                    economic_data,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
            
            # Forward fill missing values
            returns_df = returns_df.fillna(method='ffill')
            
            # Ensure we have the current date
            if current_date not in returns_df.index:
                self.logger.warning(f"Current date {current_date} not in prepared data")
                return pd.DataFrame()
            
            return returns_df

    def generate_signal(self, current_date: datetime) -> dict:
        """Generate trading signals with improved data handling and regime awareness."""
        try:
            # Get current regime with improved handling
            current_regime_info = self.calculate_regimes(current_date)
            current_regime = f"{current_regime_info['volatility_regime']}_{current_regime_info['trend_regime']}"
            
            # Log regime information
            self.logger.info(f"Current regime for {current_date.date()}: {current_regime}")
            self.logger.info(f"Using data through: {current_regime_info['effective_date']}")
            
            # If we don't have enough data points for regime calculation, return default signal
            if current_regime_info['data_points'] < 20:  # Minimum required for reliable calculation
                self.logger.warning("Insufficient data points for reliable signal generation")
                return self.create_default_signal(current_date, 'insufficient_data')
            
            # Use the effective date from regime calculation for data fetching
            effective_date = pd.to_datetime(current_regime_info['effective_date'])
            
            # Fetch and prepare features
            all_features = self.fetch_and_prepare_data(current_date)
            if all_features.empty:
                return self.create_default_signal(current_date, 'no_features_available')
            
            # Calculate PCA features
            pca_features = self.calculate_pca_features(all_features)
            if pca_features.empty:
                return self.create_default_signal(current_date, 'pca_calculation_failed')
                
            # Generate signals for each horizon
            signals = {}
            for horizon in ['3d', '5d', '10d', '20d']:
                horizon_key = f'{self.target_asset}_r_{horizon}'
                
                try:
                    # Get recent metrics for this horizon
                    horizon_metrics = self.rolling_metrics[
                        self.rolling_metrics['horizon'] == horizon_key
                    ]
                    
                    if horizon_metrics.empty:
                        self.logger.warning(f"No rolling metrics found for {horizon_key}")
                        continue
                    
                    recent_metrics = horizon_metrics.iloc[-1]
                    
                    # Calculate signal components
                    confidence = self.get_model_confidence(current_regime, horizon_key)
                    signal_strength = self._calculate_signal_strength(
                        pca_features,
                        json.loads(recent_metrics['top_features']),
                        json.loads(recent_metrics['feature_coefficients'])
                    )
                    
                    signals[horizon] = {
                        'signal_strength': signal_strength,
                        'confidence': confidence,
                        'regime': current_regime,
                        'weighted_signal': signal_strength * confidence if not pd.isna(signal_strength) else 0,
                        'pca_features_used': pca_features.columns.tolist()
                    }
                except Exception as e:
                    self.logger.error(f"Error calculating signal for {horizon}: {str(e)}")
                    signals[horizon] = {
                        'signal_strength': 0,
                        'confidence': 0,
                        'regime': current_regime,
                        'weighted_signal': 0,
                        'pca_features_used': []
                    }
            
            if not signals:
                return self.create_default_signal(current_date, 'no_signals_generated')
            
            # Calculate final signal from horizon signals
            final_signal = self._combine_horizon_signals(signals)
            
            # Create final signal dictionary
            signal_dict = {
                'date': current_date,
                'final_signal': final_signal,
                'horizon_signals': signals,
                'regime': current_regime,
                'regime_details': current_regime_info,
                'confidence_metrics': {
                    horizon: sig['confidence'] for horizon, sig in signals.items()
                },
                'effective_date': effective_date,
                'debug_info': {
                    'available_features': all_features.columns.tolist(),
                    'pca_groups_used': list(self.pca_models.keys()),
                    'data_quality': {
                        'missing_rate': all_features.isnull().mean().to_dict(),
                        'feature_counts': len(all_features.columns)
                    }
                }
            }
            
            return signal_dict
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {current_date}: {str(e)}")
            return self.create_default_signal(current_date, str(e))

    def create_default_signal(self, current_date, reason='unknown'):
        """Create a default signal with proper documentation."""
        signal_dict = {
            'date': current_date,
            'regime': 'Medium_Up',  # Default conservative regime
            'final_signal': 0,  # Hold
            'horizon_signals': {
                horizon: {
                    'signal_strength': 0,
                    'confidence': 0,
                    'regime': 'Medium_Up',
                    'weighted_signal': 0,
                    'pca_features_used': []
                } for horizon in ['3d', '5d', '10d', '20d']
            },
            'confidence_metrics': {
                horizon: 0 for horizon in ['3d', '5d', '10d', '20d']
            },
            'debug_info': {
                'reason': reason,
                'timestamp': datetime.now()
            }
        }
        
        self.logger.info(f"Generated default signal for {current_date.date()} due to: {reason}")
        return signal_dict

    def _calculate_signal_strength(self, features, important_features, feature_coefficients):
        """
        Calculate signal strength with robust handling of missing features.
        
        This function implements a weighted approach that:
        1. Uses available features with their proper coefficients
        2. Adjusts the final signal based on feature availability
        3. Provides detailed logging of feature usage
        """
        signal = 0
        total_importance = sum(abs(coef) for coef in feature_coefficients)
        used_importance = 0
        used_features = []
        
        for feat, coef in zip(important_features, feature_coefficients):
            if feat in features.columns:
                feat_value = features[feat].iloc[0]
                if not np.isnan(feat_value):  # Check for NaN values
                    signal += feat_value * coef
                    used_importance += abs(coef)
                    used_features.append(feat)
                else:
                    print(f"Warning: NaN value for feature: {feat}")
            else:
                print(f"Warning: Missing important feature: {feat}")
        
        # Calculate feature usage ratio
        feature_usage_ratio = used_importance / total_importance if total_importance > 0 else 0
        print(f"Used {len(used_features)} out of {len(important_features)} features "
            f"(importance ratio: {feature_usage_ratio:.2f})")
        
        # Only return signal if we have sufficient feature coverage
        if feature_usage_ratio >= 0.5:  # At least 50% of important features by weight
            return np.tanh(signal * (1 / feature_usage_ratio))  # Adjust signal strength
        else:
            print("Warning: Insufficient feature coverage for reliable signal")
            return np.nan

    def _combine_horizon_signals(self, signals: dict) -> int:
        """Combine signals from different horizons into final trading decision."""
        # Calculate weighted average signal
        weighted_sum = sum(s['weighted_signal'] * (i + 1) 
                         for i, s in enumerate(signals.values()))
        total_weights = sum(i + 1 for i in range(len(signals)))
        avg_signal = weighted_sum / total_weights
        
        # Convert to trading decision
        if avg_signal > 0.2:
            return 1  # Buy
        elif avg_signal < -0.2:
            return -1  # Sell
        return 0  # Hold

    def run_november_simulation(self, start_date=None, end_date=None):
        """Generate signals for each trading day and save to MySQL database."""
        # Use parameters if provided, otherwise use global config
        simulation_start = pd.to_datetime(start_date or SIMULATION_START_DATE)
        simulation_end = pd.to_datetime(end_date or SIMULATION_END_DATE)
        
        simulation_dates = pd.date_range(
            start=simulation_start,
            end=simulation_end,
            freq='B'  # Business days only
        )
        
        signals = []
        current_timestamp = datetime.now()

        # Create database connection for writing
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
        
        for date in simulation_dates:
            signal = self.generate_signal(date)
            if signal:
                # Calculate trade_signal based on confidence and signal strength
                trade_signal = 0  # Default to no trade
                for horizon in ['3d', '5d', '10d', '20d']:
                    horizon_data = signal['horizon_signals'][horizon]
                    if (abs(horizon_data['confidence']) > 0.8 and 
                        abs(horizon_data['signal_strength']) > 0.8):
                        trade_signal = signal['final_signal']
                        break  # Use first horizon that meets criteria

                # Replace NaN values with None for MySQL
                def replace_nan(value):
                    return None if pd.isna(value) or np.isnan(value) else float(value)

                # Prepare data for database insertion with NaN handling
                signal_data = (
                    self.target_asset,
                    current_timestamp,
                    date.date(),
                    signal['regime'],
                    replace_nan(trade_signal),
                    replace_nan(signal['final_signal']),
                    replace_nan(signal['horizon_signals']['3d']['signal_strength']),
                    replace_nan(signal['horizon_signals']['5d']['signal_strength']),
                    replace_nan(signal['horizon_signals']['10d']['signal_strength']),
                    replace_nan(signal['horizon_signals']['20d']['signal_strength']),
                    replace_nan(signal['horizon_signals']['3d']['confidence']),
                    replace_nan(signal['horizon_signals']['5d']['confidence']),
                    replace_nan(signal['horizon_signals']['10d']['confidence']),
                    replace_nan(signal['horizon_signals']['20d']['confidence'])
                )
                
                try:
                    # Execute database insertion
                    with self.engine.cursor() as cursor:
                        cursor.execute(insert_query, signal_data)
                    signals.append(signal_data)
                except Exception as e:
                    print(f"Error inserting data for date {date}: {str(e)}")
                    continue
            
            # Print daily signal summary
            print(f"\nDate: {date.date()}")
            if signal:
                print(f"Regime: {signal['regime']}")
                print("Horizon Signals:")
                for horizon, details in signal['horizon_signals'].items():
                    print(f"  {horizon}: {details['signal_strength']:.2f} "
                          f"(confidence: {details['confidence']:.2f})")
                print(f"Final Signal: {signal['final_signal']}")
            else:
                print("No signal generated")
        
        # Commit all changes to database
        self.engine.commit()
        print(f"\nResults saved to database for {self.target_asset} at {current_timestamp}")
        
        return signals

if __name__ == "__main__":
    # Example of how to override parameters
    custom_target = os.getenv('CUSTOM_TARGET')
    custom_start = os.getenv('CUSTOM_START_DATE')
    custom_end = os.getenv('CUSTOM_END_DATE')
    custom_timestamp = os.getenv('CUSTOM_TIMESTAMP')
    
    generator = SignalGenerator(
        target_asset=custom_target,
        model_timestamp=custom_timestamp
    )
    signals = generator.run_november_simulation(
        start_date=custom_start,
        end_date=custom_end
    )