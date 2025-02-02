import pandas as pd
import numpy as np
import mysql.connector
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration parameters
TARGET_SYMBOL = 'NCLH'  # Can be changed to other symbols
SIGNAL_TIMESTAMP = '2025-01-21 00:21:05'  # Should match the timestamp in quant2_3_MuAs_StReg_TradeSignal

class FICOStrategySimulator:
    def __init__(self, initial_capital=10000, target=TARGET_SYMBOL, timestamp=SIGNAL_TIMESTAMP):
        """Initialize the strategy simulator with starting capital."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = 0  # Number of shares held
        self.position_value = 0  # Dollar value of position
        self.in_position = False
        self.last_signal = 0
        self.target = target
        self.timestamp = timestamp
        
        # Connect to MySQL database
        self.engine = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345678",
            database="StockAnalysis"
        )
        
        # Initialize storage for results
        self.trades = []
        self.daily_stats = []
        
    def fetch_trading_signals(self):
        """Fetch trading signals from MySQL database."""
        query = """
        SELECT 
            date,
            regime,
            trade_signal,
            final_signal,
            3d_signal as three_d_signal,
            5d_signal as five_d_signal,
            10d_signal as ten_d_signal,
            20d_signal as twenty_d_signal,
            3d_confidence as three_d_confidence,
            5d_confidence as five_d_confidence,
            10d_confidence as ten_d_confidence,
            20d_confidence as twenty_d_confidence
        FROM quant2_3_MuAs_StReg_TradeSignal
        WHERE target = %s
        AND timestamp = %s
        ORDER BY date ASC
        """
        
        return pd.read_sql(query, self.engine, params=(self.target, self.timestamp))
        
    def fetch_price_data(self, start_date, end_date):
        """Fetch FICO price data from MySQL database."""
        query = """
        SELECT date, open_price, close_price, high_price, low_price
        FROM quant1_1_daily_prices
        WHERE ticker = %s
        AND date BETWEEN %s AND %s
        ORDER BY date ASC
        """
        
        return pd.read_sql(query, self.engine, params=(self.target, start_date, end_date))
        
    def calculate_position_size(self, signal_row, current_price):
        """Calculate the position size based on regime and signals."""
        # Base position sizing factors
        regime_factors = {
            'Low_Up': 1.0,
            'Medium_Up': 0.8,
            'High_Up': 0.6,
            'Low_Down': 0.5,
            'High_Down': 0.3
        }
        
        # Get base factor from regime
        base_factor = regime_factors.get(signal_row['regime'], 0.6)  # Default to 0.6 if regime not found
        
        # Check signal alignment
        horizon_signals = [
            signal_row['three_d_signal'],
            signal_row['five_d_signal'],
            signal_row['ten_d_signal'],
            signal_row['twenty_d_signal']
        ]
        
        # Calculate average confidence
        confidences = [
            signal_row['three_d_confidence'],
            signal_row['five_d_confidence'],
            signal_row['ten_d_confidence'],
            signal_row['twenty_d_confidence']
        ]
        avg_confidence = np.mean([c for c in confidences if c is not None])
        
        # Adjust factor based on signal alignment and confidence
        all_signals_align = all(np.sign(s) == np.sign(horizon_signals[0]) 
                              for s in horizon_signals if s is not None)
        if all_signals_align:
            base_factor += 0.1
            
        if avg_confidence > 0.7:
            base_factor += 0.1
            
        # Calculate maximum shares based on available capital
        max_shares = int((self.current_capital * base_factor) / current_price)
        
        return max_shares
        
    def should_exit_position(self, signal_row, current_position_type):
        """Determine if we should exit the current position."""
        if current_position_type == 0:  # No position
            return False
            
        # Exit conditions based on signal changes
        if current_position_type == 1:  # Long position
            if signal_row['final_signal'] == -1:
                return True
                
        elif current_position_type == -1:  # Short position
            if signal_row['final_signal'] == 1:
                return True
                
        # Check confidence levels
        confidences = [
            signal_row['three_d_confidence'],
            signal_row['five_d_confidence'],
            signal_row['ten_d_confidence'],
            signal_row['twenty_d_confidence']
        ]
        avg_confidence = np.mean([c for c in confidences if c is not None])
        
        if avg_confidence < 0.4:
            return True
            
        return False
        
    def execute_trade(self, date, signal_row, price_data, is_entry=True):
        """Execute a trade and record the details."""
        execution_price = price_data['open_price']
        old_position = self.position_size
        
        if is_entry:
            # Calculate new position size
            new_position = self.calculate_position_size(signal_row, execution_price)
            position_type = np.sign(signal_row['final_signal'])
            
            # Update position and capital
            trade_value = new_position * execution_price
            if trade_value <= self.current_capital:
                self.position_size = new_position
                self.position_value = trade_value
                self.current_capital -= trade_value
                self.in_position = True
            
        else:
            # Exit position
            self.current_capital += self.position_size * execution_price
            self.position_size = 0
            self.position_value = 0
            self.in_position = False
            position_type = 0
            
        # Record trade
        self.trades.append({
            'date': date,
            'action': 'entry' if is_entry else 'exit',
            'price': execution_price,
            'position_size': self.position_size - old_position if is_entry else -old_position,
            'position_type': position_type,
            'trade_value': abs((self.position_size - old_position) * execution_price),
            'capital_after_trade': self.current_capital,
            'regime': signal_row['regime']
        })
        
    def run_simulation(self):
        """Run the complete trading simulation."""
        # Fetch signals and price data
        signals_df = self.fetch_trading_signals()
        
        # Convert dates to datetime first
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        
        # Now perform date arithmetic
        start_date = signals_df['date'].min() - timedelta(days=1)
        end_date = signals_df['date'].max() + timedelta(days=1)
        price_df = self.fetch_price_data(start_date, end_date)
        
        # Convert price dates to datetime
        price_df['date'] = pd.to_datetime(price_df['date'])
        
        # Create a complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Initialize tracking variables
        self.current_capital = self.initial_capital
        self.position_size = 0
        self.in_position = False
        
        for i in range(len(signals_df)-1):  # Stop one day early to avoid look-ahead
            current_date = signals_df['date'].iloc[i]
            current_row = signals_df.iloc[i]
            
            # Get next day's price data for execution
            next_date = current_date + timedelta(days=1)
            next_day_price = price_df[price_df['date'] == next_date]
            
            if next_day_price.empty:
                continue
                
            # Record daily statistics before any trades
            portfolio_value = self.current_capital + self.position_value
            position_type = 'none'
            if self.position_size > 0:
                position_type = 'long'
            elif self.position_size < 0:
                position_type = 'short'
                
            self.daily_stats.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'position_value': self.position_value,
                'position_size': self.position_size,
                'position_type': position_type,
                'regime': current_row['regime'],
                'signal': current_row['final_signal']
            })
            
            # Check if we need to exit current position
            if self.in_position and self.should_exit_position(current_row, np.sign(self.position_size)):
                self.execute_trade(next_date, current_row, next_day_price.iloc[0], is_entry=False)
                
            # Check for new entry signals
            elif not self.in_position and current_row['final_signal'] != 0:
                # Verify signal strength and confidence
                horizon_confidences = [
                    current_row['three_d_confidence'],
                    current_row['five_d_confidence'],
                    current_row['ten_d_confidence'],
                    current_row['twenty_d_confidence']
                ]
                avg_confidence = np.mean([c for c in horizon_confidences if c is not None])
                
                if avg_confidence >= 0.6:  # Only enter if confidence is high enough
                    self.execute_trade(next_date, current_row, next_day_price.iloc[0], is_entry=True)
        
        # Create DataFrames from results
        trades_df = pd.DataFrame(self.trades)
        daily_stats_df = pd.DataFrame(self.daily_stats)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join('analysis_output', 'Q2_3_StReg1', 'Simulation')
        os.makedirs(output_dir, exist_ok=True)
        
        # Format timestamp for filename
        timestamp_str = self.timestamp.replace(' ', '_').replace(':', '')
        base_filename = f"{self.target}_{timestamp_str}"
        
        # Save results to CSV
        trades_df.to_csv(os.path.join(output_dir, f'{base_filename}_trades.csv'), index=False)
        daily_stats_df.to_csv(os.path.join(output_dir, f'{base_filename}_daily_stats.csv'), index=False)
        
        return trades_df, daily_stats_df

class MLEnhancedStrategy:
    """
    Machine Learning enhancement for the FICO trading strategy.
    This class works alongside the base FICOStrategySimulator to provide
    ML-driven insights for trading decisions.
    """
    def __init__(self, base_simulator):
        """
        Initialize the ML strategy enhancement.
        
        Parameters:
        base_simulator: FICOStrategySimulator instance to enhance
        """
        self.base_simulator = base_simulator
        self.engine = base_simulator.engine
        
        # Initialize ML models
        self.return_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
        
        self.risk_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def prepare_training_data(self, start_date='2021-01-01'):
        """Prepare historical data for ML model training with NaN handling."""
        query = """
        SELECT 
            dp.date,
            dp.open_price,
            dp.close_price,
            dp.high_price,
            dp.low_price,
            ts.regime,
            ts.final_signal,
            ts.3d_signal as three_d_signal,
            ts.5d_signal as five_d_signal,
            ts.10d_signal as ten_d_signal,
            ts.20d_signal as twenty_d_signal
        FROM quant1_1_daily_prices dp
        LEFT JOIN quant2_3_MuAs_StReg_TradeSignal ts 
        ON dp.date = ts.date AND ts.target = 'FICO'
        WHERE dp.ticker = 'FICO'
        AND dp.date >= %s
        ORDER BY dp.date ASC
        """
        
        df = pd.read_sql(query, self.engine, params=(start_date,))
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate technical indicators
        df = self._add_technical_features(df)
        
        # Handle NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    def _add_technical_features(self, df):
        """Add technical analysis features for ML models."""
        # Price-based features
        df['returns'] = df['close_price'].pct_change()
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        
        # Volatility features
        for window in [5, 10, 20]:
            # Rolling volatility
            df[f'volatility_{window}d'] = df['returns'].rolling(window).std()
            # Rolling mean price
            df[f'ma_{window}d'] = df['close_price'].rolling(window).mean()
            # Price momentum
            df[f'momentum_{window}d'] = df['close_price'].pct_change(window)
        
        # RSI calculation
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving average crossovers
        df['ma_cross'] = df['ma_5d'] / df['ma_20d'] - 1
        
        return df
        
    def train_models(self, training_data, forward_window=5):
        """Train ML models with proper NaN handling."""
        feature_cols = [
            'returns', 'high_low_ratio', 
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'ma_cross', 'rsi',
            'three_d_signal', 'five_d_signal', 'ten_d_signal', 'twenty_d_signal'
        ]
        
        # Create target variables
        training_data['forward_return'] = training_data['close_price'].pct_change(forward_window).shift(-forward_window)
        training_data['forward_risk'] = training_data['volatility_20d'].shift(-1)
        
        # Handle NaN values
        training_data = training_data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with NaN values
        clean_data = training_data.dropna()
        
        # Prepare features and targets
        X = clean_data[feature_cols]
        y_return = clean_data['forward_return']
        y_risk = clean_data['forward_risk']
        
        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(X)
        y_return_scaled = self.target_scaler.fit_transform(y_return.values.reshape(-1, 1))
        
        print(f"Training with {len(X_scaled)} samples after cleaning")
        
        # Train models using time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X_scaled):
            self.return_predictor.fit(X_scaled[train_idx], y_return_scaled[train_idx].ravel())
            self.risk_predictor.fit(X_scaled[train_idx], y_risk.iloc[train_idx])
        
    def predict_returns_and_risk(self, current_data):
        """Generate predictions with NaN handling."""
        feature_cols = [
            'returns', 'high_low_ratio', 
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'ma_cross', 'rsi',
            'three_d_signal', 'five_d_signal', 'ten_d_signal', 'twenty_d_signal'
        ]
        
        # Prepare features and handle NaN values
        X = current_data[feature_cols].iloc[-1:].copy()
        X = X.fillna(method='ffill').fillna(0)  # Fill remaining NaNs with 0
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        try:
            # Generate predictions
            return_pred = self.return_predictor.predict(X_scaled)
            risk_pred = self.risk_predictor.predict(X_scaled)
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(X_scaled[0])
            
            # Unscale return prediction
            return_pred_unscaled = self.target_scaler.inverse_transform(
                return_pred.reshape(-1, 1)
            )[0][0]
            
            return return_pred_unscaled, risk_pred[0], confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return neutral predictions if there's an error
            return 0.0, 0.5, 0.5

    def _calculate_prediction_confidence(self, features):
        """Calculate confidence score based on feature importance and values."""
        # Get feature importances from return predictor
        importance = self.return_predictor.feature_importances_
        
        # Calculate weighted sum of normalized feature values
        confidence = np.sum(np.abs(features) * importance) / np.sum(importance)
        
        # Scale confidence to [0, 1]
        confidence = 1 / (1 + np.exp(-5 * (confidence - 0.5)))
        
        return confidence
        
    def enhance_position_size(self, base_position, predicted_return, predicted_risk, confidence):
        """
        Adjust position size based on ML predictions.
        
        Parameters:
        base_position: Original position size from base strategy
        predicted_return: ML-predicted return
        predicted_risk: ML-predicted risk
        confidence: Prediction confidence score
        
        Returns:
        int: Adjusted position size
        """
        # Scale factor based on predicted return
        return_factor = np.tanh(predicted_return * 5)  # Scale and bound the return impact
        
        # Risk adjustment factor
        risk_factor = 1 / (1 + np.exp(predicted_risk * 5))  # Higher risk reduces position
        
        # Confidence adjustment
        confidence_factor = confidence * 0.5 + 0.5  # Range [0.5, 1.0]
        
        # Combine all factors
        adjustment = (1 + return_factor) * risk_factor * confidence_factor
        
        # Apply adjustment with limits
        adjusted_position = int(base_position * adjustment)
        
        # Ensure position stays within reasonable bounds
        max_position = int(base_position * 1.5)
        min_position = int(base_position * 0.5)
        
        return np.clip(adjusted_position, min_position, max_position)

def run_ml_enhanced_strategy(target=TARGET_SYMBOL, timestamp=SIGNAL_TIMESTAMP):
    """
    Run the ML-enhanced trading strategy and compare results with base strategy.
    """
    # Initialize base strategy
    base_simulator = FICOStrategySimulator(initial_capital=10000, target=target, timestamp=timestamp)
    
    # Initialize ML enhancement
    ml_strategy = MLEnhancedStrategy(base_simulator)
    
    # Prepare and train ML models
    print("Preparing training data...")
    training_data = ml_strategy.prepare_training_data()
    
    print("Training ML models...")
    ml_strategy.train_models(training_data)
    
    # Override position calculation method
    original_calculate_position = base_simulator.calculate_position_size
    
    def enhanced_position_calculation(signal_row, current_price):
        # Get base position from original method
        base_position = original_calculate_position(signal_row, current_price)
        
        # Get current market data
        current_data = ml_strategy.prepare_training_data(
            start_date=(pd.to_datetime(signal_row['date']) - timedelta(days=30)).strftime('%Y-%m-%d')
        )
        
        # Generate ML predictions
        pred_return, pred_risk, confidence = ml_strategy.predict_returns_and_risk(current_data)
        
        # Enhance position size using ML insights
        enhanced_position = ml_strategy.enhance_position_size(
            base_position, pred_return, pred_risk, confidence
        )
        
        return enhanced_position
    
    # Replace position calculation method
    base_simulator.calculate_position_size = enhanced_position_calculation
    
    # Run simulation with enhanced position sizing
    print("Running ML-enhanced simulation...")
    trades_df, daily_stats_df = base_simulator.run_simulation()
    
    # Calculate performance metrics
    daily_returns = daily_stats_df['portfolio_value'].pct_change()
    annual_return = daily_returns.mean() * 252 * 100
    annual_vol = daily_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
    max_drawdown = (daily_stats_df['portfolio_value'] / daily_stats_df['portfolio_value'].cummax() - 1).min() * 100
    
    print("\nML-Enhanced Strategy Performance:")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"Annual Volatility: {annual_vol:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    return trades_df, daily_stats_df

if __name__ == "__main__":
    # You can modify these values when running the script
    trades_df, daily_stats_df = run_ml_enhanced_strategy(
        target=TARGET_SYMBOL,
        timestamp=SIGNAL_TIMESTAMP
    )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('analysis_output', 'Q2_3_StReg1', 'Simulation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Format timestamp for filename
    timestamp_str = SIGNAL_TIMESTAMP.replace(' ', '_').replace(':', '')
    base_filename = f"{TARGET_SYMBOL}_{timestamp_str}"
    
    # Save results to CSV
    trades_df.to_csv(os.path.join(output_dir, f'{base_filename}_trades.csv'), index=False)
    daily_stats_df.to_csv(os.path.join(output_dir, f'{base_filename}_daily_stats.csv'), index=False)