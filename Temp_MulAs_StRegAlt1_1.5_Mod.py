import pandas as pd
import numpy as np
import mysql.connector
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import warnings
import os
from datetime import datetime
import json
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add timestamp to filenames
def get_timestamp():
    """Return timestamp in both string and datetime formats."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S"), now

# Define a single global timestamp variable
ANALYSIS_TIMESTAMP, _ = get_timestamp()

# Define global start and end dates
START_DATE = pd.to_datetime('2021-01-01')
END_DATE = pd.to_datetime('2023-12-31')

# Define the target asset here so it's easy to change in the future
TARGET_ASSET = 'FICO'


def create_mysql_connection():
    """Create MySQL connection using provided credentials."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345678",
        database="StockAnalysis"
    )

def read_table_as_df(table_name: str, engine) -> pd.DataFrame:
    """Read MySQL table into pandas DataFrame."""
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, engine)

class DataPreparation:
    def __init__(self, start_date=None, end_date=None):
        """Initialize data preparation class with MySQL connection and date range."""
        self.engine = create_mysql_connection()
        # Use provided dates or global defaults
        self.start_date = pd.to_datetime(start_date) if start_date else START_DATE
        self.end_date = pd.to_datetime(end_date) if end_date else END_DATE
        
    def fetch_and_prepare_price_data(self) -> pd.DataFrame:
        """
        1) Read price data from quant1_1_daily_prices table.
        2) Filter for data from 2020 to Oct 2024 (training range).
        3) Return as a DataFrame.
        """
        # Read table
        df_prices = read_table_as_df('quant1_1_daily_prices', self.engine)

        # Convert 'date' to datetime
        df_prices['date'] = pd.to_datetime(df_prices['date'])

        # Add 60-day buffer to start date
        buffer_start = self.start_date - pd.Timedelta(days=60)

        # Filter date range with buffer
        df_prices = df_prices[(df_prices['date'] >= buffer_start) & 
                            (df_prices['date'] <= self.end_date)]

        # For safety, sort by ticker/date
        df_prices.sort_values(by=['ticker','date'], inplace=True)

        return df_prices

    def compute_multi_day_returns(self, df_prices: pd.DataFrame, 
                                windows: List[int]=[3, 5, 10, 20]) -> pd.DataFrame:
        """
        For each ticker, compute percentage changes over specified time windows.
        Example:
          r_3d = (Price(t) - Price(t-3)) / Price(t-3) * 100%
        """
        if 'adjusted_close_price' not in df_prices.columns:
            df_prices['adjusted_close_price'] = df_prices['close_price']

        results = []
        for ticker, group in df_prices.groupby('ticker', sort=False):
            g_sorted = group.sort_values('date').copy()
            for w in windows:
                col_name = f'r_{w}d'
                g_sorted[col_name] = g_sorted['adjusted_close_price'].pct_change(w) * 100
            results.append(g_sorted)

        df_returns = pd.concat(results, axis=0)
        df_returns.reset_index(drop=True, inplace=True)

        return df_returns

    def remove_outliers(self, df_returns: pd.DataFrame, 
                       windows: List[int]=[3, 5, 10, 20], 
                       threshold: float=15.0) -> pd.DataFrame:
        """
        Remove rows where the absolute return for any given window exceeds the threshold.
        """
        for w in windows:
            col_name = f'r_{w}d'
            df_returns = df_returns[df_returns[col_name].abs() <= threshold]
        df_returns.reset_index(drop=True, inplace=True)
        return df_returns

    def fetch_and_prepare_economic_kpi(self) -> pd.DataFrame:
        """
        Prepare economic KPI data with 12-month changes and daily propagation.
        """
        df_kpi = read_table_as_df('quant1_1_economic_kpi', self.engine)
        df_kpi['date'] = pd.to_datetime(df_kpi['date'])

        # Filter for align_freq
        df_kpi = df_kpi[df_kpi['status'] == 'align_freq'].copy()

        # List of KPI symbols we care about
        kpi_list = ['GDP', 'CPI', 'unemploymentRate', 'federalFunds']
        
        # Pivot to have columns = symbol, rows = date
        df_pivot = df_kpi[df_kpi['symbol'].isin(kpi_list)]
        df_pivot = df_pivot.pivot_table(index='date', columns='symbol', values='value')
        df_pivot.sort_index(inplace=True)

        # Compute 12-month changes
        for k in kpi_list:
            if k in df_pivot.columns:
                col_change = f'{k}_change'
                df_pivot[col_change] = (df_pivot[k] - df_pivot[k].shift(12)) / \
                                     df_pivot[k].shift(12) * 100
        
        # Keep only the _change columns
        change_cols = [col for col in df_pivot.columns if col.endswith('_change')]
        df_kpi_final = df_pivot[change_cols].copy()

        # Daily propagation
        all_days = pd.date_range(self.start_date, self.end_date, freq='D')
        df_kpi_final = df_kpi_final.reindex(all_days)
        df_kpi_final.fillna(method='ffill', inplace=True)

        return df_kpi_final.reset_index().rename(columns={'index': 'date'})

    def merge_tsn_with_independents(self, df_returns: pd.DataFrame, 
                                  df_kpi: pd.DataFrame) -> pd.DataFrame:
        """
        Merge TSN returns with independent variables and economic KPIs.
        """
        # Split TSN and independent tickers
        df_tsn = df_returns[df_returns['ticker'] == TARGET_ASSET].copy()
        df_indep = df_returns[df_returns['ticker'] != TARGET_ASSET].copy()

        # Get return columns
        return_cols = [col for col in df_returns.columns if col.startswith('r_')]

        # Pivot independent tickers
        df_indep_pivot = df_indep.pivot_table(
            index='date',
            columns='ticker',
            values=return_cols,
            aggfunc='first'
        ).fillna(method='ffill')
        
        # Flatten multi-index columns
        df_indep_pivot.columns = [f"{col2}_{col1}" 
                                for col1, col2 in df_indep_pivot.columns]
        df_indep_pivot.reset_index(inplace=True)

        # Prepare TSN data
        df_tsn_final = df_tsn.drop_duplicates('date')[['date'] + return_cols].copy()
        for col in return_cols:
            df_tsn_final.rename(columns={col: f'{TARGET_ASSET}_{col}'}, inplace=True)

        # Merge all datasets
        df_merged = pd.merge(df_tsn_final, df_indep_pivot, on='date', how='inner')
        df_merged = pd.merge(df_merged, df_kpi, on='date', how='left')
        
        # Sort by date and reset index
        df_merged.sort_values('date', inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        return df_merged

    def prepare_merged_data(self) -> pd.DataFrame:
        """
        Execute full data preparation pipeline.
        """
        # Fetch and prepare price data
        df_prices = self.fetch_and_prepare_price_data()
        
        # Compute returns
        df_returns = self.compute_multi_day_returns(df_prices)
        
        # Remove outliers
        df_returns_clean = self.remove_outliers(df_returns)
        
        # Fetch and prepare economic KPIs
        df_kpi = self.fetch_and_prepare_economic_kpi()
        
        # Merge all data
        df_merged = self.merge_tsn_with_independents(df_returns_clean, df_kpi)
        
        return df_merged

class DimensionReductionAnalysis:
    def __init__(self, use_mysql: bool = True):
        """
        Initialize the analysis either with MySQL data or from CSV.
        
        Parameters:
        use_mysql: If True, fetch data from MySQL; if False, load from CSV
        """
        if use_mysql:
            data_prep = DataPreparation()
            self.data = data_prep.prepare_merged_data()
        else:
            self.data = pd.read_csv('05_merged_data.csv')
            
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        
        # Store column mappings for different variable groups
        self.column_groups = self._create_column_groups()
        
    def _create_column_groups(self) -> Dict[str, List[str]]:
        """
        Create groups of columns based on economic intuition.
        Returns a dictionary mapping group names to lists of column names.
        """
        # Fetch data from MySQL
        engine = create_mysql_connection()
        df_profiles = read_table_as_df('quant1_1_company_profiles', engine)
        df_commodities = read_table_as_df('quant1_1_commodity_list', engine)
        df_prices = read_table_as_df('quant1_1_daily_prices', engine)
        
        # Only consider stock tickers based on category='stock'
        stock_tickers = df_prices[df_prices['category'] == 'stock']['ticker'].unique()
        sector_mappings = df_profiles[df_profiles['ticker'].isin(stock_tickers)].set_index('ticker')['sector'].to_dict()
        
        # Get commodity tickers and their sub-categories
        commodity_prices = df_prices[df_prices['category'] == 'Commodity']
        commodity_mappings = pd.merge(
            commodity_prices[['ticker']].drop_duplicates(),
            df_commodities[['symbol', 'Sub_Category']],
            left_on='ticker',
            right_on='symbol',
            how='inner'
        ).set_index('ticker')['Sub_Category'].to_dict()
        
        # Initialize groups
        groups = {
            'currencies': [col for col in self.data.columns 
                         if any(x in col for x in df_prices[df_prices['category'] == 'forex']['ticker'])],
            'macro': ['GDP_change', 'CPI_change', 
                     'unemploymentRate_change', 'federalFunds_change']
        }
        
        # Add sector-based groups (only for stocks)
        unique_sectors = set(sector_mappings.values())
        for sector in unique_sectors:
            if sector:  # Skip None/NaN
                sector_tickers = [ticker for ticker, sec in sector_mappings.items() 
                                if sec == sector]
                # Filter column names to only include stock tickers with their returns
                sector_cols = []
                for col in self.data.columns:
                    for ticker in sector_tickers:
                        if col.startswith(ticker + "_r_"):  # Only match exact ticker patterns
                            sector_cols.append(col)
                            break
                groups[f'sector_{sector.lower().replace(" ", "_")}'] = sector_cols
        
        # Add commodity sub-category groups
        unique_subcategories = set(commodity_mappings.values())
        for subcategory in unique_subcategories:
            if subcategory:  # Skip None/NaN
                commodity_tickers = [ticker for ticker, sub in commodity_mappings.items() 
                                   if sub == subcategory]
                groups[f'commodity_{subcategory.lower().replace(" ", "_")}'] = \
                    [col for col in self.data.columns 
                     if any(f"{ticker}_r_" in col for ticker in commodity_tickers)]
        
        # Get TSN return columns separately
        self.target_columns = [col for col in self.data.columns 
                             if col.startswith(f'{TARGET_ASSET}_r_')]
        
        return groups
    
    # ... [Rest of the DimensionReductionAnalysis class remains the same] ...
    
    def reduce_dimensions(self, n_components: Dict[str, int] = None) -> pd.DataFrame:
        """
        Apply PCA to each group of variables separately, with NaN handling.
        
        Parameters:
        n_components: Dictionary specifying number of components for each group
                     If None, will automatically select components explaining 80% of variance
        """
        if n_components is None:
            n_components = {group: None for group in self.column_groups.keys()}
            
        reduced_data = {}
        self.pca_models = {}  # Store PCA models for later inspection
        
        # Create MySQL connection for saving PCA parameters
        conn = create_mysql_connection()
        cursor = conn.cursor()
        ts = pd.to_datetime(ANALYSIS_TIMESTAMP, format='%Y%m%d_%H%M%S')
        
        for group, columns in self.column_groups.items():
            if not columns:  # Skip empty groups
                continue
                
            # Prepare data for PCA
            X = self.data[columns].copy()
            
            # Handle missing values using SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # Determine number of components
            n_comp = n_components.get(group, None)
            if n_comp is None:
                n_comp = min(len(columns), 5)  # Maximum 5 components per group
                
            # Apply PCA
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X_scaled)
            
            # Store transformed data
            columns_pca = [f"{group}_PC{i+1}" for i in range(n_comp)]
            reduced_data[group] = pd.DataFrame(
                X_pca, 
                index=self.data.index, 
                columns=columns_pca
            )
            
            # Store PCA model and preprocessing objects
            self.pca_models[group] = {
                'imputer': imputer,
                'scaler': scaler,
                'model': pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'components': pd.DataFrame(
                    pca.components_,
                    columns=columns,
                    index=columns_pca
                )
            }
            
            # Save PCA parameters to MySQL
            insert_sql = """
                INSERT INTO quant2_3_MuAs_StReg_PcaParams
                (target, timestamp, variable_group, columns, imputer_mean, 
                 scaler_mean, scaler_std, n_components, explained_variance_ratio,
                 components, pca_mean)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    columns = VALUES(columns),
                    imputer_mean = VALUES(imputer_mean),
                    scaler_mean = VALUES(scaler_mean),
                    scaler_std = VALUES(scaler_std),
                    n_components = VALUES(n_components),
                    explained_variance_ratio = VALUES(explained_variance_ratio),
                    components = VALUES(components),
                    pca_mean = VALUES(pca_mean)
            """
            
            # Prepare parameters as JSON strings with 10-digit rounding
            params = {
                'target': TARGET_ASSET,
                'timestamp': ts,
                'variable_group': group,
                'columns': json.dumps(columns),
                'imputer_mean': json.dumps([round(x, 10) for x in imputer.statistics_.tolist()]),
                'scaler_mean': json.dumps([round(x, 10) for x in scaler.mean_.tolist()]),
                'scaler_std': json.dumps([round(x, 10) for x in scaler.scale_.tolist()]),
                'n_components': n_comp,
                'explained_variance_ratio': json.dumps([round(x, 10) for x in pca.explained_variance_ratio_.tolist()]),
                'components': json.dumps([[round(x, 10) for x in row] for row in pca.components_.tolist()]),
                'pca_mean': json.dumps([round(x, 10) for x in pca.mean_.tolist()] if hasattr(pca, 'mean_') else None)
            }
            
            cursor.execute(insert_sql, tuple(params.values()))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Combine all reduced data
        self.reduced_data = pd.concat(reduced_data.values(), axis=1)
        
        # Save reduced data and PCA details
        
        # Save reduced data
        output_file = os.path.join(OUTPUT_DIR, f"reduced_data_{ANALYSIS_TIMESTAMP}.csv")
        self.reduced_data.to_csv(output_file)
        
        # Save PCA components for each group
        pca_details = {}
        for group, pca_info in self.pca_models.items():
            pca_details[group] = {
                'explained_variance_ratio': pca_info['explained_variance_ratio'].tolist(),
                'components': pca_info['components'].to_dict()
            }
        
        pca_file = os.path.join(OUTPUT_DIR, f"pca_details_{ANALYSIS_TIMESTAMP}.csv")
        pd.DataFrame(pca_details).to_csv(pca_file)
        
        # Save list of all PCs with their explained variance
        pc_info = []
        for group, pca_info in self.pca_models.items():
            for i, var_ratio in enumerate(pca_info['explained_variance_ratio']):
                pc_name = f"{group}_PC{i+1}"
                pc_info.append({
                    'pc_name': pc_name,
                    'group': group,
                    'component_number': i+1,
                    'explained_variance_ratio': var_ratio,
                    'cumulative_variance_ratio': sum(pca_info['explained_variance_ratio'][:i+1])
                })
        
        # Save PC information
        pc_info_df = pd.DataFrame(pc_info)
        pc_info_file = os.path.join(OUTPUT_DIR, f"principal_components_list_{ANALYSIS_TIMESTAMP}.csv")
        pc_info_df.to_csv(pc_info_file, index=False)
        print(f"Saved PC information to {pc_info_file}")
        
        print(f"Saved reduced data to {output_file}")
        print(f"Saved PCA details to {pca_file}")
        
        return self.reduced_data
    
    def analyze_components(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze the composition of principal components for each group.
        Returns a dictionary of DataFrames showing component loadings.
        """
        component_analysis = {}
        for group, pca_info in self.pca_models.items():
            # Get the component loadings
            loadings = pca_info['components']
            
            # Calculate total contribution of each original variable
            total_contribution = (loadings ** 2).sum()
            
            # Create summary DataFrame
            summary = pd.DataFrame({
                'Total_Contribution': total_contribution,
                'Max_Loading_Component': loadings.abs().idxmax(),
                'Max_Loading_Value': loadings.abs().max()
            })
            
            component_analysis[group] = summary.sort_values('Total_Contribution', ascending=False)
            
        return component_analysis
    
    def build_predictive_model(self, target_horizon: str = f'{TARGET_ASSET}_r_10d', 
                             n_splits: int = 5) -> Dict[str, float]:
        """
        Build and evaluate a predictive model using the reduced dimensions.
        
        Parameters:
        target_horizon: The target variable to predict (e.g., 'TSN_r_10d')
        n_splits: Number of splits for time series cross-validation
        
        Returns:
        Dictionary containing model performance metrics
        """
        # Prepare data
        X = self.reduced_data
        y = self.data[target_horizon]
        
        # Initialize time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Initialize performance metrics
        metrics = {
            'r2_scores': [],
            'mse_scores': [],
            'feature_importances': pd.DataFrame(index=X.columns)
        }
        
        # Perform cross-validation
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            model = LassoCV(cv=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics['r2_scores'].append(r2_score(y_test, y_pred))
            metrics['mse_scores'].append(mean_squared_error(y_test, y_pred))
            metrics['feature_importances'][f'fold_{len(metrics["r2_scores"])}'] = model.coef_
        
        # Calculate average metrics
        final_metrics = {
            'mean_r2': np.mean(metrics['r2_scores']),
            'std_r2': np.std(metrics['r2_scores']),
            'mean_mse': np.mean(metrics['mse_scores']),
            'std_mse': np.std(metrics['mse_scores']),
            'feature_importance': metrics['feature_importances'].mean(axis=1).sort_values(ascending=False)
        }
        
        # Save model metrics
        output_file = os.path.join(OUTPUT_DIR, f"model_metrics_{ANALYSIS_TIMESTAMP}.csv")
        pd.DataFrame({
            'metric': list(final_metrics.keys()),
            'value': list(final_metrics.values())
        }).to_csv(output_file, index=False)
        print(f"Saved model metrics to {output_file}")
        
        self.model_metrics = final_metrics
        return final_metrics
    
    def plot_feature_importance(self):
        """
        Plot the importance of each principal component in predicting TSN returns.
        """
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'Feature': self.model_metrics['feature_importance'].index,
            'Importance': self.model_metrics['feature_importance'].values
        })
        
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance in Predicting TSN Returns')
        plt.tight_layout()
        plt.show()

    def save_pca_composition(self):
        """
        Save detailed information about PCA components into MySQL instead of CSV.
        """
        import json
        
        # Convert ANALYSIS_TIMESTAMP to a proper datetime
        ts = pd.to_datetime(ANALYSIS_TIMESTAMP, format='%Y%m%d_%H%M%S')
        
        # Create MySQL connection
        conn = create_mysql_connection()
        cursor = conn.cursor()

        for group, pca_info in self.pca_models.items():
            components = pca_info['components']
            explained_var = pca_info['explained_variance_ratio']

            # For each principal component in this group
            for pc_idx, row_values in enumerate(components.values):
                pc_name = f"{group}_PC{pc_idx + 1}"
                var_ratio = explained_var[pc_idx]
                
                # Round loadings to 6 digits
                col_names = components.columns.tolist()
                rounded_values = [round(val, 6) for val in row_values]
                loadings_dict = dict(zip(col_names, rounded_values))
                content_str = json.dumps(loadings_dict)
                
                # Modified INSERT statement with ON DUPLICATE KEY UPDATE
                insert_sql = """
                    INSERT INTO quant2_3_MuAs_StReg_PcaDetailComp
                    ( target, timestamp, range_start, range_end, pca_group, component,
                      explained_variance, content )
                    VALUES
                    ( %s, %s, %s, %s, %s, %s, %s, %s )
                    ON DUPLICATE KEY UPDATE
                      explained_variance = VALUES(explained_variance),
                      content = VALUES(content)
                """
                cursor.execute(
                    insert_sql,
                    (
                        TARGET_ASSET,
                        ts,
                        START_DATE.date(),
                        END_DATE.date(),
                        group,
                        pc_name,
                        var_ratio,
                        content_str
                    )
                )

        conn.commit()
        cursor.close()
        conn.close()

class AdvancedTSNAnalysis:
    def __init__(self, base_analysis):
        """
        Initialize with an existing DimensionReductionAnalysis object.
        This allows us to build upon the previous analysis.
        """
        self.base = base_analysis
        self.data = self.base.data.copy()
        self.reduced_data = self.base.reduced_data.copy()
        
    def analyze_rolling_windows(self, target_col: str = f'{TARGET_ASSET}_r_10d', 
                              window_size: int = 126, timestamp=None,
                              end_date=None) -> pd.DataFrame:
        """
        Perform rolling window analysis to identify periods of better predictability.
        
        Parameters:
        target_col: The target TSN return column to predict
        window_size: Size of rolling window in trading days (126 ≈ 6 months)
        timestamp: Optional timestamp for database records
        end_date: Optional end date for analysis (defaults to latest available)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Use provided end_date or latest available
        if end_date is not None:
            data_end_idx = self.reduced_data.index.get_indexer([end_date], method='ffill')[0]
            if data_end_idx >= 0:
                effective_data = self.reduced_data.iloc[:data_end_idx+1]
            else:
                effective_data = self.reduced_data
        else:
            effective_data = self.reduced_data
            
        # Initialize storage for results
        rolling_metrics = []
        
        # Create windows
        total_samples = len(effective_data)
        for start_idx in range(0, total_samples - window_size, 20):  # Step by 20 days
            end_idx = start_idx + window_size
            
            # Get window data
            X_window = self.reduced_data.iloc[start_idx:end_idx]
            y_window = self.data[target_col].iloc[start_idx:end_idx]
            
            # Perform analysis on window
            model = LassoCV(cv=5, random_state=42)
            model.fit(X_window, y_window)
            
            # Calculate metrics
            y_pred = model.predict(X_window)
            r2 = r2_score(y_window, y_pred)
            mse = mean_squared_error(y_window, y_pred)
            
            # Store results
            window_start = self.data.index[start_idx]
            window_end = self.data.index[end_idx-1]
            
            # Get top features for this window
            feature_importance = pd.Series(model.coef_, index=X_window.columns)
            top_features = feature_importance[feature_importance != 0].nlargest(3)
            
            rolling_metrics.append({
                'window_start': window_start,
                'window_end': window_end,
                'r2_score': r2,
                'mse': mse,
                'top_features': top_features.index.tolist(),
                'feature_coefficients': top_features.values.tolist(),
                'avg_target_volatility': y_window.std()
            })
        
        # Convert to DataFrame
        df_metrics = pd.DataFrame(rolling_metrics)
        
        # Save to MySQL
        conn = create_mysql_connection()
        cursor = conn.cursor()
        ts = pd.to_datetime(ANALYSIS_TIMESTAMP, format='%Y%m%d_%H%M%S')
        
        insert_sql = """
            INSERT INTO quant2_3_MuAs_StReg_RollMetrics
            (target, timestamp, horizon, window_start, window_end, r2_score, mse, 
             top_features, feature_coefficients, avg_target_volatility)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                r2_score = VALUES(r2_score),
                mse = VALUES(mse),
                window_end = VALUES(window_end),
                top_features = VALUES(top_features),
                feature_coefficients = VALUES(feature_coefficients),
                avg_target_volatility = VALUES(avg_target_volatility)
        """
        
        for _, row in df_metrics.iterrows():
            cursor.execute(
                insert_sql,
                (
                    TARGET_ASSET,
                    ts,
                    target_col,
                    row['window_start'].date(),
                    row['window_end'].date(),
                    float(row['r2_score']),
                    float(row['mse']),
                    json.dumps(row['top_features']),
                    json.dumps(row['feature_coefficients']),
                    float(row['avg_target_volatility'])
                )
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Saved rolling metrics to MySQL database for horizon {target_col}")
        
        return df_metrics
    
    def plot_rolling_performance(self, rolling_metrics: pd.DataFrame):
        """
        Visualize the rolling window analysis results.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot R² scores
        ax1.plot(rolling_metrics['window_start'], rolling_metrics['r2_score'])
        ax1.set_title('Rolling Window R² Scores')
        ax1.set_xlabel('Window Start Date')
        ax1.set_ylabel('R² Score')
        ax1.grid(True)
        
        # Plot MSE and volatility
        ax2.plot(rolling_metrics['window_start'], rolling_metrics['mse'], 
                label='MSE', color='red')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(rolling_metrics['window_start'], 
                     rolling_metrics['avg_target_volatility'], 
                     label='Target Volatility', color='blue', linestyle='--')
        ax2.set_title('Rolling Window MSE and Target Volatility')
        ax2.set_xlabel('Window Start Date')
        ax2.set_ylabel('MSE', color='red')
        ax2_twin.set_ylabel('Volatility', color='blue')
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    def add_nonlinear_features(self, degree: int = 2) -> pd.DataFrame:
        """
        Add polynomial and interaction terms to the feature set.
        
        Parameters:
        degree: Maximum degree of polynomial features
        """
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(self.reduced_data)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(self.reduced_data.columns)
        
        # Convert to DataFrame
        X_poly_df = pd.DataFrame(X_poly, 
                               columns=feature_names, 
                               index=self.reduced_data.index)
        
        return X_poly_df

class RegimeBasedAnalysis:
    def __init__(self, data, original_data, target_horizon=None):
        """
        Initialize with both the reduced dataset and original dataset.
        
        Parameters:
        data: The reduced (PCA) dataset
        original_data: The original dataset containing raw returns
        target_horizon: Specific return horizon to analyze (e.g. 'EURUSD_r_3d')
        """
        self.data = data
        self.original_data = original_data
        self.target_col = target_horizon if target_horizon else \
            [col for col in original_data.columns if col.startswith(f'{TARGET_ASSET}_r_')][0]
        
    def identify_regimes(self):
        """
        Identify market regimes using:
        1. Rolling volatility
        2. Trend direction
        3. Correlation with sector indices
        """
        window = 20  # 20-day rolling window
        
        # Calculate volatility regime using original TSN returns
        returns = self.original_data[self.target_col]
        rolling_vol = returns.rolling(window).std()
        vol_regimes = pd.qcut(rolling_vol, q=3, labels=['Low', 'Medium', 'High'])
        
        # Calculate trend regime
        rolling_mean = returns.rolling(window).mean()
        trend_regimes = np.where(rolling_mean > 0, 'Up', 'Down')
        
        # Calculate correlation regime with consumer defensive sector
        if 'sector_consumer_defensive_PC1' in self.data.columns:
            sector_data = self.data['sector_consumer_defensive_PC1']
        else:
            # Fallback to first principal component if specific sector not found
            sector_data = self.data.iloc[:, 0]
            
        sector_corr = returns.rolling(window).corr(sector_data)
        corr_regimes = pd.qcut(sector_corr.fillna(0), q=3, labels=['Low', 'Medium', 'High'])
        
        # Save regime identification results
        output_file = os.path.join(OUTPUT_DIR, f"market_regimes_{ANALYSIS_TIMESTAMP}.csv")
        pd.DataFrame({
            'volatility_regime': vol_regimes,
            'trend_regime': trend_regimes,
            'correlation_regime': corr_regimes
        }).to_csv(output_file)
        print(f"Saved market regimes to {output_file}")
        
        return pd.DataFrame({
            'volatility_regime': vol_regimes,
            'trend_regime': trend_regimes,
            'correlation_regime': corr_regimes
        })

    def create_event_features(self):
        """
        Create features based on market events and conditions
        """
        features = pd.DataFrame(index=self.data.index)
        
        # Use original data for commodity moves
        for commodity in ['GC', 'CL', 'NG']:
            col = f"{commodity}_r_5d"
            if col in self.original_data.columns:
                features[f"{commodity}_large_move"] = np.abs(self.original_data[col]) > \
                    self.original_data[col].std() * 2
        
        # Use PCA components for sector rotation
        if 'sector_consumer_defensive_PC1' in self.data.columns and 'market_indices_PC1' in self.data.columns:
            defensive_returns = self.data['sector_consumer_defensive_PC1']
            market_returns = self.data['market_indices_PC1']
            features['defensive_outperformance'] = \
                defensive_returns.rolling(20).mean() > market_returns.rolling(20).mean()
        
        return features

    def build_conditional_models(self, regimes, features):
        """
        Build separate models for different market conditions
        """
        models = {}
        for vol_regime in ['Low', 'Medium', 'High']:
            for trend_regime in ['Up', 'Down']:
                condition = (regimes['volatility_regime'] == vol_regime) & \
                           (regimes['trend_regime'] == trend_regime)
                
                if condition.sum() > 60:  # Minimum sample size
                    X = pd.concat([self.data[condition], features[condition]], axis=1)
                    y = self.original_data[condition][self.target_col]
                    
                    model = LassoCV(cv=5)
                    models[f"{vol_regime}_{trend_regime}"] = model.fit(X, y)
        
        return models

    def evaluate_regime_performance(self, models, regimes, features):
        """
        Evaluate model performance in different regimes and track important variables.
        """
        performance = {}
        for regime_name, model in models.items():
            vol_regime, trend_regime = regime_name.split('_')
            condition = (regimes['volatility_regime'] == vol_regime) & \
                       (regimes['trend_regime'] == trend_regime)
            
            if condition.sum() > 0:
                X_regime = pd.concat([self.data[condition], features[condition]], axis=1)
                y_regime = self.original_data[condition][self.target_col]
                y_pred = model.predict(X_regime)
                
                # Get important features for this regime
                feature_importance = pd.Series(model.coef_, index=X_regime.columns)
                significant_features = feature_importance[feature_importance != 0]
                top_features = significant_features.nlargest(10)  # Get top 10 features
                
                performance[regime_name] = {
                    'r2': r2_score(y_regime, y_pred),
                    'mse': mean_squared_error(y_regime, y_pred),
                    'sample_size': condition.sum(),
                    'important_features': top_features.to_dict(),  # Save feature coefficients
                    'feature_importance_sum': abs(feature_importance).sum()  # Total feature importance
                }
        
        # Remove CSV saving and instead save to MySQL
        import json
        conn = create_mysql_connection()
        cursor = conn.cursor()
        ts = pd.to_datetime(ANALYSIS_TIMESTAMP, format='%Y%m%d_%H%M%S')
        for regime_name, data in performance.items():
            # Round important features
            important_features_dict = {k: round(v, 6) for k, v in data['important_features'].items()}
            content_str = json.dumps(important_features_dict)
            insert_sql = """
                INSERT INTO quant2_3_MuAs_StReg_RegimePerf
                ( target, timestamp, horizon, regime, range_start, range_end,
                  r2, mse, sample_size, importance_features, feature_importance_sum )
                VALUES
                ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s )
                ON DUPLICATE KEY UPDATE
                  r2 = VALUES(r2),
                  mse = VALUES(mse),
                  sample_size = VALUES(sample_size),
                  importance_features = VALUES(importance_features),
                  feature_importance_sum = VALUES(feature_importance_sum)
            """
            cursor.execute(
                insert_sql,
                (
                    TARGET_ASSET,
                    ts,
                    str(self.target_col),
                    str(regime_name),
                    START_DATE.date(),
                    END_DATE.date(),
                    float(data['r2']),
                    float(data['mse']),
                    int(data['sample_size']),
                    content_str,
                    float(data['feature_importance_sum'])
                )
            )
        conn.commit()
        cursor.close()
        conn.close()

        return performance

def run_advanced_analysis(base_analysis):
    """
    Run the complete advanced analysis pipeline for multiple TSN return horizons.
    """
    # Initialize advanced analysis
    advanced = AdvancedTSNAnalysis(base_analysis)
    
    horizons = [f'{TARGET_ASSET}_r_3d', f'{TARGET_ASSET}_r_5d', f'{TARGET_ASSET}_r_10d', f'{TARGET_ASSET}_r_20d']
    all_rolling_results = {}
    
    for horizon in horizons:
        print(f"\nPerforming rolling window analysis for {horizon}...")
        rolling_metrics = advanced.analyze_rolling_windows(target_col=horizon)
        advanced.plot_rolling_performance(rolling_metrics)
        
        best_windows = rolling_metrics.nlargest(3, 'r2_score')
        print(f"\nBest Performing Time Windows for {horizon}:")
        for _, window in best_windows.iterrows():
            print(f"Window: {window['window_start'].date()} to {window['window_end'].date()}")
            print(f"R² Score: {window['r2_score']:.4f}")
            print("Top Features:", window['top_features'])
        
        all_rolling_results[horizon] = rolling_metrics
    
    return all_rolling_results

def run_full_analysis(start_date=None, end_date=None, target_asset=None, timestamp=None):
    """
    Run the complete analysis pipeline with dynamic date range and target asset.
    
    Parameters:
    start_date (datetime): Start date for analysis period
    end_date (datetime): End date for analysis period
    target_asset (str): Target asset to analyze
    timestamp (datetime): Optional execution timestamp
    
    Returns:
    dict: Results including reduced_data, rolling_metrics, and regime information
    """
    # Use provided timestamp or get new one
    if timestamp is None:
        str_timestamp, execution_ts = get_timestamp()
    else:
        # If timestamp is string, convert to datetime
        if isinstance(timestamp, str):
            try:
                execution_ts = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")
            except:
                execution_ts = pd.to_datetime(timestamp)
        else:
            execution_ts = timestamp
            
    # Override global variables if parameters are provided
    global START_DATE, END_DATE, TARGET_ASSET
    
    if start_date is not None:
        START_DATE = pd.to_datetime(start_date)
    if end_date is not None:
        END_DATE = pd.to_datetime(end_date)
    if target_asset is not None:
        TARGET_ASSET = target_asset
        
    print(f"Running full analysis for {TARGET_ASSET}")
    print(f"Analysis period: {START_DATE.date()} to {END_DATE.date()}")
    
    # Initialize base analysis
    base_analysis = DimensionReductionAnalysis(use_mysql=True)
    reduced_data = base_analysis.reduce_dimensions()
    base_analysis.save_pca_composition()
    
    # Initialize for final results
    all_rolling_results = {}
    all_regime_performance = {}
    all_regimes = {}
    
    horizons = [f'{TARGET_ASSET}_r_3d', f'{TARGET_ASSET}_r_5d', 
                f'{TARGET_ASSET}_r_10d', f'{TARGET_ASSET}_r_20d']
                
    for horizon in horizons:
        print(f"\n=== Running analysis for {horizon} ===")
        advanced = AdvancedTSNAnalysis(base_analysis)
        rolling_metrics = advanced.analyze_rolling_windows(target_col=horizon)
        all_rolling_results[horizon] = rolling_metrics

        # Regime-based analysis for this horizon
        regime_analysis = RegimeBasedAnalysis(reduced_data, base_analysis.data, 
                                            target_horizon=horizon)
        regimes = regime_analysis.identify_regimes()
        event_features = regime_analysis.create_event_features()
        regime_models = regime_analysis.build_conditional_models(regimes, event_features)
        regime_performance = regime_analysis.evaluate_regime_performance(
            regime_models, regimes, event_features
        )
        
        all_regimes[horizon] = regimes
        all_regime_performance[horizon] = regime_performance
    
    # Save regime data to MySQL
    conn = create_mysql_connection()
    cursor = conn.cursor()
    ts = pd.to_datetime(execution_ts)

    # Save regime data with the first horizon's data
    first_horizon = horizons[0]
    regimes = all_regimes[first_horizon]
    regimes_with_date = regimes.reset_index()
    
    insert_sql = """
        INSERT INTO quant2_3_MuAs_StReg_RegimeSchedule
        (target, timestamp, date, volatility_regime, trend_regime, correlation_regime)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            volatility_regime = VALUES(volatility_regime),
            trend_regime = VALUES(trend_regime),
            correlation_regime = VALUES(correlation_regime)
    """
    
    for _, row in regimes_with_date.iterrows():
        volatility = None if pd.isna(row['volatility_regime']) else str(row['volatility_regime'])
        trend = None if pd.isna(row['trend_regime']) else str(row['trend_regime'])
        correlation = None if pd.isna(row['correlation_regime']) else str(row['correlation_regime'])
        
        cursor.execute(
            insert_sql,
            (
                TARGET_ASSET,
                ts,
                row['date'].date(),
                volatility,
                trend,
                correlation
            )
        )

    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Analysis completed for {TARGET_ASSET}")
    print(f"Period analyzed: {START_DATE.date()} to {END_DATE.date()}")
    
    return {
        'reduced_data': reduced_data,
        'rolling_metrics': all_rolling_results,
        'regime_performance': all_regime_performance,
        'regimes': all_regimes
    }

if __name__ == "__main__":
    # Run the complete analysis pipeline
    results = run_full_analysis()