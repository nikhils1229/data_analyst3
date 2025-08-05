import pandas as pd
import numpy as np
import re
from typing import Any, Dict, List, Union
from utils.logger import Logger

class DataProcessor:
    """
    Utility class for common data processing tasks
    """
    
    def __init__(self):
        self.logger = Logger()
    
    def clean_monetary_values(self, series: pd.Series) -> pd.Series:
        """
        Clean monetary values by removing currency symbols and converting to float
        """
        cleaned = series.astype(str)
        # Remove currency symbols and commas
        cleaned = cleaned.str.replace(r'[\$,€£¥]', '', regex=True)
        # Remove any non-digit characters except dots
        cleaned = cleaned.str.replace(r'[^\d.]', '', regex=True)
        return pd.to_numeric(cleaned, errors='coerce')
    
    def extract_years(self, series: pd.Series) -> pd.Series:
        """
        Extract 4-digit years from strings
        """
        years = series.astype(str).str.extract(r'(\d{4})', expand=False)
        return pd.to_numeric(years, errors='coerce')
    
    def clean_numeric_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Clean specified columns to ensure they're numeric
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def remove_outliers(self, series: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series:
        """
        Remove outliers from a series using IQR or Z-score method
        """
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return series[(series >= lower_bound) & (series <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return series[z_scores < factor]
        
        return series
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names by removing spaces, special characters
        """
        df = df.copy()
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
        df.columns = df.columns.str.lower()
        return df
    
    def fill_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Fill missing values using different strategies per column
        strategy: {'column_name': 'mean'|'median'|'mode'|'ffill'|'bfill'|value}
        """
        df = df.copy()
        
        for column, method in strategy.items():
            if column not in df.columns:
                continue
                
            if method == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif method == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif method == 'mode':
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column].fillna(mode_value[0], inplace=True)
            elif method == 'ffill':
                df[column].fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                df[column].fillna(method='bfill', inplace=True)
            else:
                # Assume it's a specific value
                df[column].fillna(method, inplace=True)
        
        return df
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect likely data types for each column
        """
        type_suggestions = {}
        
        for column in df.columns:
            sample_values = df[column].dropna().head(100)
            
            if sample_values.empty:
                type_suggestions[column] = 'unknown'
                continue
            
            # Check if it's numeric
            try:
                pd.to_numeric(sample_values)
                type_suggestions[column] = 'numeric'
                continue
            except:
                pass
            
            # Check if it's date
            try:
                pd.to_datetime(sample_values)
                type_suggestions[column] = 'datetime'
                continue
            except:
                pass
            
            # Check if it's categorical (limited unique values)
            unique_ratio = len(sample_values.unique()) / len(sample_values)
            if unique_ratio < 0.1:
                type_suggestions[column] = 'categorical'
            else:
                type_suggestions[column] = 'text'
        
        return type_suggestions
    
    def convert_data_types(self, df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Convert columns to specified data types
        """
        df = df.copy()
        
        for column, dtype in type_mapping.items():
            if column not in df.columns:
                continue
            
            try:
                if dtype == 'numeric':
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif dtype == 'datetime':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif dtype == 'categorical':
                    df[column] = df[column].astype('category')
                elif dtype == 'text':
                    df[column] = df[column].astype(str)
                    
            except Exception as e:
                self.logger.warning(f"Could not convert {column} to {dtype}: {e}")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data quality and return quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'duplicates': df.duplicated().sum(),
            'quality_score': 0
        }
        
        # Check missing data
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            quality_report['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
        
        # Data types
        quality_report['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Calculate quality score (0-100)
        total_missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        duplicate_pct = (quality_report['duplicates'] / len(df)) * 100
        
        quality_score = max(0, 100 - total_missing_pct - duplicate_pct)
        quality_report['quality_score'] = round(quality_score, 2)
        
        return quality_report