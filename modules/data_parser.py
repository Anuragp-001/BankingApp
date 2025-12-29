"""
Data Parser Module
Handles parsing, validation, and preprocessing of uploaded banking datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import re


class DataParser:
    """Parses and preprocesses banking data files."""
    
    REQUIRED_TRANSACTION_COLS = ['amount']
    OPTIONAL_TRANSACTION_COLS = ['id', 'customer_id', 'timestamp', 'date', 'type', 'category', 'description']
    
    NUMERIC_COLS = ['amount', 'balance', 'fee', 'id', 'customer_id']
    DATE_COLS = ['timestamp', 'date', 'transaction_date', 'created_at']
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def parse(self, file, filename: str) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
        """
        Parse uploaded file and return cleaned DataFrame.
        
        Args:
            file: Uploaded file object
            filename: Name of the file
            
        Returns:
            Tuple of (DataFrame or None, errors list, warnings list)
        """
        self.errors = []
        self.warnings = []
        
        try:
            # Determine file type and read
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                self.errors.append(f"Unsupported file format: {filename}. Please upload CSV or Excel files.")
                return None, self.errors, self.warnings
            
            if df.empty:
                self.errors.append("The uploaded file is empty.")
                return None, self.errors, self.warnings
            
            # Clean column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Validate and clean data
            df = self._clean_data(df)
            
            # Check for PII
            self._check_for_pii(df)
            
            return df, self.errors, self.warnings
            
        except Exception as e:
            self.errors.append(f"Error parsing file: {str(e)}")
            return None, self.errors, self.warnings
    
    def _clean_column_name(self, col: str) -> str:
        """Standardize column names."""
        col = str(col).lower().strip()
        col = re.sub(r'[^a-z0-9_]', '_', col)
        col = re.sub(r'_+', '_', col)
        return col.strip('_')
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the DataFrame."""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Convert numeric columns
        for col in df.columns:
            if col in self.NUMERIC_COLS or any(nc in col for nc in ['amount', 'balance', 'fee', 'price', 'cost']):
                df[col] = self._convert_to_numeric(df[col])
        
        # Convert date columns
        for col in df.columns:
            if col in self.DATE_COLS or any(dc in col for dc in ['date', 'time', 'created', 'updated']):
                df[col] = self._convert_to_datetime(df[col])
        
        # Fill missing values for string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].fillna('Unknown')
                self.warnings.append(f"Filled {missing_count} missing values in '{col}' with 'Unknown'")
        
        return df
    
    def _convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert series to numeric, handling currency symbols."""
        if series.dtype in ['int64', 'float64']:
            return series
            
        # Remove currency symbols and commas
        series = series.astype(str).str.replace(r'[$€£¥,]', '', regex=True)
        series = series.str.replace(r'\s+', '', regex=True)
        
        # Convert to numeric
        return pd.to_numeric(series, errors='coerce')
    
    def _convert_to_datetime(self, series: pd.Series) -> pd.Series:
        """Convert series to datetime."""
        try:
            return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        except:
            return series
    
    def _check_for_pii(self, df: pd.DataFrame) -> None:
        """Check for potential PII in the dataset."""
        pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        suspicious_cols = ['name', 'email', 'phone', 'address', 'ssn', 'social_security']
        
        for col in df.columns:
            # Check column names
            if any(sus in col.lower() for sus in suspicious_cols):
                self.warnings.append(f"Column '{col}' may contain PII. Please ensure data is properly anonymized.")
            
            # Check content patterns (sample first 100 rows)
            if df[col].dtype == 'object':
                sample = df[col].head(100).astype(str)
                for pii_type, pattern in pii_patterns.items():
                    if sample.str.contains(pattern, regex=True, na=False).any():
                        self.warnings.append(f"Column '{col}' may contain {pii_type} data. Please verify anonymization.")
                        break
    
    def get_schema_info(self, df: pd.DataFrame) -> Dict:
        """Get schema information about the DataFrame."""
        schema = {
            'columns': [],
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'unique_count': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info['min'] = df[col].min()
                col_info['max'] = df[col].max()
                col_info['mean'] = df[col].mean()
            elif df[col].dtype == 'datetime64[ns]':
                col_info['min'] = str(df[col].min())
                col_info['max'] = str(df[col].max())
            else:
                col_info['sample_values'] = df[col].dropna().head(5).tolist()
            
            schema['columns'].append(col_info)
        
        return schema


class DataUploader:
    """Handles file upload and validation."""
    
    ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_ROWS = 100000
    
    def __init__(self):
        self.parser = DataParser()
    
    def validate_format(self, filename: str, file_size: int) -> Tuple[bool, str]:
        """Validate file format and size."""
        # Check extension
        ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"Invalid file format. Allowed formats: {', '.join(self.ALLOWED_EXTENSIONS)}"
        
        # Check size
        if file_size > self.MAX_FILE_SIZE:
            return False, f"File too large. Maximum size: {self.MAX_FILE_SIZE // (1024*1024)}MB"
        
        return True, "Valid file"
    
    def upload_file(self, file, filename: str) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
        """
        Process uploaded file.
        
        Returns:
            Tuple of (DataFrame or None, errors list, warnings list)
        """
        df, errors, warnings = self.parser.parse(file, filename)
        
        if df is not None and len(df) > self.MAX_ROWS:
            warnings.append(f"Dataset truncated to {self.MAX_ROWS} rows for performance.")
            df = df.head(self.MAX_ROWS)
        
        return df, errors, warnings
    
    def get_schema(self, df: pd.DataFrame) -> Dict:
        """Get schema information for the DataFrame."""
        return self.parser.get_schema_info(df)
