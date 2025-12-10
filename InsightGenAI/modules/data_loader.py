import pandas as pd
import numpy as np
from typing import Optional, Dict
import io

class DataLoader:
    """
    Handles data loading from multiple formats and automatic column detection
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls', 'json']
    
    def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            DataFrame or None if error occurs
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return df
        
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return None
    
    def detect_columns(self, df: pd.DataFrame, sample_size: int = 100) -> Dict[str, Optional[str]]:
        """
        Automatically detect important columns based on content analysis
        
        Args:
            df: Input DataFrame
            sample_size: Number of rows to analyze for detection
            
        Returns:
            Dictionary with detected column names
        """
        mapping = {
            'text_column': None,
            'rating_column': None,
            'id_column': None,
            'date_column': None
        }
        
        sample_df = df.head(min(sample_size, len(df)))
        
        # Detect text column (longest average text)
        text_candidates = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = sample_df[col].astype(str).str.len().mean()
                if avg_length > 50:  # Minimum length for review text
                    text_candidates.append((col, avg_length))
        
        if text_candidates:
            text_candidates.sort(key=lambda x: x[1], reverse=True)
            mapping['text_column'] = text_candidates[0][0]
        
        # Detect rating column (numeric, typically 1-5 or 1-10)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10 and df[col].min() >= 0 and df[col].max() <= 10:
                    # Check if column name suggests rating
                    if any(keyword in col.lower() for keyword in ['star', 'rating', 'score', 'review']):
                        mapping['rating_column'] = col
                        break
        
        # If no rating column found by name, use first numeric column with range 1-5 or 1-10
        if not mapping['rating_column']:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    unique_vals = df[col].dropna().unique()
                    if 3 <= len(unique_vals) <= 10:
                        val_min, val_max = df[col].min(), df[col].max()
                        if (val_min >= 1 and val_max <= 5) or (val_min >= 1 and val_max <= 10):
                            mapping['rating_column'] = col
                            break
        
        # Detect ID column (high cardinality, could be string or numeric)
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'business', 'user', 'customer']):
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.1:  # At least 10% unique values
                    mapping['id_column'] = col
                    break
        
        # Detect date column
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    mapping['date_column'] = col
                    break
                except:
                    pass
        
        return mapping
    
    def normalize_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
        """
        Create a normalized version of the DataFrame with standard column names
        
        Args:
            df: Original DataFrame
            column_mapping: Detected column mapping
            
        Returns:
            Normalized DataFrame
        """
        normalized_df = df.copy()
        
        # Rename columns to standard names
        rename_dict = {}
        if column_mapping['text_column']:
            rename_dict[column_mapping['text_column']] = 'text'
        if column_mapping['rating_column']:
            rename_dict[column_mapping['rating_column']] = 'rating'
        if column_mapping['id_column']:
            rename_dict[column_mapping['id_column']] = 'id'
        if column_mapping['date_column']:
            rename_dict[column_mapping['date_column']] = 'date'
        
        # Only rename if not already using standard names
        for old_name, new_name in rename_dict.items():
            if old_name != new_name and old_name in normalized_df.columns:
                if new_name in normalized_df.columns:
                    # If standard name already exists, append suffix
                    new_name = f"{new_name}_detected"
                normalized_df = normalized_df.rename(columns={old_name: new_name})
        
        return normalized_df
    
    def validate_data(self, df: pd.DataFrame, column_mapping: Dict[str, Optional[str]]) -> Dict[str, any]:
        """
        Validate the loaded data and provide quality metrics
        
        Args:
            df: Input DataFrame
            column_mapping: Column mapping dictionary
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }
        
        # Check if text column exists
        if not column_mapping['text_column']:
            validation['is_valid'] = False
            validation['errors'].append("No text column detected. Cannot proceed with analysis.")
        else:
            text_col = column_mapping['text_column']
            null_count = df[text_col].isnull().sum()
            if null_count > 0:
                validation['warnings'].append(f"Text column has {null_count} null values")
            
            # Check text length
            avg_length = df[text_col].astype(str).str.len().mean()
            if avg_length < 20:
                validation['warnings'].append(f"Average text length is very short ({avg_length:.0f} chars)")
        
        # Info about rating column
        if column_mapping['rating_column']:
            rating_col = column_mapping['rating_column']
            validation['info']['rating_range'] = f"{df[rating_col].min()} - {df[rating_col].max()}"
            validation['info']['rating_distribution'] = df[rating_col].value_counts().to_dict()
        
        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            validation['warnings'].append(f"Found {dup_count} duplicate rows")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        validation['info']['memory_usage_mb'] = f"{memory_mb:.2f} MB"
        
        return validation
