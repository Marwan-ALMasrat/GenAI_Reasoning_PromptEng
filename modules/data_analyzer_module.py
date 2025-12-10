import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class DataAnalyzer:
    """
    Perform exploratory data analysis and statistical computations
    """
    
    def __init__(self, df: pd.DataFrame, column_mapping: Dict[str, Optional[str]]):
        """
        Initialize analyzer with data and column mapping
        
        Args:
            df: Input DataFrame
            column_mapping: Dictionary mapping column roles to actual column names
        """
        self.df = df
        self.column_mapping = column_mapping
    
    def get_basic_stats(self) -> Dict[str, any]:
        """
        Calculate basic statistics about the dataset
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        # Add text column statistics if available
        if self.column_mapping.get('text_column'):
            text_col = self.column_mapping['text_column']
            stats['avg_text_length'] = int(self.df[text_col].astype(str).str.len().mean())
            stats['empty_texts'] = self.df[text_col].isnull().sum()
        
        # Add rating statistics if available
        if self.column_mapping.get('rating_column'):
            rating_col = self.column_mapping['rating_column']
            stats['avg_rating'] = round(self.df[rating_col].mean(), 2)
            stats['rating_std'] = round(self.df[rating_col].std(), 2)
        
        return stats
    
    def analyze_text_statistics(self) -> Optional[Dict]:
        """
        Analyze text-specific statistics
        
        Returns:
            Dictionary with text analysis results
        """
        if not self.column_mapping.get('text_column'):
            return None
        
        text_col = self.column_mapping['text_column']
        
        # Calculate text lengths
        text_lengths = self.df[text_col].astype(str).str.len()
        word_counts = self.df[text_col].astype(str).str.split().str.len()
        
        analysis = {
            'length_stats': {
                'mean': int(text_lengths.mean()),
                'median': int(text_lengths.median()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'std': int(text_lengths.std())
            },
            'word_count_stats': {
                'mean': int(word_counts.mean()),
                'median': int(word_counts.median()),
                'min': int(word_counts.min()),
                'max': int(word_counts.max())
            },
            'empty_count': self.df[text_col].isnull().sum(),
            'unique_count': self.df[text_col].nunique()
        }
        
        return analysis
    
    def analyze_rating_distribution(self) -> Optional[Dict]:
        """
        Analyze rating distribution and patterns
        
        Returns:
            Dictionary with rating analysis
        """
        if not self.column_mapping.get('rating_column'):
            return None
        
        rating_col = self.column_mapping['rating_column']
        
        analysis = {
            'distribution': self.df[rating_col].value_counts().sort_index().to_dict(),
            'statistics': {
                'mean': round(self.df[rating_col].mean(), 2),
                'median': self.df[rating_col].median(),
                'mode': self.df[rating_col].mode().iloc[0] if not self.df[rating_col].mode().empty else None,
                'std': round(self.df[rating_col].std(), 2),
                'min': self.df[rating_col].min(),
                'max': self.df[rating_col].max()
            },
            'percentiles': {
                '25th': self.df[rating_col].quantile(0.25),
                '50th': self.df[rating_col].quantile(0.50),
                '75th': self.df[rating_col].quantile(0.75),
                '90th': self.df[rating_col].quantile(0.90)
            }
        }
        
        # Calculate rating concentration
        total_reviews = len(self.df)
        if total_reviews > 0:
            high_ratings = len(self.df[self.df[rating_col] >= 4])
            low_ratings = len(self.df[self.df[rating_col] <= 2])
            
            analysis['concentration'] = {
                'high_rating_pct': round((high_ratings / total_reviews) * 100, 1),
                'low_rating_pct': round((low_ratings / total_reviews) * 100, 1),
                'polarization_index': round(abs(high_ratings - low_ratings) / total_reviews, 2)
            }
        
        return analysis
    
    def identify_outliers(self) -> Dict:
        """
        Identify outliers in numeric columns
        
        Returns:
            Dictionary with outlier information
        """
        outliers = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round((outlier_count / len(self.df)) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
        
        return outliers
    
    def analyze_temporal_patterns(self) -> Optional[Dict]:
        """
        Analyze temporal patterns if date column exists
        
        Returns:
            Dictionary with temporal analysis
        """
        if not self.column_mapping.get('date_column'):
            return None
        
        date_col = self.column_mapping['date_column']
        
        # Convert to datetime
        df_temp = self.df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=[date_col])
        
        if len(df_temp) == 0:
            return None
        
        analysis = {
            'date_range': {
                'start': df_temp[date_col].min().strftime('%Y-%m-%d'),
                'end': df_temp[date_col].max().strftime('%Y-%m-%d'),
                'span_days': (df_temp[date_col].max() - df_temp[date_col].min()).days
            },
            'frequency': {
                'daily_avg': round(len(df_temp) / ((df_temp[date_col].max() - df_temp[date_col].min()).days + 1), 2)
            }
        }
        
        # Day of week analysis
        df_temp['day_of_week'] = df_temp[date_col].dt.day_name()
        day_dist = df_temp['day_of_week'].value_counts().to_dict()
        analysis['day_of_week_distribution'] = day_dist
        
        # Monthly trend
        df_temp['month'] = df_temp[date_col].dt.to_period('M')
        monthly_counts = df_temp.groupby('month').size().to_dict()
        analysis['monthly_trend'] = {str(k): v for k, v in monthly_counts.items()}
        
        return analysis
    
    def get_correlation_analysis(self) -> Optional[pd.DataFrame]:
        """
        Calculate correlations between numeric columns
        
        Returns:
            Correlation matrix DataFrame
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        correlation_matrix = self.df[numeric_cols].corr()
        return correlation_matrix
    
    def analyze_missing_data(self) -> Dict:
        """
        Detailed analysis of missing data patterns
        
        Returns:
            Dictionary with missing data information
        """
        missing_info = {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': round((missing_count / len(self.df)) * 100, 2),
                    'dtype': str(self.df[col].dtype)
                }
        
        return missing_info
    
    def generate_data_quality_report(self) -> Dict:
        """
        Generate comprehensive data quality report
        
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'completeness': {
                'total_cells': len(self.df) * len(self.df.columns),
                'missing_cells': self.df.isnull().sum().sum(),
                'completeness_score': round((1 - (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)))) * 100, 2)
            },
            'consistency': {
                'duplicate_rows': int(self.df.duplicated().sum()),
                'unique_ratio': round(len(self.df.drop_duplicates()) / len(self.df) * 100, 2)
            },
            'validity': {}
        }
        
        # Check text column validity
        if self.column_mapping.get('text_column'):
            text_col = self.column_mapping['text_column']
            empty_texts = (self.df[text_col].astype(str).str.strip() == '').sum()
            report['validity']['text_empty_count'] = int(empty_texts)
            report['validity']['text_valid_pct'] = round((1 - empty_texts / len(self.df)) * 100, 2)
        
        # Check rating column validity
        if self.column_mapping.get('rating_column'):
            rating_col = self.column_mapping['rating_column']
            valid_range = (self.df[rating_col] >= self.df[rating_col].min()) & (self.df[rating_col] <= self.df[rating_col].max())
            report['validity']['rating_valid_pct'] = round(valid_range.sum() / len(self.df) * 100, 2)
        
        # Overall quality score
        weights = {
            'completeness': 0.4,
            'consistency': 0.3,
            'validity': 0.3
        }
        
        quality_score = (
            report['completeness']['completeness_score'] * weights['completeness'] +
            report['consistency']['unique_ratio'] * weights['consistency']
        )
        
        if 'text_valid_pct' in report['validity']:
            quality_score += report['validity']['text_valid_pct'] * weights['validity']
        else:
            quality_score = quality_score / (weights['completeness'] + weights['consistency']) * 100
        
        report['overall_quality_score'] = round(quality_score, 1)
        
        return report