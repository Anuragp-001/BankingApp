"""
Summarizer Module
Generates AI-powered summaries of trends, patterns, and anomalies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
import statistics


class Summarizer:
    """Generates summaries and insights from banking data."""
    
    def __init__(self):
        self.insights_cache = {}
    
    def summarize_trends(self, data: pd.DataFrame, 
                         similar_results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Summarize trends in the data.
        
        Args:
            data: The DataFrame to analyze
            similar_results: Optional list of similar data points from vector search
            
        Returns:
            Dictionary containing trend summaries
        """
        summary = {
            'overview': self._generate_overview(data),
            'numeric_trends': self._analyze_numeric_trends(data),
            'categorical_distribution': self._analyze_categories(data),
            'temporal_trends': self._analyze_temporal_trends(data),
            'key_insights': []
        }
        
        # Generate key insights
        summary['key_insights'] = self._generate_key_insights(summary)
        
        return summary
    
    def summarize_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify and summarize anomalies in the data.
        
        Args:
            data: The DataFrame to analyze
            
        Returns:
            Dictionary containing anomaly summaries
        """
        anomalies = {
            'numeric_outliers': self._detect_numeric_outliers(data),
            'missing_data': self._analyze_missing_data(data),
            'unusual_patterns': self._detect_unusual_patterns(data),
            'summary': ''
        }
        
        # Generate summary text
        anomalies['summary'] = self._generate_anomaly_summary(anomalies)
        
        return anomalies
    
    def _generate_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a high-level overview of the data."""
        return {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime64']).columns),
            'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
    
    def _analyze_numeric_trends(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze trends in numeric columns."""
        trends = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
                
            trends[col] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75)),
                'skewness': float(col_data.skew()) if len(col_data) > 2 else 0,
                'trend_direction': self._calculate_trend_direction(col_data)
            }
        
        return trends
    
    def _analyze_categories(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze distribution of categorical columns."""
        distributions = {}
        cat_cols = data.select_dtypes(include=['object']).columns
        
        for col in cat_cols:
            value_counts = data[col].value_counts()
            total = len(data[col].dropna())
            
            distributions[col] = {
                'unique_values': int(data[col].nunique()),
                'top_values': [
                    {'value': str(val), 'count': int(count), 'percentage': round(count/total*100, 2)}
                    for val, count in value_counts.head(5).items()
                ],
                'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None
            }
        
        return distributions
    
    def _analyze_temporal_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends over time if datetime columns exist."""
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        temporal_analysis = {}
        
        for col in datetime_cols:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            temporal_analysis[col] = {
                'date_range': {
                    'start': str(col_data.min()),
                    'end': str(col_data.max())
                },
                'span_days': (col_data.max() - col_data.min()).days,
            }
            
            # Analyze by day of week if enough data
            if len(col_data) > 7:
                day_counts = col_data.dt.dayofweek.value_counts().sort_index()
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                temporal_analysis[col]['by_day_of_week'] = {
                    days[i]: int(day_counts.get(i, 0)) for i in range(7)
                }
            
            # Analyze by month if enough data
            if len(col_data) > 30:
                month_counts = col_data.dt.month.value_counts().sort_index()
                temporal_analysis[col]['by_month'] = {
                    int(month): int(count) for month, count in month_counts.items()
                }
        
        return temporal_analysis
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate if a series is trending up, down, or stable."""
        if len(series) < 10:
            return 'insufficient_data'
        
        # Compare first half mean to second half mean
        mid = len(series) // 2
        first_half_mean = series.iloc[:mid].mean()
        second_half_mean = series.iloc[mid:].mean()
        
        change_pct = (second_half_mean - first_half_mean) / first_half_mean * 100 if first_half_mean != 0 else 0
        
        if change_pct > 10:
            return 'increasing'
        elif change_pct < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def _detect_numeric_outliers(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect outliers in numeric columns using IQR method."""
        outliers = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 4:
                continue
            
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_values = col_data[outlier_mask]
            
            if len(outlier_values) > 0:
                outliers[col] = {
                    'count': int(len(outlier_values)),
                    'percentage': round(len(outlier_values) / len(col_data) * 100, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'sample_outliers': outlier_values.head(5).tolist()
                }
        
        return outliers
    
    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze missing data patterns."""
        missing_analysis = {}
        
        for col in data.columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                missing_analysis[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_count / len(data) * 100, 2)
                }
        
        return missing_analysis
    
    def _detect_unusual_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect unusual patterns in the data."""
        patterns = []
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            patterns.append({
                'type': 'duplicates',
                'description': f'Found {duplicate_count} duplicate rows ({round(duplicate_count/len(data)*100, 2)}%)',
                'severity': 'medium' if duplicate_count / len(data) < 0.05 else 'high'
            })
        
        # Check for constant columns
        for col in data.columns:
            if data[col].nunique() == 1:
                patterns.append({
                    'type': 'constant_column',
                    'description': f"Column '{col}' has only one unique value",
                    'severity': 'low'
                })
        
        # Check for highly correlated numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = data[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.95:
                        patterns.append({
                            'type': 'high_correlation',
                            'description': f"Columns '{col1}' and '{col2}' are highly correlated ({corr:.3f})",
                            'severity': 'medium'
                        })
        
        return patterns
    
    def _generate_key_insights(self, summary: Dict) -> List[str]:
        """Generate key insights from the summary."""
        insights = []
        
        # Overview insights
        overview = summary['overview']
        insights.append(f"Dataset contains {overview['total_rows']:,} records with {overview['total_columns']} attributes.")
        
        # Numeric trends insights
        for col, stats in summary['numeric_trends'].items():
            if 'amount' in col.lower() or 'balance' in col.lower():
                insights.append(
                    f"Average {col}: ${stats['mean']:,.2f} (range: ${stats['min']:,.2f} to ${stats['max']:,.2f})"
                )
                if stats['trend_direction'] == 'increasing':
                    insights.append(f"📈 {col} shows an increasing trend over time.")
                elif stats['trend_direction'] == 'decreasing':
                    insights.append(f"📉 {col} shows a decreasing trend over time.")
        
        # Categorical insights
        for col, dist in summary['categorical_distribution'].items():
            if dist['unique_values'] <= 10 and dist['top_values']:
                top = dist['top_values'][0]
                insights.append(
                    f"Most common {col}: '{top['value']}' ({top['percentage']}% of records)"
                )
        
        return insights[:10]  # Limit to top 10 insights
    
    def _generate_anomaly_summary(self, anomalies: Dict) -> str:
        """Generate a text summary of anomalies."""
        parts = []
        
        if anomalies['numeric_outliers']:
            outlier_cols = list(anomalies['numeric_outliers'].keys())
            parts.append(f"Outliers detected in {len(outlier_cols)} column(s): {', '.join(outlier_cols[:3])}")
        
        if anomalies['missing_data']:
            missing_cols = list(anomalies['missing_data'].keys())
            parts.append(f"Missing data found in {len(missing_cols)} column(s)")
        
        if anomalies['unusual_patterns']:
            high_severity = [p for p in anomalies['unusual_patterns'] if p['severity'] == 'high']
            if high_severity:
                parts.append(f"{len(high_severity)} high-severity pattern(s) detected")
        
        if not parts:
            return "No significant anomalies detected in the dataset."
        
        return " | ".join(parts)
    
    def generate_natural_language_summary(self, data: pd.DataFrame) -> str:
        """Generate a natural language summary of the data."""
        trends = self.summarize_trends(data)
        anomalies = self.summarize_anomalies(data)
        
        summary_parts = [
            f"## Data Overview\n",
            f"This dataset contains **{trends['overview']['total_rows']:,} records** across **{trends['overview']['total_columns']} columns**.\n"
        ]
        
        # Add numeric summary
        if trends['numeric_trends']:
            summary_parts.append("\n## Key Metrics\n")
            for col, stats in list(trends['numeric_trends'].items())[:3]:
                summary_parts.append(
                    f"- **{col}**: Average {stats['mean']:,.2f}, ranging from {stats['min']:,.2f} to {stats['max']:,.2f}\n"
                )
        
        # Add categorical summary
        if trends['categorical_distribution']:
            summary_parts.append("\n## Categories\n")
            for col, dist in list(trends['categorical_distribution'].items())[:3]:
                if dist['top_values']:
                    top = dist['top_values'][0]
                    summary_parts.append(
                        f"- **{col}**: {dist['unique_values']} unique values. Most common: '{top['value']}' ({top['percentage']}%)\n"
                    )
        
        # Add anomaly summary
        if anomalies['numeric_outliers'] or anomalies['unusual_patterns']:
            summary_parts.append(f"\n## Data Quality Notes\n{anomalies['summary']}\n")
        
        return "".join(summary_parts)
