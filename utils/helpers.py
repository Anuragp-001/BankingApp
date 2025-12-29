"""
Utility Helper Functions
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def format_number(value: Any, format_type: str = 'default') -> str:
    """Format a number for display."""
    if pd.isna(value):
        return 'N/A'
    
    if format_type == 'currency':
        return f"${value:,.2f}"
    elif format_type == 'percentage':
        return f"{value:.2f}%"
    elif format_type == 'integer':
        return f"{int(value):,}"
    else:
        if isinstance(value, float):
            return f"{value:,.2f}"
        return str(value)


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect semantic types of columns."""
    type_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        if 'amount' in col_lower or 'balance' in col_lower or 'price' in col_lower:
            type_mapping[col] = 'currency'
        elif 'percent' in col_lower or 'rate' in col_lower:
            type_mapping[col] = 'percentage'
        elif 'id' in col_lower or 'count' in col_lower:
            type_mapping[col] = 'integer'
        elif df[col].dtype == 'datetime64[ns]':
            type_mapping[col] = 'datetime'
        elif df[col].dtype in ['int64', 'float64']:
            type_mapping[col] = 'numeric'
        else:
            type_mapping[col] = 'text'
    
    return type_mapping


def generate_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate sample banking transaction data for testing."""
    np.random.seed(42)
    
    categories = ['Groceries', 'Entertainment', 'Utilities', 'Dining', 'Shopping', 
                  'Transportation', 'Healthcare', 'Education', 'Travel', 'Other']
    types = ['Debit', 'Credit', 'Transfer']
    
    data = {
        'id': range(1, n_rows + 1),
        'customer_id': np.random.randint(1000, 2000, n_rows),
        'amount': np.random.exponential(100, n_rows).round(2),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='H'),
        'type': np.random.choice(types, n_rows),
        'category': np.random.choice(categories, n_rows),
        'description': [f"Transaction {i}" for i in range(1, n_rows + 1)]
    }
    
    # Add some outliers
    outlier_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    for idx in outlier_indices:
        data['amount'][idx] = np.random.uniform(1000, 5000)
    
    return pd.DataFrame(data)


def export_to_csv(df: pd.DataFrame) -> bytes:
    """Export DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')


def export_to_pdf(summary: Dict, filename: str = "report.pdf") -> bytes:
    """Export summary report to PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("BankDataLens Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Overview section
    if 'overview' in summary:
        story.append(Paragraph("Data Overview", styles['Heading2']))
        overview = summary['overview']
        overview_data = [
            ["Metric", "Value"],
            ["Total Rows", f"{overview.get('total_rows', 'N/A'):,}"],
            ["Total Columns", str(overview.get('total_columns', 'N/A'))],
            ["Memory Usage", overview.get('memory_usage', 'N/A')]
        ]
        t = Table(overview_data, colWidths=[3*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
    
    # Key Insights
    if 'key_insights' in summary and summary['key_insights']:
        story.append(Paragraph("Key Insights", styles['Heading2']))
        for insight in summary['key_insights'][:5]:
            # Clean markdown formatting for PDF
            clean_insight = insight.replace('**', '').replace('📈', '↑').replace('📉', '↓')
            story.append(Paragraph(f"• {clean_insight}", styles['Normal']))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 20))
    
    # Numeric Statistics
    if 'numeric_trends' in summary and summary['numeric_trends']:
        story.append(Paragraph("Numeric Column Statistics", styles['Heading2']))
        for col, stats in list(summary['numeric_trends'].items())[:5]:
            story.append(Paragraph(f"<b>{col}</b>", styles['Normal']))
            stats_data = [
                ["Mean", "Median", "Min", "Max", "Std Dev"],
                [
                    f"${stats['mean']:,.2f}" if stats['mean'] else 'N/A',
                    f"${stats['median']:,.2f}" if stats['median'] else 'N/A',
                    f"${stats['min']:,.2f}" if stats['min'] else 'N/A',
                    f"${stats['max']:,.2f}" if stats['max'] else 'N/A',
                    f"${stats['std']:,.2f}" if stats['std'] else 'N/A'
                ]
            ]
            t = Table(stats_data, colWidths=[1.2*inch]*5)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(t)
            story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def get_color_palette() -> Dict[str, str]:
    """Get the color palette for visualizations."""
    return {
        'primary': '#1E3A5F',
        'secondary': '#3D7EA6',
        'accent': '#5DA9E9',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'error': '#F44336',
        'background': '#F5F7FA',
        'text': '#2C3E50',
        'chart_colors': ['#1E3A5F', '#3D7EA6', '#5DA9E9', '#7FCDBB', '#A8E6CF', 
                         '#FFD93D', '#FF9800', '#F44336', '#9C27B0', '#673AB7']
    }
