"""
BankDataLens - Interactive Banking Data Explorer
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os
from typing import Optional

# Add the current directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules.data_parser import DataUploader
from modules.embeddings import EmbeddingGenerator
from modules.vector_db import VectorDBClient
from modules.summarizer import Summarizer
from modules.rag_pipeline import RAGPipeline
from utils.helpers import (
    generate_sample_data, 
    export_to_csv, 
    export_to_pdf,
    get_color_palette,
    detect_column_types
)

# Page configuration
st.set_page_config(
    page_title="BankDataLens",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono&display=swap');
    
    .main { font-family: 'DM Sans', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #3D7EA6 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8ECF0;
        transition: transform 0.2s, box-shadow 0.2s;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
        font-family: 'Space Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1E3A5F 0%, #3D7EA6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, #EBF4FF 0%, #F0F9FF 100%);
        border-left: 4px solid #3D7EA6;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
        border-left: 4px solid #10B981;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
    }
    
    .answer-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .suggestion-btn {
        display: inline-block;
        background: #EBF4FF;
        color: #1E3A5F;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        border: 1px solid #BFDBFE;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data': None,
        'embeddings_generated': False,
        'vector_db': None,
        'embedding_generator': None,
        'rag_pipeline': None,
        'summarizer': Summarizer(),
        'chat_history': [],
        'data_summary': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 BankDataLens</h1>
        <p>Interactive Banking Data Explorer with AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)


def process_uploaded_file(uploaded_file):
    """Process the uploaded file."""
    uploader = DataUploader()
    
    is_valid, message = uploader.validate_format(uploaded_file.name, uploaded_file.size)
    
    if not is_valid:
        st.error(message)
        return
    
    with st.spinner("Processing file..."):
        df, errors, warnings = uploader.upload_file(uploaded_file, uploaded_file.name)
        
        if errors:
            for error in errors:
                st.error(error)
            return
        
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        st.session_state.data = df
        st.session_state.embeddings_generated = False
        st.session_state.data_summary = None
        st.success(f"✅ Loaded {len(df):,} rows successfully!")


def generate_embeddings():
    """Generate embeddings for the loaded data."""
    if st.session_state.data is None:
        st.error("Please load data first!")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing embedding model...")
        progress_bar.progress(10)
        
        st.session_state.embedding_generator = EmbeddingGenerator()
        progress_bar.progress(30)
        
        status_text.text("Generating embeddings...")
        embeddings = st.session_state.embedding_generator.generate_embeddings(st.session_state.data)
        progress_bar.progress(60)
        
        status_text.text("Storing in vector database...")
        st.session_state.vector_db = VectorDBClient(dimension=1536)  # text-embedding-3-small dimension
        st.session_state.vector_db.upsert(embeddings)
        progress_bar.progress(80)
        
        status_text.text("Setting up RAG pipeline...")
        st.session_state.rag_pipeline = RAGPipeline(
            st.session_state.embedding_generator,
            st.session_state.vector_db,
            st.session_state.summarizer
        )
        progress_bar.progress(100)
        
        st.session_state.embeddings_generated = True
        status_text.text("✅ Processing complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        st.rerun()
        
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def render_sidebar():
    """Render the sidebar with upload and navigation."""
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload your banking dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Upload anonymized banking data in CSV or Excel format"
        )
        
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Quick Start")
        if st.button("📊 Load Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                st.session_state.data = generate_sample_data(1000)
                st.session_state.embeddings_generated = False
                st.session_state.data_summary = None
                st.success("Sample data loaded!")
                st.rerun()
        
        if st.session_state.data is not None:
            st.markdown("---")
            st.markdown("### 📊 Dataset Info")
            st.markdown(f"**Rows:** {len(st.session_state.data):,}")
            st.markdown(f"**Columns:** {len(st.session_state.data.columns)}")
            
            st.markdown("---")
            st.markdown("### 🧠 AI Processing")
            
            if not st.session_state.embeddings_generated:
                if st.button("⚡ Generate Embeddings", use_container_width=True):
                    generate_embeddings()
            else:
                st.markdown('<div class="success-box">✅ Embeddings ready!</div>', unsafe_allow_html=True)
        
        if st.session_state.data is not None and st.session_state.data_summary is not None:
            st.markdown("---")
            st.markdown("### 📥 Export")
            
            col1, col2 = st.columns(2)
            with col1:
                csv_data = export_to_csv(st.session_state.data)
                st.download_button("📄 CSV", csv_data, "banking_data.csv", "text/csv", use_container_width=True)
            
            with col2:
                pdf_data = export_to_pdf(st.session_state.data_summary)
                st.download_button("📑 PDF", pdf_data, "report.pdf", "application/pdf", use_container_width=True)


def render_overview_tab():
    """Render the data overview tab."""
    if st.session_state.data is None:
        st.info("📤 Upload a dataset to get started!")
        return
    
    df = st.session_state.data
    
    if st.session_state.data_summary is None:
        st.session_state.data_summary = st.session_state.summarizer.summarize_trends(df)
    
    summary = st.session_state.data_summary
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{summary['overview']['total_rows']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Columns</div>
            <div class="metric-value">{summary['overview']['total_columns']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Numeric Columns</div>
            <div class="metric-value">{summary['overview']['numeric_columns']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value">{summary['overview']['memory_usage']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if summary['key_insights']:
        st.markdown("### 💡 Key Insights")
        for insight in summary['key_insights']:
            st.markdown(f"• {insight}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Data Preview")
    st.dataframe(df.head(100), use_container_width=True, height=400)


def render_visualization_tab():
    """Render the visualization tab."""
    if st.session_state.data is None:
        st.info("📤 Upload a dataset to see visualizations!")
        return
    
    df = st.session_state.data
    colors = get_color_palette()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribution Analysis")
        if numeric_cols:
            selected_numeric = st.selectbox("Select numeric column", numeric_cols, key="dist_numeric")
            
            fig = px.histogram(df, x=selected_numeric, nbins=50, color_discrete_sequence=[colors['primary']])
            fig.update_layout(title=f"Distribution of {selected_numeric}", template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            fig_box = px.box(df, y=selected_numeric, color_discrete_sequence=[colors['secondary']])
            fig_box.update_layout(title=f"Box Plot of {selected_numeric}", template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.markdown("### 🥧 Category Analysis")
        if cat_cols:
            selected_cat = st.selectbox("Select categorical column", cat_cols, key="cat_select")
            
            value_counts = df[selected_cat].value_counts().head(10)
            
            fig_pie = px.pie(values=value_counts.values, names=value_counts.index, color_discrete_sequence=colors['chart_colors'])
            fig_pie.update_layout(title=f"Distribution of {selected_cat}", template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            fig_bar = px.bar(x=value_counts.index, y=value_counts.values, color_discrete_sequence=[colors['accent']])
            fig_bar.update_layout(title=f"Count by {selected_cat}", template="plotly_white", xaxis_title=selected_cat, yaxis_title="Count")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No categorical columns found in the dataset.")
    
    if datetime_cols and numeric_cols:
        st.markdown("### 📈 Time Series Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            time_col = st.selectbox("Select time column", datetime_cols)
        with col2:
            value_col = st.selectbox("Select value column", numeric_cols, key="ts_value")
        
        df_sorted = df.sort_values(time_col).copy()
        df_sorted = df_sorted.set_index(time_col)
        
        date_range = (df_sorted.index.max() - df_sorted.index.min()).days
        freq = 'M' if date_range > 365 else 'W' if date_range > 30 else 'D'
        freq_label = {'M': 'Monthly', 'W': 'Weekly', 'D': 'Daily'}[freq]
        
        resampled = df_sorted[value_col].resample(freq).agg(['mean', 'sum', 'count'])
        
        fig_ts = make_subplots(rows=2, cols=1, subplot_titles=(f'{freq_label} Average {value_col}', f'{freq_label} Total {value_col}'), vertical_spacing=0.15)
        
        fig_ts.add_trace(go.Scatter(x=resampled.index, y=resampled['mean'], mode='lines+markers', name='Average', line=dict(color=colors['primary'])), row=1, col=1)
        fig_ts.add_trace(go.Bar(x=resampled.index, y=resampled['sum'], name='Total', marker_color=colors['secondary']), row=2, col=1)
        
        fig_ts.update_layout(height=600, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    if len(numeric_cols) > 1:
        st.markdown("### 🔥 Correlation Heatmap")
        corr_matrix = df[numeric_cols].corr()
        
        fig_heatmap = px.imshow(corr_matrix, labels=dict(color="Correlation"), color_continuous_scale='RdBu_r', aspect="auto")
        fig_heatmap.update_layout(title="Correlation Matrix", template="plotly_white")
        st.plotly_chart(fig_heatmap, use_container_width=True)


def render_anomalies_tab():
    """Render the anomalies detection tab."""
    if st.session_state.data is None:
        st.info("📤 Upload a dataset to detect anomalies!")
        return
    
    df = st.session_state.data
    colors = get_color_palette()
    
    anomalies = st.session_state.summarizer.summarize_anomalies(df)
    
    st.markdown("### 🔍 Anomaly Detection Results")
    
    st.markdown(f'<div class="info-box"><strong>Summary:</strong> {anomalies["summary"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Numeric Outliers")
        
        if anomalies['numeric_outliers']:
            for col, info in anomalies['numeric_outliers'].items():
                st.markdown(f"""
                **{col}**
                - Outliers found: {info['count']} ({info['percentage']}%)
                - Normal range: [{info['lower_bound']:,.2f}, {info['upper_bound']:,.2f}]
                """)
                
                col_data = df[col].dropna()
                
                fig = go.Figure()
                fig.add_trace(go.Box(y=col_data, name=col, boxpoints='outliers', marker_color=colors['primary']))
                fig.update_layout(title=f"Outliers in {col}", template="plotly_white", height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No significant outliers detected in numeric columns!")
    
    with col2:
        st.markdown("#### ⚠️ Unusual Patterns")
        
        if anomalies['unusual_patterns']:
            for pattern in anomalies['unusual_patterns']:
                severity_color = {'high': '#F44336', 'medium': '#FF9800', 'low': '#4CAF50'}
                st.markdown(f"""
                <div style="padding: 0.75rem; background: {severity_color.get(pattern['severity'], '#E2E8F0')}22; 
                border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {severity_color.get(pattern['severity'], '#E2E8F0')}">
                    <strong>{pattern['type'].replace('_', ' ').title()}</strong><br>
                    <span style="color: #666">{pattern['description']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No unusual patterns detected!")
        
        st.markdown("#### 📭 Missing Data")
        
        if anomalies['missing_data']:
            missing_df = pd.DataFrame([
                {'Column': col, 'Missing': info['missing_count'], 'Percentage': f"{info['missing_percentage']}%"}
                for col, info in anomalies['missing_data'].items()
            ])
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing data!")


def render_qa_tab():
    """Render the Q&A tab with RAG functionality."""
    if st.session_state.data is None:
        st.info("📤 Upload a dataset to ask questions!")
        return
    
    if not st.session_state.embeddings_generated:
        st.warning("⚡ Please generate embeddings first (see sidebar) to enable AI-powered Q&A!")
        return
    
    st.markdown("### 💬 Ask Questions About Your Data")
    
    suggestions = st.session_state.rag_pipeline.get_suggested_questions(st.session_state.data)
    
    st.markdown("**Suggested Questions:**")
    suggestion_cols = st.columns(4)
    for i, suggestion in enumerate(suggestions[:8]):
        with suggestion_cols[i % 4]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.current_question = suggestion
    
    st.markdown("---")
    
    user_question = st.text_input(
        "Your question:",
        value=st.session_state.get('current_question', ''),
        placeholder="e.g., What is the average transaction amount?",
        key="user_question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    if ask_button and user_question:
        with st.spinner("Analyzing your question..."):
            result = st.session_state.rag_pipeline.generate_answer(
                user_question,
                data=st.session_state.data
            )
            
            st.session_state.chat_history.append({
                'question': user_question,
                'answer': result
            })
    
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Conversation History")
        
        for i, exchange in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.container():
                st.markdown(f"**🙋 Question:** {exchange['question']}")
                
                result = exchange['answer']
                confidence_color = "#10B981" if result['confidence'] > 0.8 else "#F59E0B" if result['confidence'] > 0.6 else "#EF4444"
                
                st.markdown(f"""
                <div class="answer-box">
                    <div style="margin-bottom: 0.5rem;">
                        <span style="background: {confidence_color}22; color: {confidence_color}; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                            Confidence: {result['confidence']:.0%}
                        </span>
                        <span style="background: #E2E8F0; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; margin-left: 0.5rem;">
                            Type: {result['query_type']}
                        </span>
                    </div>
                    <div style="margin-top: 1rem;">
                        {result['answer'].replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if result.get('relevant_data'):
                    with st.expander(f"📊 View {len(result['relevant_data'])} relevant data points"):
                        for j, item in enumerate(result['relevant_data'][:5], 1):
                            metadata = item.get('metadata', {})
                            score = item.get('score', 0)
                            st.markdown(f"**#{j}** (Relevance: {score:.2%})")
                            st.json({k: v for k, v in metadata.items() if k not in ['text', 'row_index']})
                
                st.markdown("---")


def render_schema_tab():
    """Render the schema/data dictionary tab."""
    if st.session_state.data is None:
        st.info("📤 Upload a dataset to view schema!")
        return
    
    df = st.session_state.data
    uploader = DataUploader()
    schema = uploader.get_schema(df)
    
    st.markdown("### 📖 Data Dictionary")
    
    st.markdown(f"""
    <div class="info-box">
        <strong>Dataset Overview:</strong> {schema['row_count']:,} rows × {schema['column_count']} columns
    </div>
    """, unsafe_allow_html=True)
    
    schema_data = []
    for col_info in schema['columns']:
        row = {
            'Column': col_info['name'],
            'Type': col_info['dtype'],
            'Non-Null': f"{col_info['non_null_count']:,}",
            'Null': f"{col_info['null_count']:,}",
            'Unique': f"{col_info['unique_count']:,}"
        }
        
        if 'min' in col_info:
            row['Min'] = f"{col_info['min']:,.2f}" if isinstance(col_info['min'], float) else str(col_info['min'])
            row['Max'] = f"{col_info['max']:,.2f}" if isinstance(col_info['max'], float) else str(col_info['max'])
            if 'mean' in col_info:
                row['Mean'] = f"{col_info['mean']:,.2f}"
        elif 'sample_values' in col_info:
            row['Sample Values'] = ', '.join(str(v) for v in col_info['sample_values'][:3])
        
        schema_data.append(row)
    
    schema_df = pd.DataFrame(schema_data)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
    
    col_types = detect_column_types(df)
    
    st.markdown("### 🏷️ Detected Column Types")
    type_counts = {}
    for col, col_type in col_types.items():
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    for i, (col_type, count) in enumerate(type_counts.items()):
        with cols[i % 4]:
            st.metric(col_type.title(), count)


def main():
    """Main application function."""
    init_session_state()
    render_header()
    render_sidebar()
    
    tabs = st.tabs(["📊 Overview", "📈 Visualizations", "🔍 Anomalies", "💬 Q&A", "📖 Schema"])
    
    with tabs[0]:
        render_overview_tab()
    
    with tabs[1]:
        render_visualization_tab()
    
    with tabs[2]:
        render_anomalies_tab()
    
    with tabs[3]:
        render_qa_tab()
    
    with tabs[4]:
        render_schema_tab()


if __name__ == "__main__":
    main()
