import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import json

# ==================== DataLoader Class ====================
class DataLoader:
    """Handles data loading from various sources"""
    
    @staticmethod
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                return pd.read_json(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Automatically detect text, rating, and ID columns"""
        column_mapping = {
            'text_column': None,
            'rating_column': None,
            'id_column': None
        }
        
        # Detect text column (longest average text length)
        text_candidates = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:
                    text_candidates.append((col, avg_length))
        
        if text_candidates:
            column_mapping['text_column'] = max(text_candidates, key=lambda x: x[1])[0]
        
        # Detect rating column
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['rating', 'score', 'stars', 'rate']):
                column_mapping['rating_column'] = col
                break
        
        # Detect ID column
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'index', 'key']):
                column_mapping['id_column'] = col
                break
        
        return column_mapping


# ==================== DataAnalyzer Class ====================
class DataAnalyzer:
    """Performs statistical analysis on data"""
    
    def __init__(self, df: pd.DataFrame, column_mapping: Dict):
        self.df = df
        self.column_mapping = column_mapping
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics"""
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }


# ==================== NLPProcessor Class ====================
class NLPProcessor:
    """Processes text data and performs sentiment analysis"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean_text_pipeline(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Clean text column"""
        df = df.copy()
        df['clean_text'] = df[text_column].apply(self.clean_text)
        return df
    
    @staticmethod
    def simple_sentiment_analysis(text: str) -> tuple:
        """Simple rule-based sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'love', 'best', 'perfect', 'awesome', 'brilliant', 'outstanding']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'hate',
                         'disappointing', 'useless', 'waste', 'regret', 'never']
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = min(0.5 + (positive_count * 0.1), 1.0)
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = max(0.5 - (negative_count * 0.1), 0.0)
        else:
            sentiment = 'neutral'
            score = 0.5
        
        return sentiment, score
    
    def analyze_sentiment_batched(self, df: pd.DataFrame, text_column: str, 
                                  batch_size: int = 200, progress_callback=None) -> pd.DataFrame:
        """Analyze sentiment in batches"""
        df = df.copy()
        total_rows = len(df)
        
        sentiments = []
        scores = []
        
        for i in range(0, total_rows, batch_size):
            batch = df[text_column].iloc[i:i+batch_size]
            
            for text in batch:
                sentiment, score = self.simple_sentiment_analysis(text)
                sentiments.append(sentiment)
                scores.append(score)
            
            if progress_callback:
                progress = (i + batch_size) / total_rows
                progress_callback(min(progress, 1.0))
        
        df['sentiment'] = sentiments
        df['sentiment_score'] = scores
        
        return df


# ==================== GenAIInsights Class ====================
class GenAIInsights:
    """Generates AI-powered insights"""
    
    def __init__(self):
        self.api_key = None
    
    def set_api_key(self, api_key: str):
        """Set API key for AI service"""
        self.api_key = api_key
    
    def analyze_review_chain_of_thought(self, text: str, sentiment: str, 
                                       confidence: float) -> Optional[Dict[str, str]]:
        """Generate AI analysis using chain-of-thought reasoning"""
        if not self.api_key:
            return None
        
        # Simulated AI response (replace with actual API call)
        analysis = f"""
**Sentiment Analysis:** {sentiment.capitalize()} (Confidence: {confidence:.1%})

**Key Observations:**
- The review expresses {sentiment} sentiment
- Confidence level indicates {'strong' if confidence > 0.7 else 'moderate'} certainty
- Text length suggests {'detailed' if len(text) > 100 else 'brief'} feedback

**Emotional Indicators:**
- Tone: {'Enthusiastic' if sentiment == 'positive' else 'Critical' if sentiment == 'negative' else 'Balanced'}
- Language intensity: {'High' if confidence > 0.8 else 'Medium' if confidence > 0.5 else 'Low'}
"""

        recommendations = f"""
**Action Items:**
1. {'Leverage this positive feedback in marketing' if sentiment == 'positive' else 'Address concerns raised immediately' if sentiment == 'negative' else 'Monitor for trends'}
2. {'Consider featuring this testimonial' if sentiment == 'positive' else 'Investigate root cause' if sentiment == 'negative' else 'Gather more detailed feedback'}
3. {'Share with product team for validation' if sentiment == 'positive' else 'Escalate to management' if sentiment == 'negative' else 'Continue monitoring'}

**Priority Level:** {'Low' if sentiment == 'positive' else 'High' if sentiment == 'negative' else 'Medium'}
"""
        
        return {
            'analysis': analysis,
            'recommendations': recommendations
        }


# ==================== Visualizer Class ====================
class Visualizer:
    """Creates visualizations for data insights"""
    
    @staticmethod
    def plot_rating_distribution(df: pd.DataFrame, rating_column: str):
        """Plot rating distribution"""
        fig = px.histogram(df, x=rating_column, 
                          title='Rating Distribution',
                          labels={rating_column: 'Rating', 'count': 'Frequency'},
                          color_discrete_sequence=['#667eea'])
        fig.update_layout(bargap=0.1)
        return fig
    
    @staticmethod
    def plot_sentiment_distribution(df: pd.DataFrame):
        """Plot sentiment distribution"""
        sentiment_counts = df['sentiment'].value_counts()
        
        colors = {'positive': '#38ef7d', 'neutral': '#667eea', 'negative': '#f45c43'}
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker=dict(colors=[colors.get(s, '#666') for s in sentiment_counts.index]),
            hole=0.4
        )])
        fig.update_layout(title='Sentiment Distribution')
        return fig
    
    @staticmethod
    def plot_sentiment_timeline(df: pd.DataFrame, date_column: str):
        """Plot sentiment over time"""
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        df_copy = df_copy.dropna(subset=[date_column])
        
        sentiment_over_time = df_copy.groupby([pd.Grouper(key=date_column, freq='D'), 'sentiment']).size().reset_index(name='count')
        
        fig = px.line(sentiment_over_time, x=date_column, y='count', color='sentiment',
                     title='Sentiment Trends Over Time',
                     color_discrete_map={'positive': '#38ef7d', 'neutral': '#667eea', 'negative': '#f45c43'})
        return fig
    
    @staticmethod
    def plot_rating_sentiment_heatmap(df: pd.DataFrame, rating_column: str):
        """Plot rating vs sentiment heatmap"""
        cross_tab = pd.crosstab(df[rating_column], df['sentiment'])
        
        fig = px.imshow(cross_tab, 
                       title='Rating vs Sentiment Heatmap',
                       labels=dict(x='Sentiment', y='Rating', color='Count'),
                       color_continuous_scale='Blues')
        return fig
    
    @staticmethod
    def plot_sentiment_score_distribution(df: pd.DataFrame):
        """Plot sentiment score distribution"""
        fig = px.histogram(df, x='sentiment_score', color='sentiment',
                          title='Sentiment Score Distribution',
                          nbins=50,
                          color_discrete_map={'positive': '#38ef7d', 'neutral': '#667eea', 'negative': '#f45c43'})
        return fig
    
    @staticmethod
    def generate_wordcloud(df: pd.DataFrame, text_column: str):
        """Generate word cloud visualization"""
        from io import BytesIO
        import matplotlib.pyplot as plt
        
        try:
            from wordcloud import WordCloud
            
            text = ' '.join(df[text_column].dropna().astype(str))
            
            wordcloud = WordCloud(width=400, height=300, 
                                background_color='white',
                                colormap='viridis').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            return fig
        except ImportError:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'WordCloud library not installed', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig


# ==================== STREAMLIT APP ====================

# Page configuration
st.set_page_config(
    page_title="InsightGenAI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}

# Header
st.markdown('<h1 class="main-header">🔮 InsightGenAI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Data Analysis & Sentiment Insights Platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["📁 Data Upload", "📊 Data Analysis", "🧠 NLP Processing", "🤖 AI Insights", "📈 Visualizations"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("InsightGenAI automates data analysis, sentiment analysis, and generates AI-powered business insights from your data.")
    
    if st.session_state.data_loaded:
        st.success(f"✅ Data Loaded: {len(st.session_state.df)} rows")

# Page 1: Data Upload
if page == "📁 Data Upload":
    st.markdown('<h2 class="section-header">📁 Data Upload & Column Detection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your data file (CSV, Excel, JSON)",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload any data file containing text reviews or feedback"
        )
    
    with col2:
        st.markdown("### Supported Formats")
        st.markdown("- 📄 CSV files")
        st.markdown("- 📊 Excel files")
        st.markdown("- 📋 JSON files")
    
    if uploaded_file is not None:
        with st.spinner("Loading and analyzing your data..."):
            loader = DataLoader()
            df = loader.load_file(uploaded_file)
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Automatic column detection
                st.markdown('<h3 class="section-header">🔍 Automatic Column Detection</h3>', unsafe_allow_html=True)
                
                column_mapping = loader.detect_columns(df)
                st.session_state.column_mapping = column_mapping
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📝 Text Column Detected:**")
                    if column_mapping['text_column']:
                        st.success(f"✅ {column_mapping['text_column']}")
                    else:
                        st.warning("⚠️ No text column found")
                
                with col2:
                    st.markdown("**⭐ Rating Column Detected:**")
                    if column_mapping['rating_column']:
                        st.success(f"✅ {column_mapping['rating_column']}")
                    else:
                        st.info("ℹ️ No rating column found")
                
                with col3:
                    st.markdown("**🔢 ID Column Detected:**")
                    if column_mapping['id_column']:
                        st.success(f"✅ {column_mapping['id_column']}")
                    else:
                        st.info("ℹ️ No ID column found")
                
                # Manual override option
                with st.expander("🔧 Manual Column Selection (Optional)"):
                    cols = st.columns(3)
                    with cols[0]:
                        text_col = st.selectbox("Select Text Column", ['Auto'] + list(df.columns))
                        if text_col != 'Auto':
                            column_mapping['text_column'] = text_col
                    with cols[1]:
                        rating_col = st.selectbox("Select Rating Column", ['Auto'] + list(df.columns))
                        if rating_col != 'Auto':
                            column_mapping['rating_column'] = rating_col
                    with cols[2]:
                        id_col = st.selectbox("Select ID Column", ['Auto'] + list(df.columns))
                        if id_col != 'Auto':
                            column_mapping['id_column'] = id_col
                
                # Data preview
                st.markdown('<h3 class="section-header">📋 Data Preview</h3>', unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                with col4:
                    st.metric("Duplicates", f"{df.duplicated().sum():,}")

# Page 2: Data Analysis
elif page == "📊 Data Analysis":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the Data Upload page.")
    else:
        st.markdown('<h2 class="section-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        analyzer = DataAnalyzer(st.session_state.df, st.session_state.column_mapping)
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Data Overview")
            stats = analyzer.get_basic_stats()
            for key, value in stats.items():
                st.metric(key.replace('_', ' ').title(), value)
        
        with col2:
            st.markdown("### 📊 Data Types")
            dtype_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Type': st.session_state.df.dtypes.astype(str),
                'Non-Null': st.session_state.df.count().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        # Rating distribution if available
        if st.session_state.column_mapping['rating_column']:
            st.markdown('<h3 class="section-header">⭐ Rating Distribution</h3>', unsafe_allow_html=True)
            visualizer = Visualizer()
            fig = visualizer.plot_rating_distribution(
                st.session_state.df,
                st.session_state.column_mapping['rating_column']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Text length analysis
        if st.session_state.column_mapping['text_column']:
            st.markdown('<h3 class="section-header">📝 Text Length Analysis</h3>', unsafe_allow_html=True)
            text_col = st.session_state.column_mapping['text_column']
            st.session_state.df['text_length'] = st.session_state.df[text_col].astype(str).str.len()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Length", f"{st.session_state.df['text_length'].mean():.0f} chars")
            with col2:
                st.metric("Median Length", f"{st.session_state.df['text_length'].median():.0f} chars")
            with col3:
                st.metric("Max Length", f"{st.session_state.df['text_length'].max():.0f} chars")

# Page 3: NLP Processing
elif page == "🧠 NLP Processing":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the Data Upload page.")
    elif not st.session_state.column_mapping['text_column']:
        st.error("❌ No text column detected. Cannot perform NLP processing.")
    else:
        st.markdown('<h2 class="section-header">🧠 NLP Processing Pipeline</h2>', unsafe_allow_html=True)
        
        nlp_processor = NLPProcessor()
        
        # Processing options
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Processing Configuration")
        with col2:
            batch_size = st.number_input("Batch Size", min_value=50, max_value=500, value=200, step=50)
        
        if st.button("🚀 Start NLP Processing", type="primary"):
            text_column = st.session_state.column_mapping['text_column']
            
            # Step 1: Text Cleaning
            st.markdown("#### Step 1: Text Cleaning")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Cleaning text data..."):
                cleaned_df = nlp_processor.clean_text_pipeline(
                    st.session_state.df.copy(),
                    text_column
                )
                progress_bar.progress(33)
                status_text.success("✅ Text cleaning completed!")
            
            # Step 2: Sentiment Analysis
            st.markdown("#### Step 2: Sentiment Analysis")
            with st.spinner(f"Analyzing sentiment in batches of {batch_size}..."):
                processed_df = nlp_processor.analyze_sentiment_batched(
                    cleaned_df,
                    'clean_text',
                    batch_size=batch_size,
                    progress_callback=lambda x: progress_bar.progress(33 + int(x * 0.67))
                )
                progress_bar.progress(100)
                status_text.success("✅ Sentiment analysis completed!")
            
            st.session_state.processed_df = processed_df
            
            # Results summary
            st.markdown('<h3 class="section-header">📊 Sentiment Analysis Results</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            sentiment_counts = processed_df['sentiment'].value_counts()
            
            with col1:
                positive_pct = (sentiment_counts.get('positive', 0) / len(processed_df)) * 100
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    <h3 style="margin: 0;">😊 Positive</h3>
                    <h2 style="margin: 10px 0;">{sentiment_counts.get('positive', 0):,}</h2>
                    <p style="margin: 0;">{positive_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                neutral_pct = (sentiment_counts.get('neutral', 0) / len(processed_df)) * 100
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3 style="margin: 0;">😐 Neutral</h3>
                    <h2 style="margin: 10px 0;">{sentiment_counts.get('neutral', 0):,}</h2>
                    <p style="margin: 0;">{neutral_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                negative_pct = (sentiment_counts.get('negative', 0) / len(processed_df)) * 100
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);">
                    <h3 style="margin: 0;">😞 Negative</h3>
                    <h2 style="margin: 10px 0;">{sentiment_counts.get('negative', 0):,}</h2>
                    <p style="margin: 0;">{negative_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Download processed data
            st.markdown("### 💾 Download Processed Data")
            csv = processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="processed_data_with_sentiment.csv",
                mime="text/csv"
            )

# Page 4: AI Insights
elif page == "🤖 AI Insights":
    if st.session_state.processed_df is None:
        st.warning("⚠️ Please complete NLP Processing first.")
    else:
        st.markdown('<h2 class="section-header">🤖 AI-Generated Insights</h2>', unsafe_allow_html=True)
        
        genai = GenAIInsights()
        
        # API Key input
        with st.expander("🔑 API Configuration", expanded=True):
            api_key = st.text_input("Enter OpenRouter API Key", type="password")
            if api_key:
                genai.set_api_key(api_key)
        
        # Sample selection
        st.markdown("### 📝 Select Review for Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            filter_sentiment = st.multiselect(
                "Filter by Sentiment",
                ['positive', 'neutral', 'negative'],
                default=['negative']
            )
        with col2:
            num_samples = st.number_input("Number of samples", 1, 10, 3)
        
        filtered_df = st.session_state.processed_df[
            st.session_state.processed_df['sentiment'].isin(filter_sentiment)
        ]
        
        if len(filtered_df) > 0:
            sample_indices = filtered_df.sample(min(num_samples, len(filtered_df))).index
            
            for idx in sample_indices:
                sample = st.session_state.processed_df.loc[idx]
                
                with st.expander(f"📄 Review #{idx} - {sample['sentiment'].upper()}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("**Original Text:**")
                        st.write(sample[st.session_state.column_mapping['text_column']])
                    
                    with col2:
                        st.metric("Sentiment", sample['sentiment'])
                        st.metric("Confidence", f"{sample['sentiment_score']:.2%}")
                    
                    if st.button(f"🔍 Generate Insights for Review #{idx}", key=f"btn_{idx}"):
                        if not api_key:
                            st.error("⚠️ Please enter API key first")
                        else:
                            with st.spinner("Generating AI insights..."):
                                analysis = genai.analyze_review_chain_of_thought(
                                    sample[st.session_state.column_mapping['text_column']],
                                    sample['sentiment'],
                                    sample['sentiment_score']
                                )
                                
                                if analysis:
                                    st.markdown("#### 🧠 AI Analysis")
                                    st.markdown(analysis['analysis'])
                                    
                                    st.markdown("#### 💡 Recommendations")
                                    st.markdown(analysis['recommendations'])
        else:
            st.info("No reviews found with selected sentiment filters.")

# Page 5: Visualizations
elif page == "📈 Visualizations":
    if st.session_state.processed_df is None:
        st.warning("⚠️ Please complete NLP Processing first to see visualizations.")
    else:
        st.markdown('<h2 class="section-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        visualizer = Visualizer()
        df = st.session_state.processed_df
        
        # Sentiment distribution
        st.markdown("### 📊 Sentiment Distribution")
        fig1 = visualizer.plot_sentiment_distribution(df)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Sentiment over time (if date column exists)
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            st.markdown("### 📅 Sentiment Trends Over Time")
            fig2 = visualizer.plot_sentiment_timeline(df, date_columns[0])
            st.plotly_chart(fig2, use_container_width=True)
        
        # Rating vs Sentiment correlation
        if st.session_state.column_mapping['rating_column']:
            st.markdown("### ⭐ Rating vs Sentiment Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig3 = visualizer.plot_rating_sentiment_heatmap(
                    df,
                    st.session_state.column_mapping['rating_column']
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig4 = visualizer.plot_sentiment_score_distribution(df)
                st.plotly_chart(fig4, use_container_width=True)
        
        # Word clouds
        st.markdown("### ☁️ Word Clouds by Sentiment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Positive Reviews")
            positive_data = df[df['sentiment'] == 'positive']
            if len(positive_data) > 0:
                fig5 = visualizer.generate_wordcloud(positive_data, 'clean_text')
                st.pyplot(fig5)
            else:
                st.info("No positive reviews available")
        
        with col2:
            st.markdown("#### Neutral Reviews")
            neutral_data = df[df['sentiment'] == 'neutral']
            if len(neutral_data) > 0:
                fig6 = visualizer.generate_wordcloud(neutral_data, 'clean_text')
                st.pyplot(fig6)
            else:
                st.info("No neutral reviews available")
        
        with col3:
            st.markdown("#### Negative Reviews")
            negative_data = df[df['sentiment'] == 'negative']
            if len(negative_data) > 0:
                fig7 = visualizer.generate_wordcloud(negative_data, 'clean_text')
                st.pyplot(fig7)
            else:
                st.info("No negative reviews available")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">Built with ❤️ using Streamlit | InsightGenAI © 2024</p>',
    unsafe_allow_html=True
)
