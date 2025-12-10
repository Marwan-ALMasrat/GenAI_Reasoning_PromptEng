import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class Visualizer:
    """
    Create interactive visualizations for data analysis
    """
    
    def __init__(self):
        self.color_scheme = {
            'positive': '#10b981',  # Green
            'neutral': '#6366f1',   # Blue
            'negative': '#ef4444'   # Red
        }
        
        self.template = 'plotly_white'
    
    def plot_sentiment_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Create an interactive pie chart of sentiment distribution
        
        Args:
            df: DataFrame with sentiment column
            
        Returns:
            Plotly figure
        """
        sentiment_counts = df['sentiment'].value_counts()
        
        colors = [self.color_scheme[sent] for sent in sentiment_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index.str.title(),
            values=sentiment_counts.values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textposition='auto',
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'Sentiment Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1f2937'}
            },
            showlegend=True,
            height=500,
            template=self.template
        )
        
        return fig
    
    def plot_rating_distribution(self, df: pd.DataFrame, rating_column: str) -> go.Figure:
        """
        Create a bar chart of rating distribution
        
        Args:
            df: DataFrame with rating column
            rating_column: Name of rating column
            
        Returns:
            Plotly figure
        """
        rating_counts = df[rating_column].value_counts().sort_index()
        
        fig = go.Figure(data=[go.Bar(
            x=rating_counts.index,
            y=rating_counts.values,
            marker=dict(
                color=rating_counts.values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Count')
            ),
            text=rating_counts.values,
            textposition='auto',
            hovertemplate='<b>Rating: %{x}</b><br>Count: %{y}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Rating Distribution',
            xaxis_title='Rating',
            yaxis_title='Number of Reviews',
            height=400,
            template=self.template,
            showlegend=False
        )
        
        return fig
    
    def plot_sentiment_timeline(self, df: pd.DataFrame, date_column: str) -> go.Figure:
        """
        Create a timeline showing sentiment trends over time
        
        Args:
            df: DataFrame with sentiment and date columns
            date_column: Name of date column
            
        Returns:
            Plotly figure
        """
        df_time = df.copy()
        df_time[date_column] = pd.to_datetime(df_time[date_column], errors='coerce')
        df_time = df_time.dropna(subset=[date_column])
        
        # Group by date and sentiment
        df_time['date'] = df_time[date_column].dt.date
        timeline = df_time.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig = go.Figure()
        
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_data = timeline[timeline['sentiment'] == sentiment]
            fig.add_trace(go.Scatter(
                x=sentiment_data['date'],
                y=sentiment_data['count'],
                name=sentiment.title(),
                mode='lines+markers',
                line=dict(color=self.color_scheme[sentiment], width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{sentiment.title()}</b><br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Sentiment Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Reviews',
            height=500,
            template=self.template,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig
    
    def plot_rating_sentiment_heatmap(self, df: pd.DataFrame, rating_column: str) -> go.Figure:
        """
        Create a heatmap showing rating vs sentiment correlation
        
        Args:
            df: DataFrame with rating and sentiment columns
            rating_column: Name of rating column
            
        Returns:
            Plotly figure
        """
        # Create cross-tabulation
        heatmap_data = pd.crosstab(df[rating_column], df['sentiment'])
        
        # Ensure all sentiments are present
        for sent in ['negative', 'neutral', 'positive']:
            if sent not in heatmap_data.columns:
                heatmap_data[sent] = 0
        
        heatmap_data = heatmap_data[['negative', 'neutral', 'positive']]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Negative', 'Neutral', 'Positive'],
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=heatmap_data.values,
            texttemplate='%{text}',
            textfont={"size": 14},
            hovertemplate='Rating: %{y}<br>Sentiment: %{x}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Rating vs Sentiment Heatmap',
            xaxis_title='Sentiment',
            yaxis_title='Rating',
            height=400,
            template=self.template
        )
        
        return fig
    
    def plot_sentiment_score_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Create violin plot of sentiment confidence scores
        
        Args:
            df: DataFrame with sentiment_score column
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_data = df[df['sentiment'] == sentiment]
            fig.add_trace(go.Violin(
                y=sentiment_data['sentiment_score'],
                name=sentiment.title(),
                box_visible=True,
                meanline_visible=True,
                fillcolor=self.color_scheme[sentiment],
                opacity=0.7,
                line_color=self.color_scheme[sentiment],
                hovertemplate=f'<b>{sentiment.title()}</b><br>Score: %{{y:.2%}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Sentiment Confidence Score Distribution',
            yaxis_title='Confidence Score',
            height=500,
            template=self.template,
            showlegend=True
        )
        
        fig.update_yaxis(tickformat='.0%')
        
        return fig
    
    def generate_wordcloud(self, df: pd.DataFrame, text_column: str) -> plt.Figure:
        """
        Generate word cloud from text data
        
        Args:
            df: DataFrame with text column
            text_column: Name of text column
            
        Returns:
            Matplotlib figure
        """
        # Combine all text
        text = ' '.join(df[text_column].dropna().astype(str).tolist())
        
        if not text.strip():
            # Return empty figure if no text
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No text available', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig
    
    def plot_review_length_analysis(self, df: pd.DataFrame, text_column: str) -> go.Figure:
        """
        Analyze review length by sentiment
        
        Args:
            df: DataFrame with text and sentiment columns
            text_column: Name of text column
            
        Returns:
            Plotly figure
        """
        df_length = df.copy()
        df_length['text_length'] = df_length[text_column].astype(str).str.len()
        
        fig = go.Figure()
        
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_data = df_length[df_length['sentiment'] == sentiment]
            fig.add_trace(go.Box(
                y=sentiment_data['text_length'],
                name=sentiment.title(),
                marker_color=self.color_scheme[sentiment],
                boxmean='sd',
                hovertemplate=f'<b>{sentiment.title()}</b><br>Length: %{{y}} chars<extra></extra>'
            ))
        
        fig.update_layout(
            title='Review Length Distribution by Sentiment',
            yaxis_title='Text Length (characters)',
            height=500,
            template=self.template,
            showlegend=True
        )
        
        return fig
    
    def create_dashboard_summary(self, df: pd.DataFrame, rating_column: str = None) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple subplots
        
        Args:
            df: DataFrame with analysis results
            rating_column: Optional rating column name
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Sentiment Confidence Scores',
                          'Rating Distribution' if rating_column else 'Review Length by Sentiment',
                          'Top Issues/Strengths'),
            specs=[[{'type': 'pie'}, {'type': 'box'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 1. Sentiment Distribution (Pie)
        sentiment_counts = df['sentiment'].value_counts()
        colors = [self.color_scheme[sent] for sent in sentiment_counts.index]
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index.str.title(),
                   values=sentiment_counts.values,
                   marker=dict(colors=colors)),
            row=1, col=1
        )
        
        # 2. Confidence Scores (Box)
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_data = df[df['sentiment'] == sentiment]
            fig.add_trace(
                go.Box(y=sentiment_data['sentiment_score'],
                       name=sentiment.title(),
                       marker_color=self.color_scheme[sentiment]),
                row=1, col=2
            )
        
        # 3. Rating or Length Distribution
        if rating_column and rating_column in df.columns:
            rating_counts = df[rating_column].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=rating_counts.index, y=rating_counts.values,
                       marker_color='#6366f1'),
                row=2, col=1
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            template=self.template,
            title_text='Sentiment Analysis Dashboard',
            title_x=0.5
        )
        
        return fig