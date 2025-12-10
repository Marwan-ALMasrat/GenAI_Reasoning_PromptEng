import pandas as pd
import numpy as np
import re
import string
from typing import Optional, Callable
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class NLPProcessor:
    """
    Handles text cleaning, preprocessing, and sentiment analysis
    """
    
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def initialize_model(self):
        """Load sentiment analysis model"""
        if self.model is None:
            print("Loading sentiment analysis model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Check for GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            print(f"Model loaded on: {self.device}")
    
    def clean_text_basic(self, text: str) -> str:
        """
        Basic text cleaning: remove newlines, extra spaces
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove newlines and replace with space
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def clean_text_advanced(self, text: str) -> str:
        """
        Advanced text cleaning: lowercase, remove punctuation, numbers
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove time indicators (am/pm)
        text = re.sub(r'\b(am|pm|AM|PM)\b', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def clean_text_pipeline(self, df: pd.DataFrame, text_column: str, 
                           keep_original: bool = True) -> pd.DataFrame:
        """
        Apply complete text cleaning pipeline
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to clean
            keep_original: Whether to keep original text column
            
        Returns:
            DataFrame with cleaned text
        """
        df_clean = df.copy()
        
        # Basic cleaning (keep for sentiment analysis)
        df_clean['clean_text'] = df_clean[text_column].apply(self.clean_text_basic)
        
        # Advanced cleaning (for visualization/analysis)
        df_clean['clean_text_advanced'] = df_clean['clean_text'].apply(self.clean_text_advanced)
        
        return df_clean
    
    def get_sentiment(self, text: str) -> tuple:
        """
        Analyze sentiment for a single text
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if pd.isna(text) or text == "":
            return "neutral", 0.0
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            scores = F.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
            labels = ["negative", "neutral", "positive"]
            
            sentiment = labels[np.argmax(scores)]
            score = float(np.max(scores))
            
            return sentiment, score
        
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return "neutral", 0.0
    
    def analyze_sentiment_batched(self, df: pd.DataFrame, text_column: str,
                                  batch_size: int = 200, 
                                  progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Analyze sentiment in batches for efficiency
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            batch_size: Number of rows per batch
            progress_callback: Optional callback function for progress updates
            
        Returns:
            DataFrame with sentiment columns added
        """
        # Initialize model if not already loaded
        self.initialize_model()
        
        df_sentiment = df.copy()
        total_rows = len(df_sentiment)
        num_batches = (total_rows + batch_size - 1) // batch_size
        
        sentiments = []
        scores = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_rows)
            
            batch_texts = df_sentiment[text_column].iloc[start_idx:end_idx]
            
            # Process each text in batch
            batch_results = [self.get_sentiment(text) for text in batch_texts]
            batch_sentiments, batch_scores = zip(*batch_results)
            
            sentiments.extend(batch_sentiments)
            scores.extend(batch_scores)
            
            # Update progress
            if progress_callback:
                progress = (end_idx / total_rows)
                progress_callback(progress)
            
            print(f"Processed batch {batch_idx + 1}/{num_batches} ({end_idx}/{total_rows} rows)")
        
        df_sentiment['sentiment'] = sentiments
        df_sentiment['sentiment_score'] = scores
        
        return df_sentiment
    
    def detect_mismatches(self, df: pd.DataFrame, rating_column: str,
                         threshold_high: float = 4.0,
                         threshold_low: float = 2.0) -> pd.DataFrame:
        """
        Detect mismatches between rating and sentiment
        
        Args:
            df: DataFrame with sentiment and rating columns
            rating_column: Name of rating column
            threshold_high: High rating threshold
            threshold_low: Low rating threshold
            
        Returns:
            DataFrame with mismatch flags
        """
        df_mismatch = df.copy()
        
        # High rating but negative sentiment
        df_mismatch['mismatch_high_negative'] = (
            (df_mismatch[rating_column] >= threshold_high) & 
            (df_mismatch['sentiment'] == 'negative')
        )
        
        # Low rating but positive sentiment
        df_mismatch['mismatch_low_positive'] = (
            (df_mismatch[rating_column] <= threshold_low) & 
            (df_mismatch['sentiment'] == 'positive')
        )
        
        # Overall mismatch flag
        df_mismatch['has_mismatch'] = (
            df_mismatch['mismatch_high_negative'] | 
            df_mismatch['mismatch_low_positive']
        )
        
        return df_mismatch
    
    def remove_mismatches(self, df: pd.DataFrame, rating_column: str) -> pd.DataFrame:
        """
        Remove rows with sentiment-rating mismatches
        
        Args:
            df: DataFrame with sentiment and rating
            rating_column: Name of rating column
            
        Returns:
            Filtered DataFrame
        """
        df_filtered = self.detect_mismatches(df, rating_column)
        
        initial_count = len(df_filtered)
        df_filtered = df_filtered[~df_filtered['has_mismatch']].copy()
        removed_count = initial_count - len(df_filtered)
        
        print(f"Removed {removed_count} mismatched rows ({removed_count/initial_count*100:.2f}%)")
        
        # Drop mismatch columns
        df_filtered = df_filtered.drop(
            columns=['mismatch_high_negative', 'mismatch_low_positive', 'has_mismatch'],
            errors='ignore'
        )
        
        return df_filtered.reset_index(drop=True)
    
    def get_sentiment_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate sentiment statistics
        
        Args:
            df: DataFrame with sentiment column
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_reviews': len(df),
            'sentiment_counts': df['sentiment'].value_counts().to_dict(),
            'sentiment_percentages': (df['sentiment'].value_counts() / len(df) * 100).to_dict(),
            'average_sentiment_score': df['sentiment_score'].mean(),
            'median_sentiment_score': df['sentiment_score'].median()
        }
        
        # Confidence statistics by sentiment
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_df = df[df['sentiment'] == sentiment]
            if len(sentiment_df) > 0:
                stats[f'{sentiment}_avg_confidence'] = sentiment_df['sentiment_score'].mean()
        
        return stats