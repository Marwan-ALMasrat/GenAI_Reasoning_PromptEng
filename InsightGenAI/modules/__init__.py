"""
InsightGenAI - AI-Powered Data Analysis & Sentiment Insights Platform

This package contains core modules for data processing, NLP analysis,
AI insights generation, and visualization.
"""

from .data_loader import DataLoader
from .data_analyzer import DataAnalyzer
from .nlp_processor import NLPProcessor
from .genai_insights import GenAIInsights
from .visualizer import Visualizer

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

__all__ = [
    'DataLoader',
    'DataAnalyzer', 
    'NLPProcessor',
    'GenAIInsights',
    'Visualizer'
]
