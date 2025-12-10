import requests
import json
from typing import Dict, Optional, List

class GenAIInsights:
    """
    Generate AI-powered insights using Chain-of-Thought reasoning
    """
    
    def __init__(self):
        self.api_key = None
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "openai/gpt-4o-2024-08-06"  # Upgraded model for better reasoning
        
    def set_api_key(self, api_key: str):
        """Set the OpenRouter API key"""
        self.api_key = api_key
    
    def _make_api_call(self, messages: List[Dict], enable_reasoning: bool = True) -> Optional[Dict]:
        """
        Make API call to OpenRouter
        
        Args:
            messages: List of message dictionaries
            enable_reasoning: Whether to enable Chain-of-Thought reasoning
            
        Returns:
            Response dictionary or None if error
        """
        if not self.api_key:
            print("Error: API key not set")
            return None
        
        payload = {
            "model": self.model,
            "messages": messages
        }
        
        # Enable reasoning for supported models
        if enable_reasoning:
            payload["reasoning"] = {"enabled": True}
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps(payload),
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API call error: {str(e)}")
            return None
    
    def analyze_review_chain_of_thought(self, review_text: str, 
                                       sentiment: str, 
                                       sentiment_score: float) -> Optional[Dict]:
        """
        Analyze a review using multi-step Chain-of-Thought reasoning
        
        Args:
            review_text: The review text
            sentiment: Detected sentiment (positive/neutral/negative)
            sentiment_score: Confidence score
            
        Returns:
            Dictionary with analysis and recommendations
        """
        # Step 1: Detailed Analysis with Chain-of-Thought
        step1_messages = [
            {
                "role": "user",
                "content": f"""You are an expert business analyst specializing in customer feedback analysis.

Review Text:
{review_text}

Detected Sentiment: {sentiment}
Confidence Score: {sentiment_score:.2%}

Please perform a thorough analysis of this review using step-by-step reasoning:

1. IDENTIFY KEY THEMES:
   - What are the main topics mentioned? (service, product quality, pricing, experience, etc.)
   - Break down each theme separately

2. ANALYZE SENTIMENT FOR EACH THEME:
   - For each identified theme, determine if the customer's feeling is positive, neutral, or negative
   - Provide specific evidence from the text

3. DETECT CRITICAL ISSUES:
   - Are there any urgent problems that need immediate attention?
   - What is the severity level? (Low/Medium/High/Critical)

4. IDENTIFY STRENGTHS:
   - What did the business do well according to this review?
   - What should they continue doing?

5. SUMMARIZE INSIGHTS:
   - Provide a clear, structured summary of your analysis

Format your response with clear sections and bullet points."""
            }
        ]
        
        response1 = self._make_api_call(step1_messages, enable_reasoning=True)
        
        if not response1 or 'choices' not in response1:
            return None
        
        analysis_content = response1['choices'][0]['message'].get('content', '')
        reasoning_details = response1['choices'][0]['message'].get('reasoning_details')
        
        # Step 2: Generate Actionable Recommendations
        step2_messages = [
            {
                "role": "user",
                "content": f"""Review Text:
{review_text}

Sentiment: {sentiment}
Confidence: {sentiment_score:.2%}
"""
            },
            {
                "role": "assistant",
                "content": analysis_content
            },
            {
                "role": "user",
                "content": """Based on your detailed analysis, now provide SPECIFIC and ACTIONABLE recommendations for the business.

Structure your recommendations as follows:

**IMMEDIATE ACTIONS** (Within 24-48 hours):
- [List 2-3 urgent actions with clear steps]

**SHORT-TERM IMPROVEMENTS** (Within 1-2 weeks):
- [List 3-4 tactical improvements]

**LONG-TERM STRATEGIES** (Within 1-3 months):
- [List 2-3 strategic initiatives]

**KEY PERFORMANCE INDICATORS TO TRACK**:
- [List metrics to monitor improvement]

Make each recommendation:
✓ Specific (what exactly to do)
✓ Measurable (how to track success)
✓ Actionable (clear next steps)
✓ Realistic (achievable with available resources)

Be direct and concise. No fluff."""
            }
        ]
        
        # Add reasoning details if available
        if reasoning_details:
            step2_messages[1]['reasoning_details'] = reasoning_details
        
        response2 = self._make_api_call(step2_messages, enable_reasoning=True)
        
        if not response2 or 'choices' not in response2:
            return {
                'analysis': analysis_content,
                'recommendations': "Error generating recommendations"
            }
        
        recommendations_content = response2['choices'][0]['message'].get('content', '')
        
        return {
            'analysis': analysis_content,
            'recommendations': recommendations_content,
            'reasoning_trace': reasoning_details  # Include reasoning for transparency
        }
    
    def analyze_multiple_reviews_batch(self, reviews_df, sample_size: int = 5) -> Dict:
        """
        Analyze multiple reviews and provide aggregated insights
        
        Args:
            reviews_df: DataFrame with reviews
            sample_size: Number of reviews to analyze
            
        Returns:
            Dictionary with aggregated insights
        """
        # Sample reviews from different sentiments
        positive_sample = reviews_df[reviews_df['sentiment'] == 'positive'].head(2)
        negative_sample = reviews_df[reviews_df['sentiment'] == 'negative'].head(2)
        neutral_sample = reviews_df[reviews_df['sentiment'] == 'neutral'].head(1)
        
        samples = pd.concat([positive_sample, negative_sample, neutral_sample])
        
        # Prepare batch analysis
        reviews_text = "\n\n---\n\n".join([
            f"Review {i+1} (Sentiment: {row['sentiment']}, Score: {row['sentiment_score']:.2%}):\n{row['text']}"
            for i, (_, row) in enumerate(samples.iterrows())
        ])
        
        messages = [
            {
                "role": "user",
                "content": f"""You are a senior business consultant analyzing customer feedback patterns.

Here are {len(samples)} customer reviews from different sentiment categories:

{reviews_text}

Please provide a comprehensive strategic analysis:

1. **PATTERN IDENTIFICATION**:
   - What common themes appear across multiple reviews?
   - Are there recurring complaints or praises?
   - What patterns emerge by sentiment type?

2. **ROOT CAUSE ANALYSIS**:
   - What are the underlying causes of negative feedback?
   - What drives positive experiences?
   - Are there systemic issues vs. isolated incidents?

3. **COMPETITIVE INSIGHTS**:
   - What do these reviews reveal about market positioning?
   - What are customers comparing the business to?
   - Where are competitive advantages and disadvantages?

4. **STRATEGIC RECOMMENDATIONS**:
   - Top 5 priority actions for business improvement
   - Resource allocation suggestions
   - Expected impact of each recommendation

5. **RISK ASSESSMENT**:
   - What risks does the business face based on this feedback?
   - What opportunities are being missed?

Provide executive-level insights that can drive real business decisions."""
            }
        ]
        
        response = self._make_api_call(messages, enable_reasoning=True)
        
        if response and 'choices' in response:
            return {
                'aggregated_insights': response['choices'][0]['message'].get('content', ''),
                'sample_size': len(samples),
                'sentiment_distribution': samples['sentiment'].value_counts().to_dict()
            }
        
        return None
    
    def generate_executive_summary(self, reviews_df, sentiment_stats: Dict) -> Optional[str]:
        """
        Generate executive summary of all customer feedback
        
        Args:
            reviews_df: Complete DataFrame with reviews
            sentiment_stats: Sentiment statistics dictionary
            
        Returns:
            Executive summary text
        """
        # Prepare summary statistics
        stats_text = f"""
Total Reviews Analyzed: {sentiment_stats['total_reviews']:,}

Sentiment Distribution:
- Positive: {sentiment_stats['sentiment_counts'].get('positive', 0):,} ({sentiment_stats['sentiment_percentages'].get('positive', 0):.1f}%)
- Neutral: {sentiment_stats['sentiment_counts'].get('neutral', 0):,} ({sentiment_stats['sentiment_percentages'].get('neutral', 0):.1f}%)
- Negative: {sentiment_stats['sentiment_counts'].get('negative', 0):,} ({sentiment_stats['sentiment_percentages'].get('negative', 0):.1f}%)

Average Confidence Score: {sentiment_stats['average_sentiment_score']:.2%}
"""
        
        messages = [
            {
                "role": "user",
                "content": f"""You are a Chief Analytics Officer presenting to the C-suite.

{stats_text}

Create a concise EXECUTIVE SUMMARY (max 300 words) that includes:

1. **OVERALL HEALTH SCORE**: Rate the business health (1-10) based on sentiment
2. **KEY FINDINGS**: Top 3 most important insights
3. **CRITICAL ACTIONS**: Top 3 immediate priorities
4. **BUSINESS IMPACT**: Expected outcomes if actions are taken
5. **NEXT STEPS**: What the executive team should do this week

Write in a professional, data-driven tone suitable for executive presentation."""
            }
        ]
        
        response = self._make_api_call(messages, enable_reasoning=False)
        
        if response and 'choices' in response:
            return response['choices'][0]['message'].get('content', '')
        
        return None
