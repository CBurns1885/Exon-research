"""LLM-powered sentiment analysis for central bank communications."""
import json
import logging
from typing import Dict, List, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze central bank speech sentiment using LLMs."""

    def __init__(self, provider: str = None, model: str = None):
        """Initialize sentiment analyzer with LLM provider."""
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL
        self.client = None

        self._setup_llm()

    def _setup_llm(self):
        """Set up LLM client based on provider."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                logger.info(f"Initialized OpenAI client with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.client = None

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                logger.info(f"Initialized Anthropic client with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
                self.client = None

        elif self.provider == "local":
            # Use local transformers model
            try:
                from transformers import pipeline
                self.client = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                logger.info("Initialized local FinBERT model")
            except Exception as e:
                logger.error(f"Failed to initialize local model: {e}")
                self.client = None

    def analyze_speech(self, text: str, title: str = "", speaker: str = "") -> Dict:
        """
        Analyze central bank speech for monetary policy sentiment.

        Args:
            text: Full speech text
            title: Speech title
            speaker: Speaker name

        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.client:
            logger.warning("No LLM client available, using fallback")
            return self._fallback_analysis(text)

        try:
            if self.provider in ["openai", "anthropic"]:
                return self._llm_analysis(text, title, speaker)
            else:
                return self._local_analysis(text)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._fallback_analysis(text)

    def _llm_analysis(self, text: str, title: str, speaker: str) -> Dict:
        """Perform sentiment analysis using GPT-4 or Claude."""

        # Truncate text if too long (to fit in context window)
        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        prompt = f"""Analyze this central bank speech for monetary policy sentiment and provide structured insights.

Speech Title: {title}
Speaker: {speaker}

Speech Text:
{text}

Provide your analysis in the following JSON format:

{{
    "sentiment": "<one of: very_dovish, dovish, neutral, hawkish, very_hawkish>",
    "sentiment_score": <number from -1 (very dovish) to +1 (very hawkish)>,
    "confidence": <number from 0 to 1 indicating confidence in assessment>,
    "key_phrases": [<list of 3-5 key phrases that signal policy stance>],
    "main_topics": [<list of 3-5 main topics discussed>],
    "policy_signals": {{
        "inflation_concern": "<low/medium/high>",
        "growth_concern": "<low/medium/high>",
        "rate_path_hint": "<likely to hike/hold/cut or unclear>",
        "data_dependency": "<high/medium/low>",
        "forward_guidance": "<specific guidance mentioned or none>"
    }},
    "summary": "<2-3 sentence summary of the key policy message>"
}}

Definitions:
- very_dovish: Strong preference for lower rates, economic stimulus
- dovish: Leans toward lower rates or patience
- neutral: Balanced, data-dependent, no clear bias
- hawkish: Leans toward higher rates or tightening
- very_hawkish: Strong preference for higher rates, inflation fighting

Return only valid JSON, no additional text."""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert central bank analyst specializing in monetary policy interpretation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model if "claude" in self.model else "claude-3-opus-20240229",
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = response.content[0].text

            # Parse JSON response
            result = json.loads(result_text)

            # Convert arrays to JSON strings for database storage
            result['key_phrases'] = json.dumps(result.get('key_phrases', []))
            result['main_topics'] = json.dumps(result.get('main_topics', []))
            result['policy_signals'] = json.dumps(result.get('policy_signals', {}))

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return self._fallback_analysis(text)
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return self._fallback_analysis(text)

    def _local_analysis(self, text: str) -> Dict:
        """Perform sentiment analysis using local FinBERT model."""
        try:
            # Use FinBERT for financial sentiment
            max_chars = 500  # FinBERT has limited context
            chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)][:5]

            sentiments = []
            for chunk in chunks:
                result = self.client(chunk)[0]
                sentiments.append(result)

            # Aggregate results
            avg_score = sum([s['score'] if s['label'] == 'positive' else -s['score']
                           for s in sentiments]) / len(sentiments)

            # Map to central bank sentiment
            if avg_score > 0.3:
                sentiment = "hawkish"
            elif avg_score > 0.1:
                sentiment = "neutral"
            else:
                sentiment = "dovish"

            return {
                "sentiment": sentiment,
                "sentiment_score": avg_score,
                "confidence": sum([s['score'] for s in sentiments]) / len(sentiments),
                "key_phrases": json.dumps([]),
                "main_topics": json.dumps([]),
                "policy_signals": json.dumps({}),
            }

        except Exception as e:
            logger.error(f"Local analysis error: {e}")
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> Dict:
        """Simple keyword-based fallback analysis."""
        text_lower = text.lower()

        # Hawkish keywords
        hawkish_words = [
            'inflation', 'tighten', 'hike', 'raise', 'increase',
            'restrictive', 'vigilant', 'firm', 'determined'
        ]

        # Dovish keywords
        dovish_words = [
            'patient', 'gradual', 'support', 'accommodate',
            'growth', 'employment', 'lower', 'cut', 'reduce'
        ]

        hawkish_count = sum(1 for word in hawkish_words if word in text_lower)
        dovish_count = sum(1 for word in dovish_words if word in text_lower)

        total = hawkish_count + dovish_count
        if total == 0:
            sentiment = "neutral"
            score = 0.0
        else:
            score = (hawkish_count - dovish_count) / total
            if score > 0.3:
                sentiment = "hawkish"
            elif score < -0.3:
                sentiment = "dovish"
            else:
                sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "sentiment_score": score,
            "confidence": 0.5,  # Low confidence for fallback
            "key_phrases": json.dumps([]),
            "main_topics": json.dumps([]),
            "policy_signals": json.dumps({}),
        }

    def batch_analyze(self, speeches: List[Dict]) -> List[Dict]:
        """Analyze multiple speeches in batch."""
        results = []

        for speech in speeches:
            logger.info(f"Analyzing: {speech.get('title', 'Untitled')}")

            result = self.analyze_speech(
                text=speech.get('text_content', ''),
                title=speech.get('title', ''),
                speaker=speech.get('speaker', '')
            )

            results.append({
                "speech_id": speech.get('id'),
                **result
            })

        return results
