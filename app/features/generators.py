from typing import Dict, Any
from .base import BaseFeatureGenerator

class SpamFeatureGenerator(BaseFeatureGenerator):
    """Generates spam detection features from email content"""
    
    def generate_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract email content - expect 'subject' and 'body' fields
        subject = str(raw_data.get('subject', ''))
        body = str(raw_data.get('body', ''))
        all_text = f"{subject} {body}".lower()
        
        # Spam word detection
        spam_words = ['free', 'winner', 'congratulations', 'click here', 'limited time', 
                      'act now', 'urgent', 'special offer', 'guaranteed', 'no risk',
                      'cash', 'money back', 'amazing', 'incredible', 'unbeatable']
        
        has_spam_words = int(any(word in all_text for word in spam_words))
        
        return {"has_spam_words": has_spam_words}
    
    @property
    def feature_names(self) -> list[str]:
        return ["has_spam_words"]


class AverageWordLengthFeatureGenerator(BaseFeatureGenerator):
    """Generates average word length feature from email content"""
    
    def generate_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract email content - expect 'subject' and 'body' fields
        subject = str(raw_data.get('subject', ''))
        body = str(raw_data.get('body', ''))
        all_text = f"{subject} {body}"
        
        # Split into words and calculate average length
        words = all_text.split()
        if not words:
            average_word_length = 0.0
        else:
            total_length = sum(len(word) for word in words)
            average_word_length = total_length / len(words)
        
        return {"average_word_length": average_word_length}
    
    @property
    def feature_names(self) -> list[str]:
        return ["average_word_length"]


class EmailEmbeddingsFeatureGenerator(BaseFeatureGenerator):
    """Generates embedding features by averaging title and detail lengths"""
    
    def generate_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract email content - expect 'subject' and 'body' fields
        subject = str(raw_data.get('subject', ''))
        body = str(raw_data.get('body', ''))
        
        # Calculate lengths
        title_length = len(subject)
        detail_length = len(body)
        
        # Average of title and detail lengths
        if title_length == 0 and detail_length == 0:
            average_embedding = 0.0
        else:
            average_embedding = (title_length + detail_length) / 2
        
        return {"average_embedding": average_embedding}
    
    @property
    def feature_names(self) -> list[str]:
        return ["average_embedding"]


class RawEmailFeatureGenerator(BaseFeatureGenerator):
    """Extracts raw email data as features"""
    
    def generate_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract email content - expect 'subject' and 'body' fields
        subject = str(raw_data.get('subject', ''))
        body = str(raw_data.get('body', ''))
        
        return {
            "email_subject": subject,
            "email_body": body
        }
    
    @property
    def feature_names(self) -> list[str]:
        return ["email_subject", "email_body"]