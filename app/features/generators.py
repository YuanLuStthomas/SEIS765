from typing import Dict, Any
from .base import BaseFeatureGenerator
from app.dataclasses import Email

class SpamFeatureGenerator(BaseFeatureGenerator):
    """Generates spam detection features from email content"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
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
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
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
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        
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
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        
        return {
            "email_subject": subject,
            "email_body": body
        }
    
    @property
    def feature_names(self) -> list[str]:
        return ["email_subject", "email_body"]


# TODO: LAB ASSIGNMENT - Part 1 of 2
# Create a NonTextCharacterFeatureGenerator class that counts non-alphanumeric characters
# (punctuation, symbols, etc.) in the email subject and body text.
# 
# Requirements:
# 1. Inherit from BaseFeatureGenerator
# 2. Implement generate_features(self, email: Email) -> Dict[str, Any]:
#    - Count non-alphanumeric characters in both subject and body
#    - Return a dictionary with "non_text_char_count" as the key
# 3. Implement the feature_names property to return ["non_text_char_count"]
# 4. Follow the same pattern as the other generators above
# 
# Hint: Use email.subject and email.body to access the text
# Hint: You can use regex or string methods to count non-alphanumeric characters