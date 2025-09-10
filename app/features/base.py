from abc import ABC, abstractmethod
from typing import Dict, Any
from app.dataclasses import Email

class BaseFeatureGenerator(ABC):
    """Base class for all feature generators"""
    
    @abstractmethod
    def generate_features(self, email: Email) -> Dict[str, Any]:
        """Generate features from Email dataclass"""
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names this generator produces"""
        pass