from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseFeatureGenerator(ABC):
    """Base class for all feature generators"""
    
    @abstractmethod
    def generate_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features from raw input data"""
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names this generator produces"""
        pass