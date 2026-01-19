"""Machine learning models and knowledge base for DexBrain"""

from .knowledge_base import KnowledgeBase

# Conditionally import ML models (requires torch)
try:
    from .ml_models import DeFiMLEngine, LiquidityPredictionModel
    ML_AVAILABLE = True
except ImportError:
    # Provide stub implementations when torch is not available
    ML_AVAILABLE = False
    DeFiMLEngine = None
    LiquidityPredictionModel = None

__all__ = ['KnowledgeBase', 'DeFiMLEngine', 'LiquidityPredictionModel', 'ML_AVAILABLE']