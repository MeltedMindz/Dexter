"""
DexBrain Hub - Entry point for the DexBrain system

This module has been refactored. The components have been moved to:
- config.py: Configuration management
- blockchain/: Blockchain connectors
- models/: ML models and knowledge base
- core.py: Main DexBrain orchestrator

To use DexBrain:
    from dexbrain.core import DexBrain
    
    dexbrain = DexBrain()
    await dexbrain.run(blockchain='solana', pool_addresses=['pool1', 'pool2'])
"""

from .core import DexBrain, main

__all__ = ['DexBrain', 'main']
