"""RESTful API Server for DexBrain Global Intelligence Network"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from .core import DexBrain
from .auth import APIKeyManager, RateLimiter, require_api_key
from .models.knowledge_base import KnowledgeBase
from .models.ml_models import DeFiMLEngine
from .config import Config


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
dex_brain = DexBrain()
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter(requests_per_minute=100)
knowledge_base = KnowledgeBase()

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/register', methods=['POST'])
def register_agent():
    """Register a new agent and get API key"""
    data = request.get_json()
    
    if not data or 'agent_id' not in data:
        return jsonify({'error': 'agent_id required'}), 400
    
    agent_id = data['agent_id']
    metadata = data.get('metadata', {})
    
    # Add registration metadata
    metadata.update({
        'ip_address': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', ''),
        'registration_time': datetime.now().isoformat()
    })
    
    try:
        api_key = api_key_manager.generate_api_key(agent_id, metadata)
        
        logger.info(f"New agent registered: {agent_id}")
        
        return jsonify({
            'api_key': api_key,
            'agent_id': agent_id,
            'message': 'Agent registered successfully'
        })
        
    except Exception as e:
        logger.error(f"Agent registration failed: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/intelligence', methods=['GET'])
@require_api_key(rate_limiter, api_key_manager)
def get_intelligence(**kwargs):
    """Get market intelligence and predictions"""
    agent_info = kwargs.get('agent_info')
    
    # Get query parameters
    blockchain = request.args.get('blockchain', 'base')
    pool_address = request.args.get('pool_address')
    category = request.args.get('category', f'{blockchain}_liquidity')
    limit = min(int(request.args.get('limit', 100)), 1000)
    
    try:
        # Get recent insights
        insights = asyncio.run(
            knowledge_base.retrieve_insights(category, limit=limit)
        )
        
        # Get predictions if pool_address provided
        predictions = None
        if pool_address:
            # This would be expanded with actual pool data fetching
            pool_data = {
                'pool_address': pool_address,
                'blockchain': blockchain,
                'total_liquidity': 1000000,  # Placeholder
                'volume_24h': 500000,
                'fee_tier': 0.003,
                'token0_reserves': 500000,
                'token1_reserves': 500000
            }
            
            predicted_apr = asyncio.run(
                dex_brain.predict_liquidity_metrics(pool_data)
            )
            
            predictions = {
                'pool_address': pool_address,
                'predicted_apr': predicted_apr,
                'confidence': 0.85,  # Would be calculated based on model metrics
                'prediction_time': datetime.now().isoformat()
            }
        
        # Get network statistics
        agent_count = len(api_key_manager.list_agents())
        insight_count = asyncio.run(knowledge_base.get_insight_count(category))
        
        response = {
            'insights': insights,
            'predictions': predictions,
            'network_stats': {
                'total_agents': agent_count,
                'total_insights': insight_count,
                'blockchain': blockchain,
                'category': category
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Intelligence served to agent {agent_info['agent_id']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Intelligence request failed: {e}")
        return jsonify({'error': 'Failed to retrieve intelligence'}), 500


@app.route('/api/submit-data', methods=['POST'])
@require_api_key(rate_limiter, api_key_manager)
def submit_data(**kwargs):
    """Submit performance data from agent"""
    agent_info = kwargs.get('agent_info')
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    required_fields = ['blockchain', 'pool_address', 'performance_data']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
    
    try:
        # Enrich data with agent information
        enhanced_data = {
            'agent_id': agent_info['agent_id'],
            'submission_time': datetime.now().isoformat(),
            'blockchain': data['blockchain'],
            'pool_address': data['pool_address'],
            **data['performance_data']
        }
        
        # Store in knowledge base
        category = f"{data['blockchain']}_liquidity"
        asyncio.run(
            knowledge_base.store_insight(category, enhanced_data)
        )
        
        # Record the submission
        api_key_manager.record_data_submission(request.headers.get('Authorization', '').replace('Bearer ', ''))
        
        logger.info(f"Data submitted by agent {agent_info['agent_id']} for {data['pool_address']}")
        
        return jsonify({
            'status': 'success',
            'message': 'Data submitted successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Data submission failed: {e}")
        return jsonify({'error': 'Failed to submit data'}), 500


@app.route('/api/agents', methods=['GET'])
@require_api_key(rate_limiter, api_key_manager)
def list_agents(**kwargs):
    """List all registered agents (admin endpoint)"""
    try:
        agents = api_key_manager.list_agents()
        
        # Remove sensitive information
        public_agents = {}
        for agent_id, info in agents.items():
            public_agents[agent_id] = {
                'agent_id': info['agent_id'],
                'created_at': info['created_at'],
                'last_used': info['last_used'],
                'request_count': info['request_count'],
                'data_submissions': info['data_submissions'],
                'is_active': info['is_active']
            }
        
        return jsonify({
            'agents': public_agents,
            'total_count': len(public_agents),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Agent listing failed: {e}")
        return jsonify({'error': 'Failed to list agents'}), 500


@app.route('/api/stats', methods=['GET'])
def get_network_stats():
    """Get public network statistics"""
    try:
        agents = api_key_manager.list_agents()
        active_agents = sum(1 for agent in agents.values() if agent.get('is_active', False))
        
        # Get insights count across categories
        categories = asyncio.run(knowledge_base.get_categories())
        total_insights = 0
        category_counts = {}
        
        for category in categories:
            count = asyncio.run(knowledge_base.get_insight_count(category))
            category_counts[category] = count
            total_insights += count
        
        # Calculate recent activity (last 24 hours)
        recent_submissions = 0
        for agent in agents.values():
            last_used = agent.get('last_used')
            if last_used:
                last_used_dt = datetime.fromisoformat(last_used)
                if datetime.now() - last_used_dt < timedelta(hours=24):
                    recent_submissions += agent.get('data_submissions', 0)
        
        return jsonify({
            'network_stats': {
                'total_agents': len(agents),
                'active_agents': active_agents,
                'total_insights': total_insights,
                'recent_submissions_24h': recent_submissions,
                'supported_blockchains': ['base', 'solana', 'ethereum'],
                'categories': category_counts
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        return jsonify({'error': 'Failed to get network stats'}), 500


@app.route('/api/models/retrain', methods=['POST'])
@require_api_key(rate_limiter, api_key_manager)
def retrain_models(**kwargs):
    """Trigger model retraining (admin endpoint)"""
    agent_info = kwargs.get('agent_info')
    data = request.get_json()
    
    category = data.get('category', 'base_liquidity') if data else 'base_liquidity'
    
    try:
        # Trigger model retraining
        training_results = asyncio.run(dex_brain.train_models(category))
        
        logger.info(f"Model retraining triggered by agent {agent_info['agent_id']}")
        
        return jsonify({
            'status': 'success',
            'training_results': training_results,
            'category': category,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        return jsonify({'error': 'Failed to retrain models'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=Config.API_PORT if hasattr(Config, 'API_PORT') else 8080,
        debug=Config.DEBUG if hasattr(Config, 'DEBUG') else False
    )