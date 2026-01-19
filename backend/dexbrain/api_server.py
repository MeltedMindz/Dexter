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
from .models import DeFiMLEngine, ML_AVAILABLE
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
        
        # Get vault statistics if available
        vault_stats = {}
        if hasattr(dex_brain, 'vault_engine') and dex_brain.vault_engine:
            vault_stats = {
                'vault_integration': True,
                'ai_strategies_active': True,
                'compound_service_active': hasattr(dex_brain, 'compound_service') and dex_brain.compound_service is not None
            }
        else:
            vault_stats = {
                'vault_integration': False,
                'ai_strategies_active': False,
                'compound_service_active': False
            }
        
        return jsonify({
            'network_stats': {
                'total_agents': len(agents),
                'active_agents': active_agents,
                'total_insights': total_insights,
                'recent_submissions_24h': recent_submissions,
                'supported_blockchains': ['base', 'solana', 'ethereum'],
                'categories': category_counts,
                'vault_stats': vault_stats
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


@app.route('/api/vault/intelligence', methods=['GET'])
@require_api_key(rate_limiter, api_key_manager)
def get_vault_intelligence(**kwargs):
    """Get AI-powered vault intelligence and strategy recommendations"""
    agent_info = kwargs.get('agent_info')
    
    # Get query parameters
    vault_address = request.args.get('vault_address')
    
    if not vault_address:
        return jsonify({'error': 'vault_address parameter required'}), 400
    
    try:
        # Mock pool and vault data for demonstration
        # In production, this would fetch from actual sources
        pool_data = {
            'current_tick': 100000,
            'current_price': 3000,
            'liquidity': 1000000,
            'volume_24h': 5000000,
            'fee_tier': 3000,
            'tick_spacing': 60,
            'prices': list(range(2900, 3100, 10))
        }
        
        vault_metrics = {
            'total_value_locked': 2000000,
            'total_fees_24h': 5000,
            'impermanent_loss': 0.02,
            'apr': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05,
            'successful_compounds': 45,
            'ai_optimization_count': 12,
            'capital_efficiency': 0.85,
            'risk_score': 0.3
        }
        
        # Generate vault intelligence
        intelligence = asyncio.run(
            dex_brain.generate_vault_intelligence(
                vault_address, pool_data, vault_metrics
            )
        )
        
        logger.info(f"Vault intelligence served to agent {agent_info['agent_id']} for vault {vault_address}")
        
        return jsonify({
            'vault_intelligence': intelligence,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Vault intelligence request failed: {e}")
        return jsonify({'error': 'Failed to generate vault intelligence'}), 500


@app.route('/api/vault/compound-opportunities', methods=['GET'])
@require_api_key(rate_limiter, api_key_manager)
def get_compound_opportunities(**kwargs):
    """Get current compound opportunities from the compound service"""
    agent_info = kwargs.get('agent_info')
    
    # Get query parameters
    limit = min(int(request.args.get('limit', 10)), 50)
    min_profit = float(request.args.get('min_profit', 5.0))
    
    try:
        if not hasattr(dex_brain, 'compound_service') or not dex_brain.compound_service:
            return jsonify({
                'error': 'Compound service not available',
                'opportunities': []
            }), 503
        
        # Get compound opportunities
        opportunities = asyncio.run(
            dex_brain.compound_service.find_compound_opportunities(
                max_positions=limit,
                min_profit_usd=min_profit
            )
        )
        
        # Convert to serializable format
        opportunities_data = []
        for opp in opportunities:
            opportunities_data.append({
                'token_id': opp.token_id,
                'owner': opp.owner,
                'current_fees_usd': opp.current_fees_usd,
                'estimated_gas_cost': opp.estimated_gas_cost,
                'profit_potential': opp.profit_potential,
                'optimal_timing_score': opp.optimal_timing_score,
                'strategy': opp.strategy.value,
                'urgency_score': opp.urgency_score,
                'risk_score': opp.risk_score,
                'ai_confidence': opp.ai_confidence,
                'estimated_apr_improvement': opp.estimated_apr_improvement,
                'priority_score': opp.priority_score
            })
        
        logger.info(f"Compound opportunities served to agent {agent_info['agent_id']}: {len(opportunities_data)} opportunities")
        
        return jsonify({
            'opportunities': opportunities_data,
            'total_count': len(opportunities_data),
            'filters': {
                'limit': limit,
                'min_profit': min_profit
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Compound opportunities request failed: {e}")
        return jsonify({'error': 'Failed to get compound opportunities'}), 500


@app.route('/api/vault/analytics', methods=['GET'])
@require_api_key(rate_limiter, api_key_manager)
def get_vault_analytics(**kwargs):
    """Get comprehensive vault analytics and performance metrics"""
    agent_info = kwargs.get('agent_info')
    
    # Get query parameters
    days = min(int(request.args.get('days', 30)), 90)
    
    try:
        if not hasattr(dex_brain, 'compound_service') or not dex_brain.compound_service:
            return jsonify({
                'error': 'Compound service not available',
                'analytics': {}
            }), 503
        
        # Get compound analytics
        analytics = asyncio.run(
            dex_brain.compound_service.get_compound_analytics(days=days)
        )
        
        logger.info(f"Vault analytics served to agent {agent_info['agent_id']} for {days} days")
        
        return jsonify({
            'analytics': analytics,
            'period_days': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Vault analytics request failed: {e}")
        return jsonify({'error': 'Failed to get vault analytics'}), 500


@app.route('/api/logs/recent', methods=['GET'])
def get_recent_logs():
    """Get recent structured logs for the BrainWindow"""
    try:
        # Get query parameters
        limit = min(int(request.args.get('limit', 100)), 500)
        log_type = request.args.get('type', 'all')  # all, vault, compound, intelligence
        since_timestamp = request.args.get('since')
        
        # Mock structured logs for demonstration
        # In production, this would read from actual log files or database
        current_time = datetime.now()
        
        logs = []
        
        # Generate sample logs based on type
        if log_type in ['all', 'vault']:
            logs.extend([
                {
                    'timestamp': (current_time - timedelta(minutes=5)).isoformat(),
                    'level': 'INFO',
                    'module': 'VaultMLEngine',
                    'message': 'Strategy prediction complete | Recommended: gamma_balanced | Confidence: 87% | Expected APR: 18.5% | Ranges: 2',
                    'type': 'vault_strategy',
                    'metadata': {
                        'strategy': 'gamma_balanced',
                        'confidence': 0.87,
                        'expected_apr': 0.185,
                        'ranges_count': 2
                    }
                },
                {
                    'timestamp': (current_time - timedelta(minutes=8)).isoformat(),
                    'level': 'INFO',
                    'module': 'GammaStyleOptimizer',
                    'message': 'Dual position optimization complete | Base Range: [98000, 102000] (75%) | Limit Range: [99500, 100500] (25%)',
                    'type': 'vault_optimization',
                    'metadata': {
                        'base_range': [98000, 102000],
                        'base_allocation': 0.75,
                        'limit_range': [99500, 100500],
                        'limit_allocation': 0.25
                    }
                }
            ])
        
        if log_type in ['all', 'compound']:
            logs.extend([
                {
                    'timestamp': (current_time - timedelta(minutes=3)).isoformat(),
                    'level': 'INFO',
                    'module': 'CompoundService',
                    'message': 'COMPOUND_SUCCESS | Token ID: 12345 | TX Hash: 0xabc...def | Gas Used: 150,000 | Net Profit: $45.67',
                    'type': 'compound_success',
                    'metadata': {
                        'token_id': 12345,
                        'tx_hash': '0xabc...def',
                        'gas_used': 150000,
                        'net_profit': 45.67,
                        'strategy': 'ai_optimized'
                    }
                },
                {
                    'timestamp': (current_time - timedelta(minutes=7)).isoformat(),
                    'level': 'INFO',
                    'module': 'CompoundService',
                    'message': 'Found 15 compound opportunities | Total Profit Potential: $1,234.56 | Top Priority Score: 0.89',
                    'type': 'compound_opportunities',
                    'metadata': {
                        'opportunities_count': 15,
                        'total_profit_potential': 1234.56,
                        'top_priority_score': 0.89
                    }
                }
            ])
        
        if log_type in ['all', 'intelligence']:
            logs.extend([
                {
                    'timestamp': (current_time - timedelta(minutes=2)).isoformat(),
                    'level': 'INFO',
                    'module': 'DexBrain',
                    'message': 'Generated vault intelligence for 0x123...abc | Strategy: gamma_balanced | Confidence: 87% | Compound Opportunities: 5',
                    'type': 'vault_intelligence',
                    'metadata': {
                        'vault_address': '0x123...abc',
                        'strategy': 'gamma_balanced',
                        'confidence': 0.87,
                        'compound_opportunities': 5
                    }
                },
                {
                    'timestamp': (current_time - timedelta(minutes=6)).isoformat(),
                    'level': 'INFO',
                    'module': 'DexBrain',
                    'message': 'Intelligence served to agent dexter_agent_1 | Insights: 45 | Predictions: 3 | Quality Score: 92%',
                    'type': 'intelligence_feed',
                    'metadata': {
                        'agent_id': 'dexter_agent_1',
                        'insights_count': 45,
                        'predictions_count': 3,
                        'quality_score': 0.92
                    }
                }
            ])
        
        # Sort logs by timestamp (most recent first)
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply limit
        logs = logs[:limit]
        
        # Filter by timestamp if provided
        if since_timestamp:
            try:
                since_dt = datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
                logs = [log for log in logs if datetime.fromisoformat(log['timestamp']) > since_dt]
            except ValueError:
                pass  # Invalid timestamp format, ignore filter
        
        return jsonify({
            'logs': logs,
            'total_count': len(logs),
            'filters': {
                'limit': limit,
                'type': log_type,
                'since': since_timestamp
            },
            'timestamp': current_time.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Recent logs request failed: {e}")
        return jsonify({'error': 'Failed to get recent logs'}), 500


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