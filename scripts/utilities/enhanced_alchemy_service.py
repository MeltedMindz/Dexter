#!/usr/bin/env python3
"""
Enhanced Alchemy-based Data Collection Service
Production-ready service with correct Base network contracts
"""

import asyncio
import logging
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from web3 import Web3
import aiohttp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

class EnhancedAlchemyService:
    """
    Enhanced Alchemy-based data collection service for Base network
    """
    
    def __init__(self, alchemy_api_key: str):
        self.alchemy_api_key = alchemy_api_key
        self.alchemy_base_rpc = f"https://base-mainnet.g.alchemy.com/v2/{alchemy_api_key}"
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.alchemy_base_rpc))
        
        # Correct Uniswap V3 Position Manager address on Base Network
        self.npm_address = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
        
        # Service statistics
        self.service_stats = {
            'start_time': datetime.now(),
            'successful_collections': 0,
            'total_positions_found': 0,
            'rpc_calls_made': 0,
            'errors_encountered': 0,
            'last_successful_collection': None
        }
        
        logger.info(f"Enhanced Alchemy service initialized. Web3 connected: {self.w3.is_connected()}")
    
    async def start_enhanced_collection_service(self):
        """
        Start the enhanced collection service
        """
        logger.info("🚀 Starting enhanced Alchemy collection service...")
        
        await self._log_service_start()
        
        # Verify connection first
        if not self.w3.is_connected():
            logger.error("❌ Failed to connect to Alchemy RPC")
            return
        
        # Get basic network info
        try:
            current_block = self.w3.eth.block_number
            network_info = {
                'current_block': current_block,
                'network_id': self.w3.eth.chain_id,
                'gas_price': self.w3.eth.gas_price
            }
            
            await self._log_network_info(network_info)
            
        except Exception as e:
            logger.error(f"Failed to get network info: {e}")
            return
        
        # Start collection tasks
        tasks = [
            asyncio.create_task(self._transaction_monitoring_loop()),
            asyncio.create_task(self._block_analysis_loop()),
            asyncio.create_task(self._health_monitoring_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Service error: {e}")
            await self._handle_service_error(e)
    
    async def _transaction_monitoring_loop(self):
        """
        Monitor recent transactions for Uniswap activity
        """
        logger.info("🔍 Starting transaction monitoring...")
        
        while True:
            try:
                current_block = self.w3.eth.block_number
                self.service_stats['rpc_calls_made'] += 1
                
                # Get recent blocks
                recent_blocks = []
                for i in range(5):  # Last 5 blocks
                    try:
                        block = self.w3.eth.get_block(current_block - i, full_transactions=True)
                        recent_blocks.append(block)
                        self.service_stats['rpc_calls_made'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to get block {current_block - i}: {e}")
                        continue
                
                # Analyze transactions in recent blocks
                uniswap_transactions = []
                total_transactions = 0
                
                for block in recent_blocks:
                    total_transactions += len(block.transactions)
                    
                    for tx in block.transactions:
                        # Check if transaction involves Uniswap contracts
                        if tx.to and (
                            tx.to.lower() == self.npm_address.lower() or
                            self._is_uniswap_related(tx.to)
                        ):
                            uniswap_transactions.append({
                                'hash': tx.hash.hex(),
                                'to': tx.to,
                                'value': float(tx.value),
                                'gas': tx.gas,
                                'gas_price': tx.gasPrice,
                                'block_number': block.number,
                                'timestamp': datetime.fromtimestamp(block.timestamp)
                            })
                
                # Log transaction analysis
                await self._log_transaction_analysis(
                    len(recent_blocks),
                    total_transactions,
                    len(uniswap_transactions),
                    current_block
                )
                
                self.service_stats['successful_collections'] += 1
                self.service_stats['last_successful_collection'] = datetime.now()
                
                # Wait 2 minutes between transaction monitoring cycles
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"Transaction monitoring error: {e}")
                self.service_stats['errors_encountered'] += 1
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _block_analysis_loop(self):
        """
        Analyze blocks for DeFi activity patterns
        """
        logger.info("📊 Starting block analysis...")
        
        while True:
            try:
                current_block = self.w3.eth.block_number
                start_block = current_block - 100  # Analyze last 100 blocks
                
                # Collect block statistics
                block_stats = {
                    'blocks_analyzed': 0,
                    'total_transactions': 0,
                    'total_gas_used': 0,
                    'defi_transactions': 0,
                    'average_gas_price': 0
                }
                
                gas_prices = []
                
                for block_num in range(start_block, current_block, 10):  # Sample every 10th block
                    try:
                        block = self.w3.eth.get_block(block_num)
                        block_stats['blocks_analyzed'] += 1
                        block_stats['total_transactions'] += block.gasUsed
                        block_stats['total_gas_used'] += block.gasUsed
                        
                        # Estimate gas price from block
                        if hasattr(block, 'baseFeePerGas') and block.baseFeePerGas:
                            gas_prices.append(block.baseFeePerGas)
                        
                        self.service_stats['rpc_calls_made'] += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze block {block_num}: {e}")
                        continue
                
                if gas_prices:
                    block_stats['average_gas_price'] = sum(gas_prices) / len(gas_prices)
                
                # Estimate DeFi activity (simplified)
                block_stats['defi_activity_score'] = min(
                    block_stats['total_gas_used'] / 10_000_000,  # Normalize by 10M gas
                    1.0
                )
                
                await self._log_block_analysis(block_stats, current_block)
                
                # Wait 30 minutes between block analysis
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Block analysis error: {e}")
                self.service_stats['errors_encountered'] += 1
                await asyncio.sleep(900)
    
    async def _health_monitoring_loop(self):
        """
        Monitor service health and performance
        """
        logger.info("💓 Starting health monitoring...")
        
        while True:
            try:
                uptime = datetime.now() - self.service_stats['start_time']
                uptime_hours = uptime.total_seconds() / 3600
                
                success_rate = (
                    self.service_stats['successful_collections'] /
                    max(self.service_stats['successful_collections'] + self.service_stats['errors_encountered'], 1)
                )
                
                rpc_calls_per_hour = self.service_stats['rpc_calls_made'] / max(uptime_hours, 1)
                
                health_metrics = {
                    'uptime_hours': uptime_hours,
                    'successful_collections': self.service_stats['successful_collections'],
                    'total_rpc_calls': self.service_stats['rpc_calls_made'],
                    'rpc_calls_per_hour': rpc_calls_per_hour,
                    'success_rate': success_rate,
                    'errors_encountered': self.service_stats['errors_encountered'],
                    'last_successful_collection': self.service_stats['last_successful_collection'].isoformat() if self.service_stats['last_successful_collection'] else None,
                    'web3_connected': self.w3.is_connected()
                }
                
                await self._log_service_health(health_metrics)
                
                # Check for issues
                if success_rate < 0.8 and self.service_stats['successful_collections'] > 5:
                    await self._alert_low_success_rate(success_rate)
                
                if not self.w3.is_connected():
                    await self._alert_connection_lost()
                
                # Wait 15 minutes between health checks
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(900)
    
    def _is_uniswap_related(self, address: str) -> bool:
        """
        Check if address is related to Uniswap contracts
        """
        if not address:
            return False
        
        # Known Uniswap V3 contract patterns on Base
        uniswap_patterns = [
            "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",  # Position Manager
            "0x33128a8fC17869897dcE68Ed026d694621f6FDfD",  # Factory
            "0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24",  # Router
        ]
        
        return any(pattern.lower() == address.lower() for pattern in uniswap_patterns)
    
    async def _log_service_start(self):
        """Log service startup"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "EnhancedAlchemyService",
            "message": "Enhanced Alchemy data collection service started",
            "details": {
                "network": "Base",
                "alchemy_endpoint": f"https://base-mainnet.g.alchemy.com/v2/{self.alchemy_api_key[:8]}...",
                "position_manager": self.npm_address,
                "capabilities": ["Transaction monitoring", "Block analysis", "DeFi activity tracking"],
                "web3_connected": self.w3.is_connected()
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_network_info(self, network_info):
        """Log network information"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "NetworkInfo",
            "message": f"Connected to Base network - Block {network_info['current_block']}",
            "details": network_info
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_transaction_analysis(self, blocks_analyzed, total_txs, uniswap_txs, current_block):
        """Log transaction analysis results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "TransactionAnalysis",
            "message": f"Analyzed {blocks_analyzed} blocks: {uniswap_txs} Uniswap txs out of {total_txs} total",
            "details": {
                "blocks_analyzed": blocks_analyzed,
                "total_transactions": total_txs,
                "uniswap_transactions": uniswap_txs,
                "uniswap_percentage": (uniswap_txs / max(total_txs, 1)) * 100,
                "current_block": current_block,
                "rpc_calls": self.service_stats['rpc_calls_made']
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_block_analysis(self, block_stats, current_block):
        """Log block analysis results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "BlockAnalysis",
            "message": f"Analyzed {block_stats['blocks_analyzed']} blocks - DeFi activity score: {block_stats['defi_activity_score']:.2f}",
            "details": {
                **block_stats,
                "current_block": current_block
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_service_health(self, health_metrics):
        """Log service health metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "ServiceHealth",
            "message": f"Service health: {health_metrics['success_rate']:.1%} success rate, {health_metrics['rpc_calls_per_hour']:.0f} RPC calls/hour",
            "details": health_metrics
        }
        website_logger.info(json.dumps(log_data))
    
    async def _alert_low_success_rate(self, success_rate):
        """Alert on low success rate"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "source": "ServiceAlert",
            "message": f"Low success rate detected: {success_rate:.1%}",
            "details": {
                "success_rate": success_rate,
                "successful_collections": self.service_stats['successful_collections'],
                "errors_encountered": self.service_stats['errors_encountered']
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _alert_connection_lost(self):
        """Alert on connection loss"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "source": "ConnectionAlert",
            "message": "Web3 connection to Alchemy lost",
            "details": {
                "alchemy_endpoint": f"https://base-mainnet.g.alchemy.com/v2/{self.alchemy_api_key[:8]}...",
                "last_successful_collection": self.service_stats['last_successful_collection'].isoformat() if self.service_stats['last_successful_collection'] else None
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _handle_service_error(self, error):
        """Handle service errors"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "source": "ServiceError",
            "message": f"Service error: {error}",
            "details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "service_stats": self.service_stats
            }
        }
        website_logger.info(json.dumps(log_data))


async def main():
    """Run the enhanced Alchemy service"""
    alchemy_key = os.getenv('ALCHEMY_API_KEY', 'ory0F2cLFNIXsovAmrtJj')
    
    service = EnhancedAlchemyService(alchemy_key)
    await service.start_enhanced_collection_service()

if __name__ == "__main__":
    asyncio.run(main())