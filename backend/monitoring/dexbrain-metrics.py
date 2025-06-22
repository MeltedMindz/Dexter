#!/usr/bin/env python3
"""
DexBrain Custom Metrics Exporter
Exposes business-specific metrics for Prometheus monitoring
"""

import time
import json
import asyncio
import aiohttp
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import psycopg2
import os
from typing import Dict, Any

# Prometheus metrics
ACTIVE_AGENTS = Gauge('dexbrain_active_agents', 'Number of active agents')
TOTAL_AGENTS = Gauge('dexbrain_total_agents', 'Total number of registered agents')
API_REQUESTS = Counter('dexbrain_api_requests_total', 'Total API requests', ['endpoint', 'status'])
API_DURATION = Histogram('dexbrain_api_duration_seconds', 'API request duration', ['endpoint'])
DATA_SUBMISSIONS = Counter('dexbrain_data_submissions_total', 'Total data submissions', ['blockchain', 'status'])
INTELLIGENCE_QUERIES = Counter('dexbrain_intelligence_queries_total', 'Total intelligence queries')
DATA_QUALITY_SCORE = Gauge('dexbrain_data_quality_score', 'Average data quality score')
NETWORK_TVL = Gauge('dexbrain_network_tvl_usd', 'Total value locked across network')
LAST_INTELLIGENCE_UPDATE = Gauge('dexbrain_last_intelligence_update', 'Timestamp of last intelligence update')

# Vault-specific metrics
VAULT_STRATEGIES_GENERATED = Counter('dexbrain_vault_strategies_total', 'Total vault strategies generated', ['strategy_type'])
VAULT_COMPOUNDS_SUCCESSFUL = Counter('dexbrain_vault_compounds_successful_total', 'Successful vault compounds')
VAULT_COMPOUNDS_FAILED = Counter('dexbrain_vault_compounds_failed_total', 'Failed vault compounds')
VAULT_TOTAL_TVL = Gauge('dexbrain_vault_total_tvl_usd', 'Total value locked in all vaults')
VAULT_AVERAGE_APR = Gauge('dexbrain_vault_average_apr', 'Average APR across all vaults')
VAULT_AI_CONFIDENCE = Gauge('dexbrain_vault_ai_confidence', 'Average AI confidence score for vault strategies')
COMPOUND_OPPORTUNITIES = Gauge('dexbrain_compound_opportunities', 'Current number of compound opportunities')
COMPOUND_PROFIT_POTENTIAL = Gauge('dexbrain_compound_profit_potential_usd', 'Total profit potential from compound opportunities')

# Database connection
DB_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/dexter_db')

class DexBrainMetricsExporter:
    def __init__(self):
        self.db_conn = None
        self.api_base_url = os.getenv('DEXBRAIN_API_URL', 'http://localhost:8080')
        
    async def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(DB_URL)
            print("✅ Connected to PostgreSQL database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            
    async def collect_agent_metrics(self):
        """Collect agent-related metrics"""
        try:
            if not self.db_conn:
                await self.connect_db()
                
            cursor = self.db_conn.cursor()
            
            # Active agents (used in last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) FROM user_agents 
                WHERE is_active = true 
                AND (last_used > NOW() - INTERVAL '24 hours' OR created_at > NOW() - INTERVAL '24 hours')
            """)
            active_count = cursor.fetchone()[0]
            ACTIVE_AGENTS.set(active_count)
            
            # Total agents
            cursor.execute("SELECT COUNT(*) FROM user_agents WHERE is_active = true")
            total_count = cursor.fetchone()[0]
            TOTAL_AGENTS.set(total_count)
            
            print(f"📊 Agents: {active_count} active, {total_count} total")
            
        except Exception as e:
            print(f"❌ Agent metrics collection failed: {e}")
            
    async def collect_api_metrics(self):
        """Collect API usage metrics from DexBrain API"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get API statistics
                async with session.get(f"{self.api_base_url}/api/stats") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Update intelligence queries counter
                        if 'total_insights' in data.get('network_stats', {}):
                            INTELLIGENCE_QUERIES._value._value = data['network_stats']['total_insights']
                            
                        print(f"📈 API stats collected: {data.get('network_stats', {}).get('total_insights', 0)} insights")
                    
        except Exception as e:
            print(f"❌ API metrics collection failed: {e}")
            
    async def collect_data_quality_metrics(self):
        """Collect data quality metrics"""
        try:
            if not self.db_conn:
                await self.connect_db()
                
            cursor = self.db_conn.cursor()
            
            # Calculate average data quality score from recent submissions
            cursor.execute("""
                SELECT AVG(
                    CAST(metrics->>'quality_score' AS FLOAT)
                ) as avg_quality
                FROM performance_metrics 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                AND metrics->>'quality_score' IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and result[0]:
                DATA_QUALITY_SCORE.set(float(result[0]))
                print(f"📋 Average data quality score: {result[0]:.2f}")
            
        except Exception as e:
            print(f"❌ Data quality metrics collection failed: {e}")
            
    async def collect_network_metrics(self):
        """Collect network-wide metrics"""
        try:
            if not self.db_conn:
                await self.connect_db()
                
            cursor = self.db_conn.cursor()
            
            # Calculate total network TVL
            cursor.execute("""
                SELECT SUM(
                    CAST(metrics->>'total_liquidity_usd' AS FLOAT)
                ) as total_tvl
                FROM performance_metrics 
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                AND metrics->>'total_liquidity_usd' IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and result[0]:
                NETWORK_TVL.set(float(result[0]))
                print(f"💰 Network TVL: ${result[0]:,.2f}")
                
            # Update last intelligence update timestamp
            LAST_INTELLIGENCE_UPDATE.set(time.time())
            
        except Exception as e:
            print(f"❌ Network metrics collection failed: {e}")
            
    async def collect_vault_metrics(self):
        """Collect vault-specific metrics"""
        try:
            # Get vault statistics from API
            async with aiohttp.ClientSession() as session:
                # Get vault analytics
                try:
                    async with session.get(f"{self.api_base_url}/api/vault/analytics?days=1") as response:
                        if response.status == 200:
                            data = await response.json()
                            analytics = data.get('analytics', {})
                            
                            # Update vault TVL if available
                            if 'compound_volume' in analytics:
                                volume_data = analytics['compound_volume']
                                if 'total_tvl' in volume_data:
                                    VAULT_TOTAL_TVL.set(volume_data['total_tvl'])
                                    
                            # Update average APR
                            if 'overall_metrics' in analytics:
                                metrics = analytics['overall_metrics']
                                if 'average_apr' in metrics:
                                    VAULT_AVERAGE_APR.set(metrics['average_apr'])
                                    
                            print(f"📊 Vault analytics updated")
                except Exception as e:
                    print(f"⚠️ Vault analytics collection failed: {e}")
                
                # Get compound opportunities
                try:
                    async with session.get(f"{self.api_base_url}/api/vault/compound-opportunities?limit=50") as response:
                        if response.status == 200:
                            data = await response.json()
                            opportunities = data.get('opportunities', [])
                            
                            COMPOUND_OPPORTUNITIES.set(len(opportunities))
                            
                            # Calculate total profit potential
                            total_profit = sum(opp.get('profit_potential', 0) for opp in opportunities)
                            COMPOUND_PROFIT_POTENTIAL.set(total_profit)
                            
                            # Calculate average AI confidence
                            if opportunities:
                                avg_confidence = sum(opp.get('ai_confidence', 0) for opp in opportunities) / len(opportunities)
                                VAULT_AI_CONFIDENCE.set(avg_confidence)
                            
                            print(f"🔄 Compound opportunities: {len(opportunities)}, Profit potential: ${total_profit:,.2f}")
                except Exception as e:
                    print(f"⚠️ Compound opportunities collection failed: {e}")
                    
        except Exception as e:
            print(f"❌ Vault metrics collection failed: {e}")
            
    async def collect_all_metrics(self):
        """Collect all metrics"""
        print(f"🔄 Collecting metrics at {datetime.now()}")
        
        await asyncio.gather(
            self.collect_agent_metrics(),
            self.collect_api_metrics(),
            self.collect_data_quality_metrics(),
            self.collect_network_metrics(),
            self.collect_vault_metrics(),
            return_exceptions=True
        )
        
        print("✅ Metrics collection completed")
        
    async def run_forever(self):
        """Run metrics collection loop"""
        while True:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                print(f"❌ Metrics collection error: {e}")
                await asyncio.sleep(30)  # Retry after 30 seconds on error

def main():
    """Main function"""
    print("🚀 Starting DexBrain Metrics Exporter")
    
    # Start Prometheus HTTP server
    start_http_server(8081)
    print("📊 Prometheus metrics server started on port 8081")
    
    # Start metrics collection
    exporter = DexBrainMetricsExporter()
    
    try:
        asyncio.run(exporter.run_forever())
    except KeyboardInterrupt:
        print("🛑 Shutting down metrics exporter")

if __name__ == "__main__":
    main()