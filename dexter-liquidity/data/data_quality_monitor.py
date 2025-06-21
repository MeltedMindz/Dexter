"""
Data Quality Monitor
Comprehensive data quality monitoring, validation, and backfill system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import aiohttp
import time

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics structure"""
    timestamp: datetime
    source: str
    completeness_score: float  # 0-1
    accuracy_score: float  # 0-1
    consistency_score: float  # 0-1
    timeliness_score: float  # 0-1
    overall_score: float  # 0-1
    
    # Detailed metrics
    total_expected_records: int
    total_actual_records: int
    missing_records: int
    duplicate_records: int
    invalid_records: int
    stale_records: int
    
    # Field-specific quality
    field_completeness: Dict[str, float]
    field_accuracy: Dict[str, float]
    
    # Recommendations
    issues_found: List[str]
    recommendations: List[str]

@dataclass
class BackfillTask:
    """Backfill task definition"""
    task_id: str
    source: str
    start_date: datetime
    end_date: datetime
    status: str  # pending, running, completed, failed
    priority: int  # 1-10
    estimated_records: int
    actual_records: int
    error_count: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

class DataQualityMonitor:
    """
    Comprehensive data quality monitoring and backfill system
    """
    
    def __init__(self, 
                 database_url: str,
                 alchemy_api_key: str,
                 graph_api_url: str = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"):
        """
        Initialize data quality monitor
        
        Args:
            database_url: Database connection string
            alchemy_api_key: Alchemy API key for RPC calls
            graph_api_url: Graph API endpoint
        """
        self.database_url = database_url
        self.alchemy_api_key = alchemy_api_key
        self.graph_api_url = graph_api_url
        
        # Database connection
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Quality monitoring state
        self.quality_history: List[DataQualityMetrics] = []
        self.backfill_queue: List[BackfillTask] = []
        self.is_monitoring = False
        
        # Configuration
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% completeness required
            'accuracy': 0.98,      # 98% accuracy required
            'consistency': 0.95,   # 95% consistency required
            'timeliness': 0.90,    # 90% timeliness required
            'overall': 0.94        # 94% overall quality required
        }
        
        # Expected data sources and their update frequencies
        self.data_sources = {
            'graph_api': {
                'name': 'Graph API Positions',
                'frequency_minutes': 15,
                'expected_fields': [
                    'id', 'owner', 'pool', 'token0', 'token1', 
                    'tickLower', 'tickUpper', 'liquidity', 'depositedToken0',
                    'depositedToken1', 'withdrawnToken0', 'withdrawnToken1',
                    'collectedFeesToken0', 'collectedFeesToken1', 'transaction'
                ],
                'table_name': 'positions'
            },
            'alchemy_rpc': {
                'name': 'Alchemy RPC Data',
                'frequency_minutes': 10,
                'expected_fields': [
                    'position_id', 'block_number', 'timestamp', 'owner',
                    'pool_address', 'tick_lower', 'tick_upper', 'liquidity',
                    'token0_amount', 'token1_amount', 'fees_earned_token0',
                    'fees_earned_token1', 'price_at_collection'
                ],
                'table_name': 'alchemy_positions'
            },
            'price_data': {
                'name': 'Price Data',
                'frequency_minutes': 5,
                'expected_fields': [
                    'token_address', 'timestamp', 'price_usd', 'volume_24h',
                    'market_cap', 'price_change_24h', 'source'
                ],
                'table_name': 'token_prices'
            }
        }
        
        logger.info("Data quality monitor initialized")
    
    async def start_monitoring(self, check_interval_minutes: int = 30):
        """
        Start continuous data quality monitoring
        
        Args:
            check_interval_minutes: Interval between quality checks
        """
        logger.info("🔍 Starting data quality monitoring...")
        self.is_monitoring = True
        
        while self.is_monitoring:
            try:
                # Run quality checks for all data sources
                for source_id, source_config in self.data_sources.items():
                    logger.info(f"Running quality check for {source_config['name']}")
                    metrics = await self.check_data_quality(source_id)
                    self.quality_history.append(metrics)
                    
                    # Check if backfill is needed
                    if metrics.overall_score < self.quality_thresholds['overall']:
                        await self._trigger_backfill_if_needed(source_id, metrics)
                
                # Process backfill queue
                await self._process_backfill_queue()
                
                # Clean up old quality history (keep last 1000 entries)
                if len(self.quality_history) > 1000:
                    self.quality_history = self.quality_history[-1000:]
                
                # Generate quality report
                await self._generate_quality_report()
                
                # Wait for next check
                await asyncio.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def check_data_quality(self, source_id: str) -> DataQualityMetrics:
        """
        Perform comprehensive data quality check for a specific source
        
        Args:
            source_id: ID of the data source to check
            
        Returns:
            DataQualityMetrics with quality assessment
        """
        try:
            source_config = self.data_sources.get(source_id)
            if not source_config:
                raise ValueError(f"Unknown source ID: {source_id}")
            
            logger.info(f"Checking quality for {source_config['name']}")
            
            # Get current data from database
            current_data = await self._fetch_current_data(source_id)
            
            # Calculate completeness
            completeness_metrics = await self._check_completeness(source_id, current_data)
            
            # Calculate accuracy
            accuracy_metrics = await self._check_accuracy(source_id, current_data)
            
            # Calculate consistency
            consistency_metrics = await self._check_consistency(source_id, current_data)
            
            # Calculate timeliness
            timeliness_metrics = await self._check_timeliness(source_id, current_data)
            
            # Calculate overall score
            overall_score = (
                completeness_metrics['score'] * 0.3 +
                accuracy_metrics['score'] * 0.3 +
                consistency_metrics['score'] * 0.2 +
                timeliness_metrics['score'] * 0.2
            )
            
            # Compile issues and recommendations
            issues = []
            recommendations = []
            
            issues.extend(completeness_metrics.get('issues', []))
            issues.extend(accuracy_metrics.get('issues', []))
            issues.extend(consistency_metrics.get('issues', []))
            issues.extend(timeliness_metrics.get('issues', []))
            
            recommendations.extend(completeness_metrics.get('recommendations', []))
            recommendations.extend(accuracy_metrics.get('recommendations', []))
            recommendations.extend(consistency_metrics.get('recommendations', []))
            recommendations.extend(timeliness_metrics.get('recommendations', []))
            
            # Create quality metrics object
            metrics = DataQualityMetrics(
                timestamp=datetime.now(),
                source=source_id,
                completeness_score=completeness_metrics['score'],
                accuracy_score=accuracy_metrics['score'],
                consistency_score=consistency_metrics['score'],
                timeliness_score=timeliness_metrics['score'],
                overall_score=overall_score,
                total_expected_records=completeness_metrics.get('expected_records', 0),
                total_actual_records=completeness_metrics.get('actual_records', 0),
                missing_records=completeness_metrics.get('missing_records', 0),
                duplicate_records=consistency_metrics.get('duplicate_records', 0),
                invalid_records=accuracy_metrics.get('invalid_records', 0),
                stale_records=timeliness_metrics.get('stale_records', 0),
                field_completeness=completeness_metrics.get('field_completeness', {}),
                field_accuracy=accuracy_metrics.get('field_accuracy', {}),
                issues_found=issues,
                recommendations=recommendations
            )
            
            logger.info(f"Quality check completed - Overall score: {overall_score:.2%}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error checking data quality for {source_id}: {e}")
            return self._empty_quality_metrics(source_id)
    
    async def trigger_historical_backfill(self, 
                                        source_id: str,
                                        start_date: datetime,
                                        end_date: datetime,
                                        priority: int = 5) -> str:
        """
        Trigger historical data backfill for a specific period
        
        Args:
            source_id: Data source to backfill
            start_date: Start date for backfill
            end_date: End date for backfill
            priority: Priority level (1-10, 10 = highest)
            
        Returns:
            Task ID for the backfill job
        """
        try:
            task_id = f"{source_id}_backfill_{int(time.time())}"
            
            # Estimate number of records
            estimated_records = await self._estimate_backfill_records(source_id, start_date, end_date)
            
            backfill_task = BackfillTask(
                task_id=task_id,
                source=source_id,
                start_date=start_date,
                end_date=end_date,
                status="pending",
                priority=priority,
                estimated_records=estimated_records,
                actual_records=0,
                error_count=0,
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                error_message=None
            )
            
            # Add to queue (sorted by priority)
            self.backfill_queue.append(backfill_task)
            self.backfill_queue.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Backfill task created: {task_id} for {source_id} ({start_date} to {end_date})")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating backfill task: {e}")
            return ""
    
    async def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data quality dashboard information
        
        Returns:
            Dashboard data with current quality status
        """
        try:
            if not self.quality_history:
                return {}
            
            # Get latest metrics for each source
            latest_metrics = {}
            for source_id in self.data_sources.keys():
                source_metrics = [m for m in self.quality_history if m.source == source_id]
                if source_metrics:
                    latest_metrics[source_id] = source_metrics[-1]
            
            # Calculate overall system quality
            if latest_metrics:
                overall_system_quality = np.mean([m.overall_score for m in latest_metrics.values()])
            else:
                overall_system_quality = 0.0
            
            # Get quality trends (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_metrics = [m for m in self.quality_history if m.timestamp >= cutoff_time]
            
            # Count active issues
            active_issues = []
            for metrics in latest_metrics.values():
                active_issues.extend(metrics.issues_found)
            
            # Backfill queue status
            pending_backfills = len([t for t in self.backfill_queue if t.status == "pending"])
            running_backfills = len([t for t in self.backfill_queue if t.status == "running"])
            
            dashboard_data = {
                'system_overview': {
                    'overall_quality': overall_system_quality,
                    'quality_status': self._get_quality_status(overall_system_quality),
                    'total_data_sources': len(self.data_sources),
                    'sources_passing': len([m for m in latest_metrics.values() if m.overall_score >= self.quality_thresholds['overall']]),
                    'active_issues': len(active_issues),
                    'last_updated': datetime.now().isoformat()
                },
                'source_quality': {
                    source_id: {
                        'name': self.data_sources[source_id]['name'],
                        'overall_score': metrics.overall_score,
                        'completeness': metrics.completeness_score,
                        'accuracy': metrics.accuracy_score,
                        'consistency': metrics.consistency_score,
                        'timeliness': metrics.timeliness_score,
                        'last_check': metrics.timestamp.isoformat(),
                        'status': 'passing' if metrics.overall_score >= self.quality_thresholds['overall'] else 'failing',
                        'issues_count': len(metrics.issues_found)
                    }
                    for source_id, metrics in latest_metrics.items()
                },
                'recent_trends': {
                    'quality_over_time': [
                        {
                            'timestamp': m.timestamp.isoformat(),
                            'source': m.source,
                            'quality_score': m.overall_score
                        }
                        for m in recent_metrics[-50:]  # Last 50 data points
                    ]
                },
                'backfill_status': {
                    'pending_tasks': pending_backfills,
                    'running_tasks': running_backfills,
                    'completed_today': len([
                        t for t in self.backfill_queue 
                        if t.status == "completed" and t.completed_at and 
                        t.completed_at >= datetime.now() - timedelta(days=1)
                    ])
                },
                'active_issues': active_issues[:10],  # Top 10 issues
                'recommendations': [
                    rec for metrics in latest_metrics.values() 
                    for rec in metrics.recommendations[:3]  # Top 3 per source
                ]
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {}
    
    async def export_quality_report(self, output_path: str) -> Dict[str, Any]:
        """
        Export comprehensive quality report
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Complete quality report
        """
        try:
            dashboard_data = await self.get_quality_dashboard_data()
            
            # Add detailed metrics history
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'monitoring_period': {
                        'start': self.quality_history[0].timestamp.isoformat() if self.quality_history else None,
                        'end': self.quality_history[-1].timestamp.isoformat() if self.quality_history else None,
                        'total_checks': len(self.quality_history)
                    },
                    'data_sources_monitored': len(self.data_sources)
                },
                'dashboard_summary': dashboard_data,
                'detailed_metrics': [asdict(m) for m in self.quality_history[-100:]],  # Last 100 checks
                'backfill_history': [asdict(t) for t in self.backfill_queue],
                'quality_thresholds': self.quality_thresholds,
                'source_configurations': self.data_sources
            }
            
            # Save report
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Quality report exported to {output_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error exporting quality report: {e}")
            return {}
    
    # Helper methods
    
    async def _fetch_current_data(self, source_id: str) -> pd.DataFrame:
        """Fetch current data for quality analysis"""
        try:
            source_config = self.data_sources[source_id]
            table_name = source_config['table_name']
            
            with self.engine.connect() as conn:
                # Get recent data (last 24 hours)
                query = text(f"""
                    SELECT * FROM {table_name} 
                    WHERE timestamp >= NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC
                    LIMIT 10000
                """)
                
                result = conn.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {source_id}: {e}")
            return pd.DataFrame()
    
    async def _check_completeness(self, source_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness"""
        try:
            source_config = self.data_sources[source_id]
            expected_fields = source_config['expected_fields']
            frequency_minutes = source_config['frequency_minutes']
            
            # Calculate expected records based on frequency
            hours_in_day = 24
            expected_records_per_day = (hours_in_day * 60) // frequency_minutes
            actual_records = len(data)
            
            # Field completeness
            field_completeness = {}
            for field in expected_fields:
                if field in data.columns:
                    non_null_count = data[field].notna().sum()
                    field_completeness[field] = non_null_count / len(data) if len(data) > 0 else 0
                else:
                    field_completeness[field] = 0.0
            
            # Overall completeness score
            avg_field_completeness = np.mean(list(field_completeness.values())) if field_completeness else 0
            record_completeness = min(actual_records / expected_records_per_day, 1.0)
            completeness_score = (avg_field_completeness + record_completeness) / 2
            
            issues = []
            recommendations = []
            
            if completeness_score < self.quality_thresholds['completeness']:
                issues.append(f"Low completeness: {completeness_score:.1%} (threshold: {self.quality_thresholds['completeness']:.1%})")
                
                if record_completeness < 0.9:
                    recommendations.append(f"Missing {expected_records_per_day - actual_records} records for today")
                
                low_fields = [f for f, score in field_completeness.items() if score < 0.95]
                if low_fields:
                    recommendations.append(f"Improve data collection for fields: {', '.join(low_fields[:3])}")
            
            return {
                'score': completeness_score,
                'expected_records': expected_records_per_day,
                'actual_records': actual_records,
                'missing_records': max(0, expected_records_per_day - actual_records),
                'field_completeness': field_completeness,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking completeness for {source_id}: {e}")
            return {'score': 0.0, 'issues': [str(e)], 'recommendations': []}
    
    async def _check_accuracy(self, source_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data accuracy"""
        try:
            if data.empty:
                return {'score': 0.0, 'field_accuracy': {}, 'invalid_records': 0}
            
            source_config = self.data_sources[source_id]
            invalid_count = 0
            field_accuracy = {}
            
            # Field-specific accuracy checks
            if source_id == 'graph_api':
                # Check for valid addresses (should be 42 characters starting with 0x)
                if 'owner' in data.columns:
                    valid_owners = data['owner'].str.match(r'^0x[a-fA-F0-9]{40}$').sum()
                    field_accuracy['owner'] = valid_owners / len(data)
                    invalid_count += len(data) - valid_owners
                
                # Check for valid liquidity values (should be positive)
                if 'liquidity' in data.columns:
                    valid_liquidity = (pd.to_numeric(data['liquidity'], errors='coerce') >= 0).sum()
                    field_accuracy['liquidity'] = valid_liquidity / len(data)
                    invalid_count += len(data) - valid_liquidity
                
            elif source_id == 'alchemy_rpc':
                # Check for valid block numbers
                if 'block_number' in data.columns:
                    valid_blocks = (pd.to_numeric(data['block_number'], errors='coerce') > 0).sum()
                    field_accuracy['block_number'] = valid_blocks / len(data)
                    invalid_count += len(data) - valid_blocks
                
                # Check for valid amounts
                if 'token0_amount' in data.columns:
                    valid_amounts = (pd.to_numeric(data['token0_amount'], errors='coerce') >= 0).sum()
                    field_accuracy['token0_amount'] = valid_amounts / len(data)
                    invalid_count += len(data) - valid_amounts
            
            elif source_id == 'price_data':
                # Check for valid prices
                if 'price_usd' in data.columns:
                    valid_prices = (pd.to_numeric(data['price_usd'], errors='coerce') > 0).sum()
                    field_accuracy['price_usd'] = valid_prices / len(data)
                    invalid_count += len(data) - valid_prices
            
            # Calculate overall accuracy
            accuracy_score = np.mean(list(field_accuracy.values())) if field_accuracy else 1.0
            
            issues = []
            recommendations = []
            
            if accuracy_score < self.quality_thresholds['accuracy']:
                issues.append(f"Low accuracy: {accuracy_score:.1%} (threshold: {self.quality_thresholds['accuracy']:.1%})")
                recommendations.append("Review data validation rules and input sources")
            
            return {
                'score': accuracy_score,
                'field_accuracy': field_accuracy,
                'invalid_records': invalid_count,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking accuracy for {source_id}: {e}")
            return {'score': 0.0, 'field_accuracy': {}, 'invalid_records': 0}
    
    async def _check_consistency(self, source_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency"""
        try:
            if data.empty:
                return {'score': 1.0, 'duplicate_records': 0}
            
            # Check for duplicates
            duplicate_count = 0
            if 'id' in data.columns:
                duplicate_count = data['id'].duplicated().sum()
            elif 'position_id' in data.columns:
                duplicate_count = data['position_id'].duplicated().sum()
            
            # Calculate consistency score
            consistency_score = 1.0 - (duplicate_count / len(data))
            
            issues = []
            recommendations = []
            
            if consistency_score < self.quality_thresholds['consistency']:
                issues.append(f"Low consistency: {consistency_score:.1%} (threshold: {self.quality_thresholds['consistency']:.1%})")
                if duplicate_count > 0:
                    recommendations.append(f"Remove {duplicate_count} duplicate records")
            
            return {
                'score': consistency_score,
                'duplicate_records': duplicate_count,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking consistency for {source_id}: {e}")
            return {'score': 1.0, 'duplicate_records': 0}
    
    async def _check_timeliness(self, source_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data timeliness"""
        try:
            if data.empty:
                return {'score': 0.0, 'stale_records': 0}
            
            source_config = self.data_sources[source_id]
            frequency_minutes = source_config['frequency_minutes']
            
            # Check timestamp freshness
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                current_time = datetime.now()
                
                # Consider records stale if they're older than 2x the expected frequency
                stale_threshold = timedelta(minutes=frequency_minutes * 2)
                stale_records = (current_time - data['timestamp'] > stale_threshold).sum()
                
                # Calculate timeliness score
                timeliness_score = 1.0 - (stale_records / len(data))
            else:
                stale_records = len(data)  # All records are stale if no timestamp
                timeliness_score = 0.0
            
            issues = []
            recommendations = []
            
            if timeliness_score < self.quality_thresholds['timeliness']:
                issues.append(f"Low timeliness: {timeliness_score:.1%} (threshold: {self.quality_thresholds['timeliness']:.1%})")
                if stale_records > 0:
                    recommendations.append(f"Update {stale_records} stale records")
            
            return {
                'score': timeliness_score,
                'stale_records': stale_records,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking timeliness for {source_id}: {e}")
            return {'score': 0.0, 'stale_records': 0}
    
    async def _trigger_backfill_if_needed(self, source_id: str, metrics: DataQualityMetrics):
        """Trigger backfill if quality is below threshold"""
        try:
            if metrics.missing_records > 100:  # Significant data gaps
                # Calculate backfill period (last 7 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                # Check if similar backfill already exists
                existing_task = next(
                    (t for t in self.backfill_queue 
                     if t.source == source_id and t.status in ['pending', 'running']), 
                    None
                )
                
                if not existing_task:
                    task_id = await self.trigger_historical_backfill(
                        source_id, start_date, end_date, priority=8
                    )
                    logger.info(f"Auto-triggered backfill for {source_id}: {task_id}")
            
        except Exception as e:
            logger.error(f"Error triggering backfill for {source_id}: {e}")
    
    async def _process_backfill_queue(self):
        """Process pending backfill tasks"""
        try:
            # Get highest priority pending task
            pending_tasks = [t for t in self.backfill_queue if t.status == "pending"]
            if not pending_tasks:
                return
            
            # Limit concurrent backfills
            running_tasks = [t for t in self.backfill_queue if t.status == "running"]
            if len(running_tasks) >= 2:  # Max 2 concurrent backfills
                return
            
            task = pending_tasks[0]  # Highest priority (already sorted)
            await self._execute_backfill_task(task)
            
        except Exception as e:
            logger.error(f"Error processing backfill queue: {e}")
    
    async def _execute_backfill_task(self, task: BackfillTask):
        """Execute a specific backfill task"""
        try:
            logger.info(f"Starting backfill task: {task.task_id}")
            
            # Update task status
            task.status = "running"
            task.started_at = datetime.now()
            
            if task.source == 'graph_api':
                records_added = await self._backfill_graph_data(task.start_date, task.end_date)
            elif task.source == 'alchemy_rpc':
                records_added = await self._backfill_alchemy_data(task.start_date, task.end_date)
            elif task.source == 'price_data':
                records_added = await self._backfill_price_data(task.start_date, task.end_date)
            else:
                raise ValueError(f"Unknown source for backfill: {task.source}")
            
            # Update task completion
            task.status = "completed"
            task.completed_at = datetime.now()
            task.actual_records = records_added
            
            logger.info(f"Backfill completed: {task.task_id} - {records_added} records added")
            
        except Exception as e:
            logger.error(f"Backfill task failed: {task.task_id} - {e}")
            task.status = "failed"
            task.error_message = str(e)
            task.error_count += 1
    
    async def _backfill_graph_data(self, start_date: datetime, end_date: datetime) -> int:
        """Backfill data from Graph API"""
        # Implementation would fetch historical data from Graph API
        # This is a simplified version
        logger.info(f"Backfilling Graph data from {start_date} to {end_date}")
        await asyncio.sleep(2)  # Simulate API calls
        return 500  # Simulated record count
    
    async def _backfill_alchemy_data(self, start_date: datetime, end_date: datetime) -> int:
        """Backfill data from Alchemy RPC"""
        # Implementation would fetch historical data from Alchemy
        logger.info(f"Backfilling Alchemy data from {start_date} to {end_date}")
        await asyncio.sleep(3)  # Simulate RPC calls
        return 300  # Simulated record count
    
    async def _backfill_price_data(self, start_date: datetime, end_date: datetime) -> int:
        """Backfill price data"""
        # Implementation would fetch historical price data
        logger.info(f"Backfilling price data from {start_date} to {end_date}")
        await asyncio.sleep(1)  # Simulate API calls
        return 1000  # Simulated record count
    
    async def _estimate_backfill_records(self, source_id: str, start_date: datetime, end_date: datetime) -> int:
        """Estimate number of records for backfill"""
        try:
            source_config = self.data_sources[source_id]
            frequency_minutes = source_config['frequency_minutes']
            
            # Calculate time span
            time_span = end_date - start_date
            total_minutes = time_span.total_seconds() / 60
            
            # Estimate records
            estimated_records = int(total_minutes / frequency_minutes)
            
            return estimated_records
            
        except Exception as e:
            logger.error(f"Error estimating backfill records: {e}")
            return 1000  # Default estimate
    
    async def _generate_quality_report(self):
        """Generate and save quality report"""
        try:
            if not self.quality_history:
                return
            
            # Save latest quality metrics
            report_dir = Path("data_quality_reports")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"quality_report_{timestamp}.json"
            
            await self.export_quality_report(str(report_path))
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
    
    def _get_quality_status(self, score: float) -> str:
        """Get quality status based on score"""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "fair"
        elif score >= 0.50:
            return "poor"
        else:
            return "critical"
    
    def _empty_quality_metrics(self, source_id: str) -> DataQualityMetrics:
        """Return empty quality metrics for error cases"""
        return DataQualityMetrics(
            timestamp=datetime.now(),
            source=source_id,
            completeness_score=0.0,
            accuracy_score=0.0,
            consistency_score=0.0,
            timeliness_score=0.0,
            overall_score=0.0,
            total_expected_records=0,
            total_actual_records=0,
            missing_records=0,
            duplicate_records=0,
            invalid_records=0,
            stale_records=0,
            field_completeness={},
            field_accuracy={},
            issues_found=["Quality check failed"],
            recommendations=["Investigate data collection issues"]
        )
    
    async def stop_monitoring(self):
        """Stop data quality monitoring"""
        logger.info("🛑 Stopping data quality monitoring...")
        self.is_monitoring = False