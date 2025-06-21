"""
Data Completeness Checker
Validates data integrity and identifies missing records across all data sources
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CompletenessResult:
    """Result of completeness check"""
    source: str
    table_name: str
    check_timestamp: datetime
    
    # Completeness metrics
    expected_records: int
    actual_records: int
    missing_records: int
    completeness_percentage: float
    
    # Time coverage
    earliest_timestamp: Optional[datetime]
    latest_timestamp: Optional[datetime]
    expected_time_span: timedelta
    actual_time_span: timedelta
    time_coverage_percentage: float
    
    # Data quality
    duplicate_count: int
    null_critical_fields: int
    invalid_records: int
    
    # Missing periods
    missing_time_periods: List[Tuple[datetime, datetime]]
    largest_gap_hours: float
    
    # Field completeness
    field_completeness: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]
    priority_level: str  # low, medium, high, critical

@dataclass
class ValidationRule:
    """Data validation rule"""
    field_name: str
    rule_type: str  # not_null, range, format, custom
    parameters: Dict[str, Any]
    description: str
    severity: str  # warning, error, critical

class DataCompletenessChecker:
    """
    Comprehensive data completeness and integrity checker
    """
    
    def __init__(self, database_url: str):
        """
        Initialize completeness checker
        
        Args:
            database_url: Database connection string
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Define data sources and their expected characteristics
        self.data_sources = {
            'positions': {
                'table_name': 'positions',
                'timestamp_field': 'timestamp',
                'primary_key': 'id',
                'expected_frequency_minutes': 15,  # New position every 15 minutes on average
                'critical_fields': ['id', 'owner', 'pool', 'token0', 'token1', 'liquidity'],
                'validation_rules': [
                    ValidationRule('owner', 'format', {'pattern': r'^0x[a-fA-F0-9]{40}$'}, 'Valid Ethereum address', 'error'),
                    ValidationRule('liquidity', 'range', {'min': 0}, 'Non-negative liquidity', 'error'),
                    ValidationRule('pool', 'not_null', {}, 'Pool address required', 'critical'),
                ]
            },
            'alchemy_positions': {
                'table_name': 'alchemy_positions',
                'timestamp_field': 'timestamp',
                'primary_key': 'position_id',
                'expected_frequency_minutes': 10,
                'critical_fields': ['position_id', 'block_number', 'owner', 'pool_address'],
                'validation_rules': [
                    ValidationRule('block_number', 'range', {'min': 1}, 'Valid block number', 'error'),
                    ValidationRule('owner', 'format', {'pattern': r'^0x[a-fA-F0-9]{40}$'}, 'Valid owner address', 'error'),
                    ValidationRule('pool_address', 'not_null', {}, 'Pool address required', 'critical'),
                ]
            },
            'token_prices': {
                'table_name': 'token_prices',
                'timestamp_field': 'timestamp',
                'primary_key': ['token_address', 'timestamp'],
                'expected_frequency_minutes': 5,  # Price updates every 5 minutes
                'critical_fields': ['token_address', 'price_usd', 'timestamp'],
                'validation_rules': [
                    ValidationRule('price_usd', 'range', {'min': 0, 'max': 1000000}, 'Reasonable price range', 'warning'),
                    ValidationRule('token_address', 'format', {'pattern': r'^0x[a-fA-F0-9]{40}$'}, 'Valid token address', 'error'),
                    ValidationRule('volume_24h', 'range', {'min': 0}, 'Non-negative volume', 'warning'),
                ]
            },
            'trades': {
                'table_name': 'trades',
                'timestamp_field': 'timestamp',
                'primary_key': 'transaction_hash',
                'expected_frequency_minutes': 2,  # Trades happen frequently
                'critical_fields': ['transaction_hash', 'block_number', 'amount0', 'amount1'],
                'validation_rules': [
                    ValidationRule('transaction_hash', 'format', {'pattern': r'^0x[a-fA-F0-9]{64}$'}, 'Valid transaction hash', 'error'),
                    ValidationRule('block_number', 'range', {'min': 1}, 'Valid block number', 'error'),
                    ValidationRule('amount0', 'not_null', {}, 'Amount0 required', 'error'),
                ]
            }
        }
        
        # Completeness thresholds
        self.thresholds = {
            'critical': 0.99,    # 99% completeness for critical priority
            'high': 0.95,        # 95% for high priority
            'medium': 0.90,      # 90% for medium priority
            'low': 0.80          # 80% for low priority
        }
        
        logger.info("Data completeness checker initialized")
    
    async def check_all_sources_completeness(self, 
                                           check_period_hours: int = 24) -> Dict[str, CompletenessResult]:
        """
        Check completeness for all data sources
        
        Args:
            check_period_hours: Period to check (hours back from now)
            
        Returns:
            Dictionary of source name to completeness results
        """
        try:
            logger.info(f"🔍 Checking completeness for all sources (last {check_period_hours} hours)")
            
            results = {}
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=check_period_hours)
            
            for source_name, source_config in self.data_sources.items():
                try:
                    logger.info(f"Checking {source_name}...")
                    result = await self.check_source_completeness(source_name, start_time, end_time)
                    results[source_name] = result
                    
                    # Log summary
                    logger.info(
                        f"  {source_name}: {result.completeness_percentage:.1%} complete "
                        f"({result.actual_records}/{result.expected_records} records)"
                    )
                    
                except Exception as e:
                    logger.error(f"Error checking {source_name}: {e}")
                    continue
            
            logger.info(f"✅ Completeness check completed for {len(results)} sources")
            return results
            
        except Exception as e:
            logger.error(f"Error in check_all_sources_completeness: {e}")
            return {}
    
    async def check_source_completeness(self, 
                                      source_name: str,
                                      start_time: datetime,
                                      end_time: datetime) -> CompletenessResult:
        """
        Check completeness for a specific data source
        
        Args:
            source_name: Name of the data source
            start_time: Start time for check
            end_time: End time for check
            
        Returns:
            CompletenessResult with detailed analysis
        """
        try:
            source_config = self.data_sources.get(source_name)
            if not source_config:
                raise ValueError(f"Unknown source: {source_name}")
            
            # Fetch data for the time period
            data = await self._fetch_source_data(source_name, start_time, end_time)
            
            # Calculate expected vs actual records
            expected_records = await self._calculate_expected_records(source_config, start_time, end_time)
            actual_records = len(data)
            missing_records = max(0, expected_records - actual_records)
            completeness_percentage = actual_records / expected_records if expected_records > 0 else 0
            
            # Time coverage analysis
            time_coverage = await self._analyze_time_coverage(data, source_config, start_time, end_time)
            
            # Find missing time periods
            missing_periods = await self._find_missing_periods(data, source_config, start_time, end_time)
            
            # Calculate largest gap
            largest_gap_hours = 0
            if missing_periods:
                gaps = [(end - start).total_seconds() / 3600 for start, end in missing_periods]
                largest_gap_hours = max(gaps)
            
            # Field completeness analysis
            field_completeness = await self._analyze_field_completeness(data, source_config)
            
            # Data quality checks
            duplicate_count = await self._count_duplicates(data, source_config)
            null_critical_fields = await self._count_null_critical_fields(data, source_config)
            invalid_records = await self._count_invalid_records(data, source_config)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                source_name, completeness_percentage, missing_periods, field_completeness, 
                duplicate_count, null_critical_fields, invalid_records
            )
            
            # Determine priority level
            priority_level = self._determine_priority_level(completeness_percentage, largest_gap_hours, invalid_records)
            
            result = CompletenessResult(
                source=source_name,
                table_name=source_config['table_name'],
                check_timestamp=datetime.now(),
                expected_records=expected_records,
                actual_records=actual_records,
                missing_records=missing_records,
                completeness_percentage=completeness_percentage,
                earliest_timestamp=time_coverage['earliest'],
                latest_timestamp=time_coverage['latest'],
                expected_time_span=end_time - start_time,
                actual_time_span=time_coverage['span'],
                time_coverage_percentage=time_coverage['coverage_percentage'],
                duplicate_count=duplicate_count,
                null_critical_fields=null_critical_fields,
                invalid_records=invalid_records,
                missing_time_periods=missing_periods,
                largest_gap_hours=largest_gap_hours,
                field_completeness=field_completeness,
                recommendations=recommendations,
                priority_level=priority_level
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking completeness for {source_name}: {e}")
            return self._empty_completeness_result(source_name)
    
    async def generate_completeness_report(self, 
                                         results: Dict[str, CompletenessResult],
                                         output_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive completeness report
        
        Args:
            results: Completeness results for all sources
            output_path: Path to save the report
            
        Returns:
            Complete report dictionary
        """
        try:
            # Calculate overall system health
            if results:
                overall_completeness = np.mean([r.completeness_percentage for r in results.values()])
                critical_issues = len([r for r in results.values() if r.priority_level == 'critical'])
                high_issues = len([r for r in results.values() if r.priority_level == 'high'])
            else:
                overall_completeness = 0.0
                critical_issues = 0
                high_issues = 0
            
            # Compile all recommendations
            all_recommendations = []
            for result in results.values():
                for rec in result.recommendations:
                    all_recommendations.append({
                        'source': result.source,
                        'recommendation': rec,
                        'priority': result.priority_level
                    })
            
            # Sort recommendations by priority
            priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            all_recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'check_period': f"{list(results.values())[0].expected_time_span.total_seconds() / 3600:.1f} hours" if results else "0 hours",
                    'sources_checked': len(results),
                    'report_version': '1.0'
                },
                'system_overview': {
                    'overall_completeness': overall_completeness,
                    'system_health': self._get_system_health_status(overall_completeness),
                    'critical_issues': critical_issues,
                    'high_priority_issues': high_issues,
                    'total_missing_records': sum(r.missing_records for r in results.values()),
                    'sources_below_threshold': len([
                        r for r in results.values() 
                        if r.completeness_percentage < self.thresholds['medium']
                    ])
                },
                'source_details': {
                    source_name: {
                        'completeness_percentage': result.completeness_percentage,
                        'expected_records': result.expected_records,
                        'actual_records': result.actual_records,
                        'missing_records': result.missing_records,
                        'time_coverage_percentage': result.time_coverage_percentage,
                        'largest_gap_hours': result.largest_gap_hours,
                        'duplicate_count': result.duplicate_count,
                        'invalid_records': result.invalid_records,
                        'priority_level': result.priority_level,
                        'field_completeness': result.field_completeness,
                        'missing_periods_count': len(result.missing_time_periods),
                        'recommendations_count': len(result.recommendations)
                    }
                    for source_name, result in results.items()
                },
                'data_quality_issues': {
                    'duplicates': {
                        source: result.duplicate_count 
                        for source, result in results.items() 
                        if result.duplicate_count > 0
                    },
                    'null_critical_fields': {
                        source: result.null_critical_fields 
                        for source, result in results.items() 
                        if result.null_critical_fields > 0
                    },
                    'invalid_records': {
                        source: result.invalid_records 
                        for source, result in results.items() 
                        if result.invalid_records > 0
                    }
                },
                'time_gap_analysis': {
                    'sources_with_gaps': len([
                        r for r in results.values() 
                        if r.missing_time_periods
                    ]),
                    'largest_gaps': {
                        source: result.largest_gap_hours
                        for source, result in results.items()
                        if result.largest_gap_hours > 1  # Gaps larger than 1 hour
                    },
                    'gap_details': {
                        source: [
                            {
                                'start': start.isoformat(),
                                'end': end.isoformat(),
                                'duration_hours': (end - start).total_seconds() / 3600
                            }
                            for start, end in result.missing_time_periods
                        ]
                        for source, result in results.items()
                        if result.missing_time_periods
                    }
                },
                'recommendations': {
                    'immediate_actions': [
                        rec for rec in all_recommendations[:10]  # Top 10 priority actions
                        if rec['priority'] in ['critical', 'high']
                    ],
                    'all_recommendations': all_recommendations,
                    'by_priority': {
                        priority: [
                            rec for rec in all_recommendations 
                            if rec['priority'] == priority
                        ]
                        for priority in ['critical', 'high', 'medium', 'low']
                    }
                },
                'detailed_results': {
                    source_name: {
                        'check_timestamp': result.check_timestamp.isoformat(),
                        'table_name': result.table_name,
                        'earliest_timestamp': result.earliest_timestamp.isoformat() if result.earliest_timestamp else None,
                        'latest_timestamp': result.latest_timestamp.isoformat() if result.latest_timestamp else None,
                        'expected_time_span_hours': result.expected_time_span.total_seconds() / 3600,
                        'actual_time_span_hours': result.actual_time_span.total_seconds() / 3600,
                        'missing_time_periods': [
                            {
                                'start': start.isoformat(),
                                'end': end.isoformat(),
                                'duration_hours': (end - start).total_seconds() / 3600
                            }
                            for start, end in result.missing_time_periods
                        ],
                        'field_completeness': result.field_completeness,
                        'recommendations': result.recommendations
                    }
                    for source_name, result in results.items()
                }
            }
            
            # Save report
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Completeness report saved to {output_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating completeness report: {e}")
            return {}
    
    async def get_critical_issues(self) -> List[Dict[str, Any]]:
        """
        Get list of critical data completeness issues requiring immediate attention
        
        Returns:
            List of critical issues with details
        """
        try:
            # Check last 6 hours for critical issues
            results = await self.check_all_sources_completeness(check_period_hours=6)
            
            critical_issues = []
            
            for source_name, result in results.items():
                # Critical completeness issues
                if result.completeness_percentage < self.thresholds['critical']:
                    critical_issues.append({
                        'type': 'low_completeness',
                        'source': source_name,
                        'severity': 'critical',
                        'description': f"Completeness below critical threshold: {result.completeness_percentage:.1%}",
                        'missing_records': result.missing_records,
                        'action_required': 'immediate_backfill'
                    })
                
                # Large data gaps
                if result.largest_gap_hours > 12:
                    critical_issues.append({
                        'type': 'large_data_gap',
                        'source': source_name,
                        'severity': 'high',
                        'description': f"Large data gap detected: {result.largest_gap_hours:.1f} hours",
                        'gap_hours': result.largest_gap_hours,
                        'action_required': 'investigate_data_source'
                    })
                
                # High number of invalid records
                if result.invalid_records > (result.actual_records * 0.05):  # More than 5% invalid
                    critical_issues.append({
                        'type': 'high_invalid_records',
                        'source': source_name,
                        'severity': 'high',
                        'description': f"High number of invalid records: {result.invalid_records}",
                        'invalid_count': result.invalid_records,
                        'action_required': 'data_validation_review'
                    })
                
                # Missing critical fields
                if result.null_critical_fields > 0:
                    critical_issues.append({
                        'type': 'missing_critical_fields',
                        'source': source_name,
                        'severity': 'critical',
                        'description': f"Critical fields missing: {result.null_critical_fields} records",
                        'affected_records': result.null_critical_fields,
                        'action_required': 'data_collection_fix'
                    })
            
            # Sort by severity
            severity_order = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
            critical_issues.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
            
            return critical_issues
            
        except Exception as e:
            logger.error(f"Error getting critical issues: {e}")
            return []
    
    # Helper methods
    
    async def _fetch_source_data(self, source_name: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch data for a source within time range"""
        try:
            source_config = self.data_sources[source_name]
            table_name = source_config['table_name']
            timestamp_field = source_config['timestamp_field']
            
            query = text(f"""
                SELECT * FROM {table_name}
                WHERE {timestamp_field} >= :start_time 
                AND {timestamp_field} <= :end_time
                ORDER BY {timestamp_field}
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                # Convert timestamp column to datetime
                if timestamp_field in df.columns:
                    df[timestamp_field] = pd.to_datetime(df[timestamp_field])
                
                return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {source_name}: {e}")
            return pd.DataFrame()
    
    async def _calculate_expected_records(self, source_config: Dict, start_time: datetime, end_time: datetime) -> int:
        """Calculate expected number of records for a time period"""
        try:
            frequency_minutes = source_config['expected_frequency_minutes']
            time_span = end_time - start_time
            total_minutes = time_span.total_seconds() / 60
            
            expected_records = int(total_minutes / frequency_minutes)
            return max(0, expected_records)
            
        except Exception as e:
            logger.error(f"Error calculating expected records: {e}")
            return 0
    
    async def _analyze_time_coverage(self, data: pd.DataFrame, source_config: Dict, 
                                   start_time: datetime, end_time: datetime) -> Dict:
        """Analyze time coverage of the data"""
        try:
            timestamp_field = source_config['timestamp_field']
            
            if data.empty or timestamp_field not in data.columns:
                return {
                    'earliest': None,
                    'latest': None,
                    'span': timedelta(0),
                    'coverage_percentage': 0.0
                }
            
            earliest = data[timestamp_field].min()
            latest = data[timestamp_field].max()
            actual_span = latest - earliest if earliest and latest else timedelta(0)
            expected_span = end_time - start_time
            
            coverage_percentage = min(1.0, actual_span.total_seconds() / expected_span.total_seconds()) if expected_span.total_seconds() > 0 else 0
            
            return {
                'earliest': earliest,
                'latest': latest,
                'span': actual_span,
                'coverage_percentage': coverage_percentage
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time coverage: {e}")
            return {'earliest': None, 'latest': None, 'span': timedelta(0), 'coverage_percentage': 0.0}
    
    async def _find_missing_periods(self, data: pd.DataFrame, source_config: Dict,
                                  start_time: datetime, end_time: datetime) -> List[Tuple[datetime, datetime]]:
        """Find missing time periods in the data"""
        try:
            timestamp_field = source_config['timestamp_field']
            frequency_minutes = source_config['expected_frequency_minutes']
            
            if data.empty or timestamp_field not in data.columns:
                return [(start_time, end_time)]
            
            # Sort by timestamp
            data_sorted = data.sort_values(timestamp_field)
            timestamps = data_sorted[timestamp_field].tolist()
            
            missing_periods = []
            expected_gap = timedelta(minutes=frequency_minutes * 2)  # Allow 2x frequency as acceptable gap
            
            # Check for gap at the beginning
            if timestamps[0] > start_time + expected_gap:
                missing_periods.append((start_time, timestamps[0]))
            
            # Check for gaps between records
            for i in range(len(timestamps) - 1):
                current_time = timestamps[i]
                next_time = timestamps[i + 1]
                gap = next_time - current_time
                
                if gap > expected_gap:
                    missing_periods.append((current_time, next_time))
            
            # Check for gap at the end
            if timestamps[-1] < end_time - expected_gap:
                missing_periods.append((timestamps[-1], end_time))
            
            return missing_periods
            
        except Exception as e:
            logger.error(f"Error finding missing periods: {e}")
            return []
    
    async def _analyze_field_completeness(self, data: pd.DataFrame, source_config: Dict) -> Dict[str, float]:
        """Analyze completeness of individual fields"""
        try:
            if data.empty:
                return {}
            
            field_completeness = {}
            total_records = len(data)
            
            for field in source_config['critical_fields']:
                if field in data.columns:
                    non_null_count = data[field].notna().sum()
                    completeness = non_null_count / total_records
                    field_completeness[field] = completeness
                else:
                    field_completeness[field] = 0.0
            
            return field_completeness
            
        except Exception as e:
            logger.error(f"Error analyzing field completeness: {e}")
            return {}
    
    async def _count_duplicates(self, data: pd.DataFrame, source_config: Dict) -> int:
        """Count duplicate records"""
        try:
            if data.empty:
                return 0
            
            primary_key = source_config['primary_key']
            
            if isinstance(primary_key, list):
                # Composite primary key
                if all(col in data.columns for col in primary_key):
                    return data.duplicated(subset=primary_key).sum()
            else:
                # Single primary key
                if primary_key in data.columns:
                    return data.duplicated(subset=[primary_key]).sum()
            
            return 0
            
        except Exception as e:
            logger.error(f"Error counting duplicates: {e}")
            return 0
    
    async def _count_null_critical_fields(self, data: pd.DataFrame, source_config: Dict) -> int:
        """Count records with null critical fields"""
        try:
            if data.empty:
                return 0
            
            null_count = 0
            for field in source_config['critical_fields']:
                if field in data.columns:
                    null_count += data[field].isna().sum()
            
            return int(null_count)
            
        except Exception as e:
            logger.error(f"Error counting null critical fields: {e}")
            return 0
    
    async def _count_invalid_records(self, data: pd.DataFrame, source_config: Dict) -> int:
        """Count invalid records based on validation rules"""
        try:
            if data.empty:
                return 0
            
            invalid_count = 0
            
            for rule in source_config.get('validation_rules', []):
                field_name = rule.field_name
                
                if field_name not in data.columns:
                    continue
                
                if rule.rule_type == 'not_null':
                    invalid_count += data[field_name].isna().sum()
                
                elif rule.rule_type == 'range':
                    min_val = rule.parameters.get('min')
                    max_val = rule.parameters.get('max')
                    
                    numeric_data = pd.to_numeric(data[field_name], errors='coerce')
                    
                    if min_val is not None:
                        invalid_count += (numeric_data < min_val).sum()
                    if max_val is not None:
                        invalid_count += (numeric_data > max_val).sum()
                
                elif rule.rule_type == 'format':
                    pattern = rule.parameters.get('pattern')
                    if pattern:
                        invalid_count += (~data[field_name].astype(str).str.match(pattern, na=False)).sum()
            
            return int(invalid_count)
            
        except Exception as e:
            logger.error(f"Error counting invalid records: {e}")
            return 0
    
    async def _generate_recommendations(self, source_name: str, completeness_percentage: float,
                                      missing_periods: List, field_completeness: Dict,
                                      duplicate_count: int, null_critical_fields: int,
                                      invalid_records: int) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Completeness recommendations
        if completeness_percentage < self.thresholds['high']:
            recommendations.append(f"Trigger backfill for {source_name} - completeness is {completeness_percentage:.1%}")
        
        # Gap recommendations
        if missing_periods:
            large_gaps = [gap for gap in missing_periods if (gap[1] - gap[0]).total_seconds() / 3600 > 6]
            if large_gaps:
                recommendations.append(f"Investigate data collection issues - {len(large_gaps)} large gaps detected")
        
        # Field completeness recommendations
        low_completeness_fields = [field for field, comp in field_completeness.items() if comp < 0.95]
        if low_completeness_fields:
            recommendations.append(f"Improve data collection for fields: {', '.join(low_completeness_fields[:3])}")
        
        # Data quality recommendations
        if duplicate_count > 0:
            recommendations.append(f"Remove {duplicate_count} duplicate records from {source_name}")
        
        if null_critical_fields > 0:
            recommendations.append(f"Fix {null_critical_fields} records with missing critical fields")
        
        if invalid_records > 0:
            recommendations.append(f"Validate and correct {invalid_records} invalid records")
        
        return recommendations
    
    def _determine_priority_level(self, completeness_percentage: float, largest_gap_hours: float, invalid_records: int) -> str:
        """Determine priority level based on metrics"""
        if (completeness_percentage < self.thresholds['critical'] or 
            largest_gap_hours > 24 or 
            invalid_records > 1000):
            return 'critical'
        elif (completeness_percentage < self.thresholds['high'] or 
              largest_gap_hours > 12 or 
              invalid_records > 100):
            return 'high'
        elif (completeness_percentage < self.thresholds['medium'] or 
              largest_gap_hours > 6 or 
              invalid_records > 10):
            return 'medium'
        else:
            return 'low'
    
    def _get_system_health_status(self, overall_completeness: float) -> str:
        """Get overall system health status"""
        if overall_completeness >= 0.95:
            return "healthy"
        elif overall_completeness >= 0.85:
            return "warning"
        elif overall_completeness >= 0.70:
            return "degraded"
        else:
            return "critical"
    
    def _empty_completeness_result(self, source_name: str) -> CompletenessResult:
        """Return empty result for error cases"""
        return CompletenessResult(
            source=source_name,
            table_name="unknown",
            check_timestamp=datetime.now(),
            expected_records=0,
            actual_records=0,
            missing_records=0,
            completeness_percentage=0.0,
            earliest_timestamp=None,
            latest_timestamp=None,
            expected_time_span=timedelta(0),
            actual_time_span=timedelta(0),
            time_coverage_percentage=0.0,
            duplicate_count=0,
            null_critical_fields=0,
            invalid_records=0,
            missing_time_periods=[],
            largest_gap_hours=0.0,
            field_completeness={},
            recommendations=["Completeness check failed"],
            priority_level="critical"
        )

# Demo function
async def demo_completeness_checker():
    """Demonstrate the completeness checker"""
    logger.info("🎯 Starting Data Completeness Checker Demo")
    
    # Initialize checker (with mock database)
    checker = DataCompletenessChecker("postgresql://mock:mock@localhost/mock")
    
    # Demo: Check all sources
    logger.info("🔍 Demo: Checking completeness for all sources...")
    results = await checker.check_all_sources_completeness(check_period_hours=24)
    
    if results:
        logger.info(f"✅ Checked {len(results)} sources:")
        for source, result in results.items():
            logger.info(f"  📊 {source}: {result.completeness_percentage:.1%} complete, {result.priority_level} priority")
    
    # Demo: Generate report
    logger.info("📄 Demo: Generating completeness report...")
    report_path = "data_completeness_reports/demo_report.json"
    report = await checker.generate_completeness_report(results, report_path)
    
    if report:
        system_health = report['system_overview']['system_health']
        critical_issues = report['system_overview']['critical_issues']
        logger.info(f"  📈 System health: {system_health}")
        logger.info(f"  🚨 Critical issues: {critical_issues}")
    
    # Demo: Get critical issues
    logger.info("🚨 Demo: Identifying critical issues...")
    critical_issues = await checker.get_critical_issues()
    
    if critical_issues:
        logger.info(f"  ⚠️ Found {len(critical_issues)} critical issues:")
        for issue in critical_issues[:3]:  # Show top 3
            logger.info(f"    - {issue['type']}: {issue['description']}")
    else:
        logger.info("  ✅ No critical issues found!")
    
    logger.info("🎉 Data Completeness Checker Demo completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_completeness_checker())