"""Data Quality and Validation System for DexBrain Network"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .schemas import (
    AgentSubmission, LiquidityPosition, PerformanceMetrics,
    DataValidator, TokenInfo, PoolInfo
)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    ERROR = "error"      # Blocks submission
    WARNING = "warning"  # Flags for review
    INFO = "info"       # Informational only


class QualityMetric(Enum):
    """Data quality metrics"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"


@dataclass
class ValidationIssue:
    """Data validation issue"""
    severity: ValidationSeverity
    metric: QualityMetric
    field: str
    message: str
    suggested_fix: Optional[str] = None


@dataclass
class QualityReport:
    """Data quality assessment report"""
    agent_id: str
    submission_id: str
    timestamp: str
    overall_score: float  # 0-100
    quality_scores: Dict[str, float]  # Per metric scores
    issues: List[ValidationIssue]
    is_accepted: bool
    recommendations: List[str]


class DataQualityEngine:
    """Advanced data quality validation and scoring engine"""
    
    def __init__(self):
        self.known_tokens: Dict[str, TokenInfo] = {}  # Token registry
        self.known_pools: Dict[str, PoolInfo] = {}    # Pool registry
        self.agent_history: Dict[str, List[Dict]] = {}  # Agent submission history
        self.network_stats: Dict[str, Any] = {}       # Network-wide statistics
    
    def validate_submission(self, submission: AgentSubmission) -> QualityReport:
        """Comprehensive validation of agent data submission
        
        Args:
            submission: Agent data submission to validate
            
        Returns:
            Quality report with validation results
        """
        issues = []
        quality_scores = {}
        
        # Basic schema validation
        schema_issues = self._validate_schema(submission)
        issues.extend(schema_issues)
        
        # Quality metric validations
        quality_scores[QualityMetric.COMPLETENESS.value] = self._assess_completeness(submission, issues)
        quality_scores[QualityMetric.ACCURACY.value] = self._assess_accuracy(submission, issues)
        quality_scores[QualityMetric.CONSISTENCY.value] = self._assess_consistency(submission, issues)
        quality_scores[QualityMetric.TIMELINESS.value] = self._assess_timeliness(submission, issues)
        quality_scores[QualityMetric.VALIDITY.value] = self._assess_validity(submission, issues)
        
        # Calculate overall score
        overall_score = np.mean(list(quality_scores.values()))
        
        # Determine acceptance
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        is_accepted = error_count == 0 and overall_score >= 60.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, quality_scores)
        
        report = QualityReport(
            agent_id=submission.agent_id,
            submission_id=submission.submission_id,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            quality_scores=quality_scores,
            issues=issues,
            is_accepted=is_accepted,
            recommendations=recommendations
        )
        
        # Update agent history
        self._update_agent_history(submission.agent_id, report)
        
        return report
    
    def _validate_schema(self, submission: AgentSubmission) -> List[ValidationIssue]:
        """Validate against data schemas"""
        issues = []
        
        # Basic submission validation
        if not DataValidator.validate_agent_submission(submission.to_dict()):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                metric=QualityMetric.VALIDITY,
                field="submission",
                message="Submission does not conform to required schema",
                suggested_fix="Check required fields and data types"
            ))
        
        # Validate individual positions
        for i, position in enumerate(submission.positions):
            if not DataValidator.validate_liquidity_position(position.to_dict()):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    metric=QualityMetric.VALIDITY,
                    field=f"positions[{i}]",
                    message="Position data invalid",
                    suggested_fix="Check position fields and data types"
                ))
        
        # Validate performance metrics
        for i, metrics in enumerate(submission.performance_metrics):
            if not DataValidator.validate_performance_metrics(metrics.to_dict()):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    metric=QualityMetric.VALIDITY,
                    field=f"performance_metrics[{i}]",
                    message="Performance metrics invalid",
                    suggested_fix="Check metrics fields and data types"
                ))
        
        return issues
    
    def _assess_completeness(self, submission: AgentSubmission, issues: List[ValidationIssue]) -> float:
        """Assess data completeness"""
        total_fields = 0
        complete_fields = 0
        
        # Check submission-level completeness
        submission_fields = ['agent_id', 'blockchain', 'dex_protocol', 'positions', 'performance_metrics']
        for field in submission_fields:
            total_fields += 1
            if hasattr(submission, field) and getattr(submission, field) is not None:
                complete_fields += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    metric=QualityMetric.COMPLETENESS,
                    field=field,
                    message=f"Missing or null field: {field}"
                ))
        
        # Check position completeness
        for i, position in enumerate(submission.positions):
            position_fields = [
                'position_id', 'pool', 'liquidity_amount', 'token0_amount',
                'token1_amount', 'position_value_usd', 'entry_price'
            ]
            for field in position_fields:
                total_fields += 1
                if hasattr(position, field) and getattr(position, field) is not None:
                    complete_fields += 1
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        metric=QualityMetric.COMPLETENESS,
                        field=f"positions[{i}].{field}",
                        message=f"Missing position field: {field}"
                    ))
        
        # Check optional but valuable fields
        optional_valuable_fields = ['current_price', 'price_range_lower', 'price_range_upper']
        for i, position in enumerate(submission.positions):
            for field in optional_valuable_fields:
                total_fields += 0.5  # Weight optional fields less
                if hasattr(position, field) and getattr(position, field) is not None:
                    complete_fields += 0.5
        
        return (complete_fields / total_fields) * 100 if total_fields > 0 else 0
    
    def _assess_accuracy(self, submission: AgentSubmission, issues: List[ValidationIssue]) -> float:
        """Assess data accuracy using various heuristics"""
        accuracy_score = 100.0
        total_checks = 0
        passed_checks = 0
        
        for i, position in enumerate(submission.positions):
            total_checks += 1
            
            # Check for reasonable liquidity amounts
            if position.position_value_usd < 0:
                accuracy_score -= 10
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    metric=QualityMetric.ACCURACY,
                    field=f"positions[{i}].position_value_usd",
                    message="Negative position value"
                ))
            else:
                passed_checks += 1
            
            # Check for reasonable price ranges
            if (position.price_range_lower and position.price_range_upper and 
                position.price_range_lower >= position.price_range_upper):
                accuracy_score -= 5
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    metric=QualityMetric.ACCURACY,
                    field=f"positions[{i}].price_range",
                    message="Invalid price range (lower >= upper)"
                ))
            else:
                passed_checks += 1
            
            total_checks += 1
        
        # Check performance metrics accuracy
        for i, metrics in enumerate(submission.performance_metrics):
            total_checks += 1
            
            # APR reasonableness check
            if abs(metrics.apr) > 10000:  # 10000% APR seems unreasonable
                accuracy_score -= 15
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    metric=QualityMetric.ACCURACY,
                    field=f"performance_metrics[{i}].apr",
                    message=f"Extremely high APR: {metrics.apr}%",
                    suggested_fix="Verify APR calculation"
                ))
            else:
                passed_checks += 1
            
            # Duration reasonableness
            if metrics.duration_hours < 0 or metrics.duration_hours > 8760:  # More than a year
                accuracy_score -= 10
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    metric=QualityMetric.ACCURACY,
                    field=f"performance_metrics[{i}].duration_hours",
                    message=f"Unusual duration: {metrics.duration_hours} hours"
                ))
            else:
                passed_checks += 1
            
            total_checks += 1
        
        # Combine heuristic score with check ratio
        check_ratio_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
        return min(accuracy_score, check_ratio_score)
    
    def _assess_consistency(self, submission: AgentSubmission, issues: List[ValidationIssue]) -> float:
        """Assess internal data consistency"""
        consistency_score = 100.0
        
        # Check position-performance consistency
        position_ids = {pos.position_id for pos in submission.positions}
        metric_position_ids = {met.position_id for met in submission.performance_metrics}
        
        missing_metrics = position_ids - metric_position_ids
        if missing_metrics:
            consistency_score -= 20
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                metric=QualityMetric.CONSISTENCY,
                field="performance_metrics",
                message=f"Missing performance metrics for positions: {missing_metrics}"
            ))
        
        orphaned_metrics = metric_position_ids - position_ids
        if orphaned_metrics:
            consistency_score -= 15
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                metric=QualityMetric.CONSISTENCY,
                field="positions",
                message=f"Performance metrics without corresponding positions: {orphaned_metrics}"
            ))
        
        # Check agent historical consistency
        agent_history = self.agent_history.get(submission.agent_id, [])
        if agent_history:
            prev_blockchain = agent_history[-1].get('blockchain')
            if prev_blockchain and prev_blockchain != submission.blockchain:
                consistency_score -= 5
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    metric=QualityMetric.CONSISTENCY,
                    field="blockchain",
                    message=f"Blockchain changed from {prev_blockchain} to {submission.blockchain}"
                ))
        
        return max(consistency_score, 0)
    
    def _assess_timeliness(self, submission: AgentSubmission, issues: List[ValidationIssue]) -> float:
        """Assess data timeliness"""
        timeliness_score = 100.0
        current_time = datetime.now()
        
        # Check submission timestamp
        try:
            submission_time = datetime.fromisoformat(submission.timestamp)
            time_diff = abs((current_time - submission_time).total_seconds())
            
            if time_diff > 3600:  # More than 1 hour old
                timeliness_score -= 20
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    metric=QualityMetric.TIMELINESS,
                    field="timestamp",
                    message=f"Submission is {time_diff/3600:.1f} hours old"
                ))
        except ValueError:
            timeliness_score -= 30
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                metric=QualityMetric.TIMELINESS,
                field="timestamp",
                message="Invalid timestamp format"
            ))
        
        # Check position timestamps
        for i, position in enumerate(submission.positions):
            if position.updated_at:
                try:
                    pos_time = datetime.fromisoformat(position.updated_at)
                    pos_age = (current_time - pos_time).total_seconds()
                    
                    if pos_age > 86400:  # More than 24 hours
                        timeliness_score -= 5
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            metric=QualityMetric.TIMELINESS,
                            field=f"positions[{i}].updated_at",
                            message=f"Position data is {pos_age/3600:.1f} hours old"
                        ))
                except ValueError:
                    timeliness_score -= 10
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        metric=QualityMetric.TIMELINESS,
                        field=f"positions[{i}].updated_at",
                        message="Invalid position timestamp"
                    ))
        
        return max(timeliness_score, 0)
    
    def _assess_validity(self, submission: AgentSubmission, issues: List[ValidationIssue]) -> float:
        """Assess business logic validity"""
        validity_score = 100.0
        
        # Validate token addresses format (basic check)
        for i, position in enumerate(submission.positions):
            if position.pool and position.pool.token0:
                if not self._is_valid_address(position.pool.token0.address):
                    validity_score -= 15
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        metric=QualityMetric.VALIDITY,
                        field=f"positions[{i}].pool.token0.address",
                        message="Invalid token0 address format"
                    ))
            
            if position.pool and position.pool.token1:
                if not self._is_valid_address(position.pool.token1.address):
                    validity_score -= 15
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        metric=QualityMetric.VALIDITY,
                        field=f"positions[{i}].pool.token1.address",
                        message="Invalid token1 address format"
                    ))
        
        # Validate mathematical relationships
        for i, metrics in enumerate(submission.performance_metrics):
            # Check if net profit calculation is consistent
            expected_net = metrics.total_return_usd - metrics.gas_costs_usd
            if abs(metrics.net_profit_usd - expected_net) > 0.01:
                validity_score -= 10
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    metric=QualityMetric.VALIDITY,
                    field=f"performance_metrics[{i}].net_profit_usd",
                    message="Net profit calculation inconsistent",
                    suggested_fix="Verify: net_profit = total_return - gas_costs"
                ))
        
        return max(validity_score, 0)
    
    def _is_valid_address(self, address: str) -> bool:
        """Basic address format validation"""
        if not address:
            return False
        
        # Ethereum-style addresses
        if address.startswith('0x') and len(address) == 42:
            return True
        
        # Solana addresses (base58, ~44 chars)
        if 32 <= len(address) <= 44 and address.replace('1', '').replace('0', '').isalnum():
            return True
        
        return False
    
    def _generate_recommendations(
        self,
        issues: List[ValidationIssue],
        quality_scores: Dict[str, float]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Low completeness recommendations
        if quality_scores.get(QualityMetric.COMPLETENESS.value, 100) < 80:
            recommendations.append("Improve data completeness by including optional fields like price ranges and current prices")
        
        # Low accuracy recommendations
        if quality_scores.get(QualityMetric.ACCURACY.value, 100) < 70:
            recommendations.append("Review calculation methods and validate extreme values before submission")
        
        # Consistency issues
        consistency_issues = [i for i in issues if i.metric == QualityMetric.CONSISTENCY]
        if consistency_issues:
            recommendations.append("Ensure all positions have corresponding performance metrics")
        
        # High-severity issues
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        if error_issues:
            recommendations.append("Address all error-level validation issues before resubmission")
        
        # General recommendations based on overall score
        overall_score = np.mean(list(quality_scores.values()))
        if overall_score < 60:
            recommendations.append("Consider implementing additional data validation on the agent side")
        elif overall_score < 80:
            recommendations.append("Good data quality - focus on addressing remaining warnings")
        else:
            recommendations.append("Excellent data quality - keep up the good work!")
        
        return recommendations
    
    def _update_agent_history(self, agent_id: str, report: QualityReport) -> None:
        """Update agent's data quality history"""
        if agent_id not in self.agent_history:
            self.agent_history[agent_id] = []
        
        # Keep last 100 submissions
        history_entry = {
            'timestamp': report.timestamp,
            'overall_score': report.overall_score,
            'blockchain': getattr(report, 'blockchain', None),
            'is_accepted': report.is_accepted,
            'issue_count': len(report.issues)
        }
        
        self.agent_history[agent_id].append(history_entry)
        self.agent_history[agent_id] = self.agent_history[agent_id][-100:]
    
    def get_agent_quality_trend(self, agent_id: str, days: int = 30) -> Dict[str, Any]:
        """Get agent's data quality trend over time"""
        if agent_id not in self.agent_history:
            return {'trend': 'no_data', 'average_score': 0, 'submissions': 0}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in self.agent_history[agent_id]
            if datetime.fromisoformat(h['timestamp']) >= cutoff_date
        ]
        
        if not recent_history:
            return {'trend': 'no_recent_data', 'average_score': 0, 'submissions': 0}
        
        scores = [h['overall_score'] for h in recent_history]
        avg_score = np.mean(scores)
        
        # Calculate trend
        if len(scores) > 1:
            recent_avg = np.mean(scores[-5:]) if len(scores) >= 5 else np.mean(scores[-len(scores)//2:])
            early_avg = np.mean(scores[:5]) if len(scores) >= 5 else np.mean(scores[:len(scores)//2])
            
            if recent_avg > early_avg + 5:
                trend = 'improving'
            elif recent_avg < early_avg - 5:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'average_score': avg_score,
            'submissions': len(recent_history),
            'acceptance_rate': sum(1 for h in recent_history if h['is_accepted']) / len(recent_history)
        }