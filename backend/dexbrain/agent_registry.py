"""Agent Registration and Tracking System for DexBrain Network"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .config import Config


class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class RiskProfile(Enum):
    """Agent risk profile enumeration"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    HYPER_AGGRESSIVE = "hyper_aggressive"
    CUSTOM = "custom"


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    total_positions: int = 0
    successful_positions: int = 0
    total_volume: float = 0.0
    total_fees_earned: float = 0.0
    total_impermanent_loss: float = 0.0
    average_apr: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    last_updated: str = ""


@dataclass
class AgentInfo:
    """Complete agent information"""
    agent_id: str
    status: AgentStatus
    risk_profile: RiskProfile
    registration_time: str
    last_active: str
    api_key_hash: str
    metadata: Dict[str, Any]
    metrics: AgentMetrics
    supported_blockchains: List[str]
    supported_dexs: List[str]
    version: str
    performance_score: float = 0.0


class AgentRegistry:
    """Registry for tracking and managing agents in the network"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Config.KNOWLEDGE_DB_PATH / "agent_registry.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.agents: Dict[str, AgentInfo] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load agent registry from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to AgentInfo objects
                for agent_id, agent_data in data.items():
                    metrics_data = agent_data.get('metrics', {})
                    metrics = AgentMetrics(**metrics_data)
                    
                    agent_info = AgentInfo(
                        agent_id=agent_data['agent_id'],
                        status=AgentStatus(agent_data['status']),
                        risk_profile=RiskProfile(agent_data['risk_profile']),
                        registration_time=agent_data['registration_time'],
                        last_active=agent_data['last_active'],
                        api_key_hash=agent_data['api_key_hash'],
                        metadata=agent_data['metadata'],
                        metrics=metrics,
                        supported_blockchains=agent_data['supported_blockchains'],
                        supported_dexs=agent_data['supported_dexs'],
                        version=agent_data['version'],
                        performance_score=agent_data.get('performance_score', 0.0)
                    )
                    
                    self.agents[agent_id] = agent_info
                    
            except Exception as e:
                print(f"Error loading agent registry: {e}")
                self.agents = {}
    
    def _save_registry(self) -> None:
        """Save agent registry to storage"""
        try:
            # Convert to serializable format
            data = {}
            for agent_id, agent_info in self.agents.items():
                data[agent_id] = {
                    'agent_id': agent_info.agent_id,
                    'status': agent_info.status.value,
                    'risk_profile': agent_info.risk_profile.value,
                    'registration_time': agent_info.registration_time,
                    'last_active': agent_info.last_active,
                    'api_key_hash': agent_info.api_key_hash,
                    'metadata': agent_info.metadata,
                    'metrics': asdict(agent_info.metrics),
                    'supported_blockchains': agent_info.supported_blockchains,
                    'supported_dexs': agent_info.supported_dexs,
                    'version': agent_info.version,
                    'performance_score': agent_info.performance_score
                }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving agent registry: {e}")
    
    def register_agent(
        self,
        agent_id: str,
        api_key_hash: str,
        risk_profile: str,
        supported_blockchains: List[str],
        supported_dexs: List[str],
        version: str = "1.0.0",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Register a new agent
        
        Args:
            agent_id: Unique identifier for the agent
            api_key_hash: Hashed API key
            risk_profile: Agent's risk profile
            supported_blockchains: List of supported blockchains
            supported_dexs: List of supported DEXs
            version: Agent software version
            metadata: Additional metadata
            
        Returns:
            True if registration successful
        """
        if agent_id in self.agents:
            return False  # Agent already exists
        
        try:
            risk_profile_enum = RiskProfile(risk_profile.lower())
        except ValueError:
            risk_profile_enum = RiskProfile.CUSTOM
        
        agent_info = AgentInfo(
            agent_id=agent_id,
            status=AgentStatus.ACTIVE,
            risk_profile=risk_profile_enum,
            registration_time=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
            api_key_hash=api_key_hash,
            metadata=metadata or {},
            metrics=AgentMetrics(),
            supported_blockchains=supported_blockchains,
            supported_dexs=supported_dexs,
            version=version
        )
        
        self.agents[agent_id] = agent_info
        self._save_registry()
        return True
    
    def update_agent_activity(self, agent_id: str) -> bool:
        """Update agent's last active timestamp
        
        Args:
            agent_id: Agent to update
            
        Returns:
            True if updated successfully
        """
        if agent_id not in self.agents:
            return False
        
        self.agents[agent_id].last_active = datetime.now().isoformat()
        self._save_registry()
        return True
    
    def update_agent_metrics(
        self,
        agent_id: str,
        metrics_update: Dict[str, Any]
    ) -> bool:
        """Update agent performance metrics
        
        Args:
            agent_id: Agent to update
            metrics_update: Dictionary of metrics to update
            
        Returns:
            True if updated successfully
        """
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Update metrics
        for key, value in metrics_update.items():
            if hasattr(agent.metrics, key):
                setattr(agent.metrics, key, value)
        
        agent.metrics.last_updated = datetime.now().isoformat()
        
        # Recalculate performance score
        agent.performance_score = self._calculate_performance_score(agent.metrics)
        
        self._save_registry()
        return True
    
    def _calculate_performance_score(self, metrics: AgentMetrics) -> float:
        """Calculate performance score for an agent
        
        Args:
            metrics: Agent metrics
            
        Returns:
            Performance score (0-100)
        """
        if metrics.total_positions == 0:
            return 0.0
        
        # Weighted scoring components
        success_rate_score = metrics.win_rate * 30  # 30% weight
        apr_score = min(metrics.average_apr * 10, 25)  # 25% weight, capped at 25
        sharpe_score = min(metrics.sharpe_ratio * 10, 20)  # 20% weight, capped at 20
        drawdown_score = max(20 - (metrics.max_drawdown * 100), 0)  # 20% weight, inverse
        volume_score = min(metrics.total_volume / 1000000 * 5, 5)  # 5% weight, per $1M
        
        total_score = success_rate_score + apr_score + sharpe_score + drawdown_score + volume_score
        return min(total_score, 100.0)
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information
        
        Args:
            agent_id: Agent to retrieve
            
        Returns:
            Agent information if found
        """
        return self.agents.get(agent_id)
    
    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        risk_profile: Optional[RiskProfile] = None,
        blockchain: Optional[str] = None
    ) -> List[AgentInfo]:
        """List agents with optional filters
        
        Args:
            status: Filter by status
            risk_profile: Filter by risk profile
            blockchain: Filter by supported blockchain
            
        Returns:
            List of matching agents
        """
        agents = list(self.agents.values())
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        if risk_profile:
            agents = [a for a in agents if a.risk_profile == risk_profile]
        
        if blockchain:
            agents = [a for a in agents if blockchain in a.supported_blockchains]
        
        return agents
    
    def get_top_performers(self, limit: int = 10) -> List[AgentInfo]:
        """Get top performing agents
        
        Args:
            limit: Maximum number of agents to return
            
        Returns:
            List of top performing agents
        """
        agents = list(self.agents.values())
        agents.sort(key=lambda a: a.performance_score, reverse=True)
        return agents[:limit]
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network-wide statistics
        
        Returns:
            Dictionary of network statistics
        """
        if not self.agents:
            return {
                'total_agents': 0,
                'active_agents': 0,
                'total_volume': 0.0,
                'total_positions': 0,
                'average_performance_score': 0.0,
                'risk_profile_distribution': {},
                'blockchain_coverage': {},
                'dex_coverage': {}
            }
        
        active_agents = [a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]
        
        # Calculate aggregated metrics
        total_volume = sum(a.metrics.total_volume for a in active_agents)
        total_positions = sum(a.metrics.total_positions for a in active_agents)
        avg_performance = sum(a.performance_score for a in active_agents) / len(active_agents)
        
        # Risk profile distribution
        risk_distribution = {}
        for agent in active_agents:
            profile = agent.risk_profile.value
            risk_distribution[profile] = risk_distribution.get(profile, 0) + 1
        
        # Blockchain coverage
        blockchain_coverage = {}
        for agent in active_agents:
            for blockchain in agent.supported_blockchains:
                blockchain_coverage[blockchain] = blockchain_coverage.get(blockchain, 0) + 1
        
        # DEX coverage
        dex_coverage = {}
        for agent in active_agents:
            for dex in agent.supported_dexs:
                dex_coverage[dex] = dex_coverage.get(dex, 0) + 1
        
        return {
            'total_agents': len(self.agents),
            'active_agents': len(active_agents),
            'total_volume': total_volume,
            'total_positions': total_positions,
            'average_performance_score': avg_performance,
            'risk_profile_distribution': risk_distribution,
            'blockchain_coverage': blockchain_coverage,
            'dex_coverage': dex_coverage,
            'last_updated': datetime.now().isoformat()
        }
    
    def suspend_agent(self, agent_id: str, reason: str = "") -> bool:
        """Suspend an agent
        
        Args:
            agent_id: Agent to suspend
            reason: Reason for suspension
            
        Returns:
            True if suspended successfully
        """
        if agent_id not in self.agents:
            return False
        
        self.agents[agent_id].status = AgentStatus.SUSPENDED
        if reason:
            self.agents[agent_id].metadata['suspension_reason'] = reason
            self.agents[agent_id].metadata['suspension_time'] = datetime.now().isoformat()
        
        self._save_registry()
        return True
    
    def reactivate_agent(self, agent_id: str) -> bool:
        """Reactivate a suspended agent
        
        Args:
            agent_id: Agent to reactivate
            
        Returns:
            True if reactivated successfully
        """
        if agent_id not in self.agents:
            return False
        
        if self.agents[agent_id].status == AgentStatus.SUSPENDED:
            self.agents[agent_id].status = AgentStatus.ACTIVE
            self.agents[agent_id].metadata.pop('suspension_reason', None)
            self.agents[agent_id].metadata.pop('suspension_time', None)
            self._save_registry()
            return True
        
        return False