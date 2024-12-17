import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from web3 import Web3
from web3.contract import Contract
import json
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import time

from utils.error_handler import ErrorHandler, DexterError, NetworkError
from utils.memory_monitor import MemoryMonitor
from utils.parallel_processor import ParallelDataProcessor
from utils.statistics import StatisticsCollector
from utils.pool_share_calculator import PoolShareCalculator
from utils.performance_tracker import PerformanceTracker
from agents.types import RiskProfile
from .execution_config import ExecutionConfig, ExecutionType

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()
memory_monitor = MemoryMonitor()

class ExecutionType(Enum):
    EVENT_BASED = "event"
    PERIODIC = "periodic"
    CONTRACT_AUTOMATION = "contract"

@dataclass
class ExecutionConfig:
    type: ExecutionType
    interval: Optional[int] = None  # seconds for periodic
    event_names: Optional[List[str]] = None
    contract_address: Optional[str] = None
    retry_attempts: int = 3
    timeout: int = 30

class ExecutionManager:
    def __init__(
        self,
        web3: Web3,
        agents: List['DexterAgent'],
        config: Dict[str, ExecutionConfig],
        parallel_chunk_size: int = 1000
    ):
        self.web3 = web3
        self.agents = agents
        self.config = config
        self.running_tasks: Set[asyncio.Task] = set()
        self.last_execution: Dict[str, datetime] = {}
        self.event_filters = {}
        self.parallel_processor = ParallelDataProcessor(chunk_size=parallel_chunk_size)
        self.health_checks: Dict[str, bool] = {}
        self.health_metrics: Dict[str, Any] = {}  # For storing execution health metrics
        
        # Initialize recovery strategies
        self._init_recovery_strategies()
        self.stats_collector = StatisticsCollector()

        logger.info("execution_manager.py: Initialized execution manager")

    def _init_recovery_strategies(self):
        """Initialize error recovery strategies"""
        error_handler.register_recovery_strategy(
            NetworkError,
            self._handle_network_error
        )
        error_handler.register_recovery_strategy(
            DexterError,
            self._handle_system_error
        )

    @error_handler.with_retries(retries=3)
    async def start(self):
        """Start all execution methods with error handling"""
        logger.info("execution_manager.py: Starting execution manager")

        try:
            # Start health monitoring
            monitor_task = asyncio.create_task(self._monitor_system_health())
            self.running_tasks.add(monitor_task)

            # Start different execution types
            for name, cfg in self.config.items():
                task = await self._create_execution_task(name, cfg)
                if task:
                    self.running_tasks.add(task)
                    task.add_done_callback(self.running_tasks.discard)

            logger.info("execution_manager.py: All execution methods started")

            # Keep running until stopped
            await asyncio.gather(*self.running_tasks)

        except Exception as e:
            logger.error(f"execution_manager.py: Error starting execution manager: {str(e)}")
            await self.stop()
            raise

    @memory_monitor.monitor
    async def _create_execution_task(
        self,
        name: str,
        config: ExecutionConfig
    ) -> Optional[asyncio.Task]:
        """Create execution task based on type"""
        try:
            if config.type == ExecutionType.EVENT_BASED:
                return asyncio.create_task(self._run_event_based(name, config))
            elif config.type == ExecutionType.PERIODIC:
                return asyncio.create_task(self._run_periodic(name, config))
            elif config.type == ExecutionType.CONTRACT_AUTOMATION:
                return asyncio.create_task(self._run_contract_automation(name, config))
            return None
        except Exception as e:
            logger.error(f"execution_manager.py: Error creating task {name}: {str(e)}")
            return None

    async def stop(self):
        """Stop all execution methods"""
        logger.info("execution_manager.py: Stopping execution manager")

        try:
            for task in self.running_tasks:
                task.cancel()

            await asyncio.gather(*self.running_tasks, return_exceptions=True)
            self.running_tasks.clear()

            # Clean up resources
            await self._cleanup_resources()

        except Exception as e:
            logger.error(f"execution_manager.py: Error during shutdown: {str(e)}")

    @error_handler.with_retries(retries=3)
    async def _run_event_based(self, name: str, config: ExecutionConfig):
        """Run event-based execution with error handling"""
        logger.info(f"execution_manager.py: Starting event-based execution for {name}")

        try:
            # Set up event filters
            await self._setup_event_filters(name, config)

            while True:
                await self._process_events(name)
                await asyncio.sleep(1)  # Poll interval

        except asyncio.CancelledError:
            logger.info(f"execution_manager.py: Event-based execution stopped for {name}")
        except Exception as e:
            logger.error(f"execution_manager.py: Error in event-based execution: {str(e)}")
            raise

    @error_handler.with_retries(retries=3)
    async def _run_periodic(self, name: str, config: ExecutionConfig):
        """Run periodic execution with error handling"""
        logger.info(f"execution_manager.py: Starting periodic execution for {name}")

        try:
            while True:
                start_time = time.time()

                await self._execute_strategies(trigger=name)

                # Calculate next execution time
                elapsed = time.time() - start_time
                sleep_time = max(0, config.interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info(f"execution_manager.py: Periodic execution stopped for {name}")
        except Exception as e:
            logger.error(f"execution_manager.py: Error in periodic execution: {str(e)}")
            raise

    @error_handler.with_retries(retries=3)
    async def _run_contract_automation(self, name: str, config: ExecutionConfig):
        """Run contract-based automation with error handling"""
        logger.info(f"execution_manager.py: Starting contract automation for {name}")

        try:
            contract = await self._get_automation_contract(config.contract_address)

            while True:
                if await self._should_execute_contract(contract):
                    await self._execute_contract_strategy(contract, name)
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"execution_manager.py: Contract automation stopped for {name}")
        except Exception as e:
            logger.error(f"execution_manager.py: Error in contract automation: {str(e)}")
            raise

    @memory_monitor.monitor
    async def _execute_strategies(self, trigger: str, event: Optional[Dict] = None):
        """Execute strategies with memory monitoring and performance tracking"""
        logger.info(f"execution_manager.py: Executing strategies for {trigger}")
        start_time = time.time()

        try:
            # Calculate pre-execution metrics for performance comparison
            pre_execution_tvl = await self._calculate_total_tvl()
            
            # Choose execution method based on number of agents
            if len(self.agents) > self.parallel_processor.chunk_size:
                results = await self._execute_strategies_parallel(trigger, event)
            else:
                results = await self._execute_strategies_sequential(trigger, event)

            # Record execution time and status
            self.last_execution[trigger] = datetime.now()
            execution_time = time.time() - start_time

            # Calculate post-execution performance metrics
            post_execution_tvl = await self._calculate_total_tvl()
            daily_fees = await self._calculate_daily_fees()
            impermanent_loss = await self._calculate_total_il()

            # Update performance metrics for each risk profile
            for risk_profile in RiskProfile:
                await self.performance_tracker.update_metrics(
                    risk_profile=risk_profile,
                    current_tvl=post_execution_tvl,
                    daily_fees=daily_fees,
                    il=impermanent_loss,
                    timestamp=datetime.now()
                )

            # Update user shares and rewards
            await self.pool_calculator.update_user_shares(
                pre_execution_tvl=pre_execution_tvl,
                post_execution_tvl=post_execution_tvl,
                fees_earned=daily_fees,
                impermanent_loss=impermanent_loss
            )

            # Record statistics
            success_count = sum(1 for r in results if r)
            await self.stats_collector.record_execution(
                trigger=trigger,
                execution_time=execution_time,
                success_count=success_count,
                total_count=len(results)
            )

            logger.info(
                f"execution_manager.py: Strategy execution completed - "
                f"Success: {success_count}/{len(results)} in {execution_time:.2f}s, "
                f"TVL Change: ${post_execution_tvl - pre_execution_tvl:,.2f}, "
                f"Fees Earned: ${daily_fees:,.2f}"
            )

            # Trigger health check after execution
            await self._check_execution_health(success_count, len(results))

        except Exception as e:
            logger.error(f"execution_manager.py: Error executing strategies: {str(e)}")
            # Record failed execution
            await self.stats_collector.record_execution(
                trigger=trigger,
                execution_time=time.time() - start_time,
                success_count=0,
                total_count=len(self.agents)
            )
            raise

    async def _execute_strategies_parallel(self, trigger: str, event: Optional[Dict] = None) -> List[bool]:
        """Execute strategies in parallel with chunking and performance tracking"""
        logger.info("execution_manager.py: Starting parallel strategy execution")
        try:
            # Create chunks of agents for parallel processing
            chunks = [
                self.agents[i:i + self.parallel_processor.chunk_size] 
                for i in range(0, len(self.agents), self.parallel_processor.chunk_size)
            ]

            # Process chunks in parallel with performance tracking
            chunk_results = await asyncio.gather(*[
                self._execute_strategy_chunk(chunk, trigger, event)
                for chunk in chunks
            ])

            # Flatten results
            results = [result for chunk in chunk_results for result in chunk]

            # Update pool shares after parallel execution
            await self.pool_calculator.reconcile_parallel_execution_results(results)

            logger.info(f"execution_manager.py: Completed parallel execution of {len(results)} strategies")
            return results

        except Exception as e:
            logger.error(f"execution_manager.py: Error in parallel execution: {str(e)}")
            return [False] * len(self.agents)

    async def _execute_strategy_chunk(
        self,
        agents: List['DexterAgent'],
        trigger: str,
        event: Optional[Dict] = None
    ) -> List[bool]:
        """Execute strategies for a chunk of agents with error handling and performance tracking"""
        results = []
        for agent in agents:
            try:
                start_time = time.time()
                
                # Get pre-execution position values for this agent
                pre_execution_positions = await self._get_agent_positions(agent)
                
                # Execute strategy
                result = await agent.execute_strategy(trigger, event)

                # Calculate position changes and update user shares
                if result:
                    post_execution_positions = await self._get_agent_positions(agent)
                    await self._update_position_changes(
                        agent,
                        pre_execution_positions,
                        post_execution_positions
                    )

                # Record individual agent execution
                await self.stats_collector.record_agent_execution(
                    agent_type=agent.risk_profile.value,
                    execution_time=time.time() - start_time,
                    success=bool(result)
                )

                results.append(result)

            except Exception as e:
                logger.error(
                    f"execution_manager.py: Error executing strategy for {agent.risk_profile.value}: {str(e)}"
                )
                results.append(False)

        return results

    async def _execute_strategies_sequential(
        self,
        trigger: str,
        event: Optional[Dict] = None
    ) -> List[bool]:
        """Execute strategies sequentially with monitoring and performance tracking"""
        logger.info("execution_manager.py: Starting sequential strategy execution")
        results = []

        for agent in self.agents:
            try:
                start_time = time.time()
                
                # Get pre-execution position values
                pre_execution_positions = await self._get_agent_positions(agent)
                
                # Execute strategy
                result = await agent.execute_strategy(trigger, event)

                # Calculate position changes and update user shares
                if result:
                    post_execution_positions = await self._get_agent_positions(agent)
                    await self._update_position_changes(
                        agent,
                        pre_execution_positions,
                        post_execution_positions
                    )

                # Record execution metrics
                await self.stats_collector.record_agent_execution(
                    agent_type=agent.risk_profile.value,
                    execution_time=time.time() - start_time,
                    success=bool(result)
                )

                results.append(result)

            except Exception as e:
                logger.error(
                    f"execution_manager.py: Error in sequential execution for {agent.risk_profile.value}: {str(e)}"
                )
                results.append(False)

        return results

    async def _execute_strategies_sequential(
        self,
        trigger: str,
        event: Optional[Dict] = None
    ) -> List[bool]:
        """Execute strategies sequentially with monitoring"""
        logger.info("execution_manager.py: Starting sequential strategy execution")
        results = []

        for agent in self.agents:
            try:
                start_time = time.time()
                result = await agent.execute_strategy(trigger, event)

                # Record execution metrics
                await self.stats_collector.record_agent_execution(
                    agent_type=agent.risk_profile.value,
                    execution_time=time.time() - start_time,
                    success=bool(result)
                )

                results.append(result)

            except Exception as e:
                logger.error(
                    f"execution_manager.py: Error in sequential execution for {agent.risk_profile.value}: {str(e)}"
                )
                results.append(False)

        return results

    async def _check_execution_health(self, success_count: int, total_count: int):
        """Monitor execution health and trigger alerts if needed"""
        success_rate = success_count / total_count if total_count > 0 else 0

        if success_rate < 0.5:  # Less than 50% success
            logger.warning(
                f"execution_manager.py: Low success rate detected: {success_rate:.2%}"
            )
            # Trigger health check
            await self._monitor_system_health()

        # Update health metrics
        self.health_metrics.update({
            'last_success_rate': success_rate,
            'last_execution_time': datetime.now().isoformat()
        })

    async def _monitor_system_health(self):
        """Monitor system health continuously"""
        while True:
            try:
                # Check Web3 connection
                self.health_checks['web3'] = self.web3.is_connected()

                # Check agent health
                for agent in self.agents:
                    self.health_checks[f'agent_{agent.risk_profile.value}'] = (
                        await agent.check_health()
                    )

                # Log health status
                logger.info(f"System health status: {self.health_checks}")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                await asyncio.sleep(5)

    async def _handle_network_error(self, error: NetworkError):
        """Handle network-related errors"""
        logger.error(f"Network error occurred: {str(error)}")
        # Implement recovery logic

    async def _handle_system_error(self, error: DexterError):
        """Handle system-level errors"""
        logger.error(f"System error occurred: {str(error)}")
        # Implement recovery logic

    async def _cleanup_resources(self):
        """Clean up system resources"""
        try:
            # Clean up event filters
            self.event_filters.clear()

            # Clean up memory monitor
            memory_monitor.clear_stats()

            logger.info("Resources cleaned up successfully")

        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")

    def _create_event_filter(self, event_name: str, config: ExecutionConfig) -> any:
        """Create Web3 event filter"""
        try:
            contract = self._get_contract(config.contract_address)
            return getattr(contract.events, event_name).create_filter(fromBlock='latest')
        except Exception as e:
            logger.error(f"execution_manager.py: Error creating event filter: {str(e)}")
            raise

    async def _get_automation_contract(self, address: str) -> Contract:
        """Get automation contract instance with error handling"""
        try:
            # Add your automation contract ABI here
            abi = []
            contract = self.web3.eth.contract(address=address, abi=abi)
            return contract
        except Exception as e:
            logger.error(f"execution_manager.py: Error getting automation contract: {str(e)}")
            raise

    @error_handler.with_retries(retries=2)
    async def _should_execute_contract(self, contract: Contract) -> bool:
        """Check if contract conditions are met"""
        try:
            # Implement your contract condition checks
            conditions_met = await self._check_contract_conditions(contract)
            return conditions_met
        except Exception as e:
            logger.error(f"execution_manager.py: Error checking contract conditions: {str(e)}")
            return False

    async def _check_contract_conditions(self, contract: Contract) -> bool:
        """Check specific contract conditions"""
        try:
            # Implement your specific condition checks here
            return True
        except Exception as e:
            logger.error(f"execution_manager.py: Error in contract conditions: {str(e)}")
            return False

    async def _execute_contract_strategy(self, contract: Contract, trigger: str):
        """Execute contract-based strategy"""
        try:
            logger.info(f"execution_manager.py: Executing contract strategy for {trigger}")
            # Implement your contract strategy execution
            await self._execute_strategies(trigger=trigger)
        except Exception as e:
            logger.error(f"execution_manager.py: Error executing contract strategy: {str(e)}")
            raise

    @memory_monitor.monitor
    async def _update_market_data(self, event: Dict):
        """Update market data based on event"""
        try:
            # Implement market data update logic
            logger.info("execution_manager.py: Updating market data")
            pass
        except Exception as e:
            logger.error(f"execution_manager.py: Error updating market data: {str(e)}")
            raise

    async def _should_execute(self, event: Dict, trigger: str) -> bool:
        """Determine if strategies should be executed"""
        try:
            # Add your execution condition logic here
            return True
        except Exception as e:
            logger.error(f"execution_manager.py: Error checking execution conditions: {str(e)}")
            return False

    async def _process_events(self, name: str):
        """Process events for event-based execution"""
        # Implement event processing and trigger strategy execution if needed
        pass

    async def _setup_event_filters(self, name: str, config: ExecutionConfig):
        """Set up filters for event-based execution"""
        # Implement event filter setup logic
        pass

    async def _calculate_total_tvl(self) -> float:
        return sum(await agent._calculate_tvl() for agent in self.agents)

    async def _calculate_daily_fees(self) -> float:
        """
        Calculate total fees earned in the last 24 hours.
        Aggregates fee calculations from all active agents.
        """
        return sum(await agent._calculate_fees_24h() for agent in self.agents)

    async def _calculate_total_il(self) -> float:
        """
        Calculate total impermanent loss across all positions.
        Combines IL calculations from all agents to get system-wide IL.
        """
        return sum(await agent._calculate_impermanent_loss() for agent in self.agents)

    async def _get_agent_positions(self, agent: 'DexterAgent') -> Dict[str, float]:
        """
        Get current position values for an agent.
        Returns a mapping of pair addresses to their current position values.
        """
        return {
            pair_address: await agent._get_position_value(pair_address)
            for pair_address in agent.active_pairs
        }

    async def _update_position_changes(
        self,
        agent: 'DexterAgent',
        pre_positions: Dict[str, float],
        post_positions: Dict[str, float]
    ):
        """
        Update user shares based on position changes.
        Tracks value changes in positions and updates user share calculations.
        
        Args:
            agent: The trading agent whose positions changed
            pre_positions: Position values before execution
            post_positions: Position values after execution
        """
        for pair_address, pre_value in pre_positions.items():
            post_value = post_positions.get(pair_address, 0)
            if post_value != pre_value:
                await self.pool_calculator.update_position_value(
                    agent.risk_profile,
                    pair_address,
                    pre_value,
                    post_value
                )

    def _get_contract(self, address: str) -> Contract:
        """Get a contract instance given an address"""
        # Replace with actual ABI and contract initialization
        abi = []
        contract = self.web3.eth.contract(address=address, abi=abi)
        return contract
