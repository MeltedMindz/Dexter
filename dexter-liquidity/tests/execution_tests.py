import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from web3 import Web3
from execution.manager import (
    ExecutionManager,
    ExecutionType,
    ExecutionConfig
)

@pytest.fixture
def mock_web3():
    web3 = Mock(spec=Web3)
    web3.eth.get_block.return_value = {'timestamp': int(datetime.now().timestamp())}
    return web3

@pytest.fixture
def mock_agents():
    return [Mock(), Mock(), Mock()]

@pytest.fixture
def test_config():
    return {
        "price_updates": ExecutionConfig(
            type=ExecutionType.EVENT_BASED,
            event_names=["Swap"]
        ),
        "rebalance": ExecutionConfig(
            type=ExecutionType.PERIODIC,
            interval=300  # 5 minutes
        )
    }

@pytest.fixture
async def execution_manager(mock_web3, mock_agents, test_config):
    manager = ExecutionManager(mock_web3, mock_agents, test_config)
    yield manager
    await manager.stop()

@pytest.mark.asyncio
async def test_event_based_execution(execution_manager, mock_web3):
    # Mock event data
    event_data = {
        'args': {
            'pair': '0x123...',
            'amount0In': 1000,
            'amount1In': 0,
            'amount0Out': 0,
            'amount1Out': 990
        }
    }
    
    # Start manager in background
    task = asyncio.create_task(execution_manager.start())
    
    # Simulate event
    await execution_manager._handle_event(event_data, "price_updates")
    
    # Verify agents were called
    for agent in execution_manager.agents:
        agent.execute_strategy.assert_called_once()
    
    # Cleanup
    await execution_manager.stop()
    await task

@pytest.mark.asyncio
async def test_periodic_execution(execution_manager):
    # Mock time
    start_time = datetime.now()
    
    # Start execution
    task = asyncio.create_task(execution_manager.start())
    
    # Wait for one execution cycle
    await asyncio.sleep(0.1)
    
    # Verify periodic execution
    assert len(execution_manager.last_execution) > 0
    execution_time = execution_manager.last_execution["rebalance"]
    assert execution_time >= start_time
    
    # Cleanup
    await execution_manager.stop()
    await task

@pytest.mark.asyncio
async def test_error_handling(execution_manager):
    # Make an agent raise an error
    execution_manager.agents[0].execute_strategy.side_effect = Exception("Test error")
    
    # Should continue executing despite error
    await execution_manager._execute_strategies("test")
    
    # Other agents should still have been called
    execution_manager.agents[1].execute_strategy.assert_called_once()
    execution_manager.agents[2].execute_strategy.assert_called_once()
