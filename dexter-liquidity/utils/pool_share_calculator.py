from decimal import Decimal
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class UserShare:
    user_id: str
    amount: Decimal
    share_percentage: Decimal
    current_value: Decimal
    earned_fees: Decimal
    impermanent_loss: Decimal

class PoolShareCalculator:
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
    def calculate_user_shares(
        self,
        pool_address: str,
        user_stakes: Dict[str, Decimal]
    ) -> List[UserShare]:
        """Calculate each user's share and performance in the pool"""
        
        # Get pool data
        pool = self.pool_manager.get_pool(pool_address)
        total_pool_value = pool.tvl
        total_fees = pool.accumulated_fees
        
        shares = []
        for user_id, staked_amount in user_stakes.items():
            # Calculate share percentage
            share_percentage = staked_amount / total_pool_value
            
            # Calculate current value including earned fees
            earned_fees = total_fees * share_percentage
            current_value = (total_pool_value * share_percentage) + earned_fees
            
            # Calculate impermanent loss for this position
            il = self._calculate_impermanent_loss(
                pool,
                staked_amount,
                share_percentage
            )
            
            shares.append(UserShare(
                user_id=user_id,
                amount=staked_amount,
                share_percentage=share_percentage,
                current_value=current_value,
                earned_fees=earned_fees,
                impermanent_loss=il
            ))
            
        return shares