// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "./ITransformer.sol";
import "../libraries/TWAPOracle.sol";

/// @title RangeTransformer
/// @notice Transformer for changing position ranges
/// @dev Implements ITransformer pattern for position range modifications
contract RangeTransformer is ITransformer, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using TWAPOracle for IUniswapV3Pool;
    
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    
    struct RangeParams {
        uint256 tokenId;
        int24 newTickLower;
        int24 newTickUpper;
        uint128 liquidityToRemove; // 0 = remove all
        uint256 amount0Min;
        uint256 amount1Min;
        uint256 deadline;
        bool useTWAPProtection;
        uint32 twapSeconds;
        int24 maxTickDifference;
        address recipient;
        bool burnOldPosition; // Whether to burn the old position
    }
    
    error InvalidTokenId();
    error InvalidTickRange();
    error DeadlineExpired();
    error TWAPValidationFailed();
    error InsufficientLiquidity();
    
    event RangeChanged(
        uint256 indexed oldTokenId,
        uint256 indexed newTokenId,
        int24 oldTickLower,
        int24 oldTickUpper,
        int24 newTickLower,
        int24 newTickUpper,
        uint256 liquidityMoved,
        address indexed recipient
    );
    
    constructor(INonfungiblePositionManager _nonfungiblePositionManager) {
        nonfungiblePositionManager = _nonfungiblePositionManager;
    }
    
    /// @notice Transform a position by changing its range
    /// @param transformParams Encoded RangeParams
    /// @return success Whether the transformation succeeded
    /// @return newTokenId New token ID for the repositioned liquidity
    /// @return amount0 Token0 amount in the new position
    /// @return amount1 Token1 amount in the new position
    function transform(bytes calldata transformParams)
        external
        override
        nonReentrant
        returns (bool success, uint256 newTokenId, uint256 amount0, uint256 amount1)
    {
        RangeParams memory params = abi.decode(transformParams, (RangeParams));
        
        // Validate deadline
        if (params.deadline > 0 && block.timestamp > params.deadline) {
            revert DeadlineExpired();
        }
        
        // Validate tick range
        if (params.newTickLower >= params.newTickUpper) {
            revert InvalidTickRange();
        }
        
        // Get position info
        (, , address token0, address token1, uint24 fee, int24 oldTickLower, int24 oldTickUpper, uint128 liquidity, , , ,) = 
            nonfungiblePositionManager.positions(params.tokenId);
            
        if (liquidity == 0) {
            revert InvalidTokenId();
        }
        
        // Determine liquidity to remove
        uint128 liquidityToRemove = params.liquidityToRemove;
        if (liquidityToRemove == 0 || liquidityToRemove > liquidity) {
            liquidityToRemove = liquidity;
        }
        
        // TWAP validation if enabled
        if (params.useTWAPProtection) {
            _validateTWAP(token0, token1, fee, params.twapSeconds, params.maxTickDifference);
        }
        
        // Remove liquidity from old position
        uint256 collected0;
        uint256 collected1;
        
        if (liquidityToRemove > 0) {
            // Decrease liquidity
            INonfungiblePositionManager.DecreaseLiquidityParams memory decreaseParams = 
                INonfungiblePositionManager.DecreaseLiquidityParams({
                    tokenId: params.tokenId,
                    liquidity: liquidityToRemove,
                    amount0Min: 0, // Will validate later when creating new position
                    amount1Min: 0,
                    deadline: params.deadline
                });
                
            nonfungiblePositionManager.decreaseLiquidity(decreaseParams);
        }
        
        // Collect all available tokens (including fees)
        INonfungiblePositionManager.CollectParams memory collectParams = 
            INonfungiblePositionManager.CollectParams({
                tokenId: params.tokenId,
                recipient: address(this),
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            });
            
        (collected0, collected1) = nonfungiblePositionManager.collect(collectParams);
        
        // Create new position with new range
        if (collected0 > 0 || collected1 > 0) {
            // Approve tokens for new position
            if (collected0 > 0) {
                IERC20(token0).safeApprove(address(nonfungiblePositionManager), collected0);
            }
            if (collected1 > 0) {
                IERC20(token1).safeApprove(address(nonfungiblePositionManager), collected1);
            }
            
            // Mint new position
            INonfungiblePositionManager.MintParams memory mintParams = 
                INonfungiblePositionManager.MintParams({
                    token0: token0,
                    token1: token1,
                    fee: fee,
                    tickLower: params.newTickLower,
                    tickUpper: params.newTickUpper,
                    amount0Desired: collected0,
                    amount1Desired: collected1,
                    amount0Min: params.amount0Min,
                    amount1Min: params.amount1Min,
                    recipient: params.recipient,
                    deadline: params.deadline
                });
                
            (newTokenId, , amount0, amount1) = nonfungiblePositionManager.mint(mintParams);
            
            // Reset approvals
            if (collected0 > 0) {
                IERC20(token0).safeApprove(address(nonfungiblePositionManager), 0);
            }
            if (collected1 > 0) {
                IERC20(token1).safeApprove(address(nonfungiblePositionManager), 0);
            }
            
            // Send any leftover tokens to recipient
            if (collected0 > amount0) {
                IERC20(token0).safeTransfer(params.recipient, collected0 - amount0);
            }
            if (collected1 > amount1) {
                IERC20(token1).safeTransfer(params.recipient, collected1 - amount1);
            }
        }
        
        // Burn old position if requested and fully emptied
        if (params.burnOldPosition && liquidityToRemove == liquidity) {
            // Check that position is indeed empty
            (, , , , , , , uint128 remainingLiquidity, , , ,) = 
                nonfungiblePositionManager.positions(params.tokenId);
            
            if (remainingLiquidity == 0) {
                nonfungiblePositionManager.burn(params.tokenId);
            }
        }
        
        emit RangeChanged(
            params.tokenId,
            newTokenId,
            oldTickLower,
            oldTickUpper,
            params.newTickLower,
            params.newTickUpper,
            liquidityToRemove,
            params.recipient
        );
        
        success = true;
    }
    
    /// @notice Validate transformation parameters
    /// @param transformParams Encoded RangeParams
    /// @return isValid Whether parameters are valid
    /// @return errorMessage Error message if invalid
    function validateTransform(bytes calldata transformParams)
        external
        view
        override
        returns (bool isValid, string memory errorMessage)
    {
        try this._decodeAndValidate(transformParams) {
            return (true, "");
        } catch Error(string memory reason) {
            return (false, reason);
        } catch {
            return (false, "Invalid parameters");
        }
    }
    
    /// @notice Internal function for validation (enables try/catch)
    function _decodeAndValidate(bytes calldata transformParams) external view {
        RangeParams memory params = abi.decode(transformParams, (RangeParams));
        
        if (params.tokenId == 0) {
            revert("Invalid token ID");
        }
        
        if (params.newTickLower >= params.newTickUpper) {
            revert("Invalid tick range");
        }
        
        if (params.deadline > 0 && params.deadline <= block.timestamp) {
            revert("Deadline expired");
        }
        
        // Validate that position exists and has liquidity
        try nonfungiblePositionManager.positions(params.tokenId) returns (
            uint96, address, address, address, uint24, int24, int24, uint128 liquidity, uint256, uint256, uint128, uint128
        ) {
            if (liquidity == 0) {
                revert("Position has no liquidity");
            }
        } catch {
            revert("Position does not exist");
        }
    }
    
    /// @notice Get transformation type identifier
    /// @return transformType String identifier for this transformer
    function getTransformType() external pure override returns (string memory transformType) {
        return "RANGE_CHANGE";
    }
    
    /// @notice Check if transformation requires approval
    /// @param transformParams Encoded transformation parameters
    /// @return requiresApproval Always true for range changes
    function requiresApproval(bytes calldata transformParams)
        external
        pure
        override
        returns (bool requiresApproval)
    {
        // Range changes always require approval since they modify positions
        return true;
    }
    
    /// @notice Estimate gas cost for transformation
    /// @param transformParams Encoded transformation parameters
    /// @return estimatedGas Estimated gas cost
    function estimateGas(bytes calldata transformParams)
        external
        view
        override
        returns (uint256 estimatedGas)
    {
        RangeParams memory params = abi.decode(transformParams, (RangeParams));
        
        // Base gas for range change (decrease + collect + mint)
        estimatedGas = 400000;
        
        // Additional gas for TWAP validation
        if (params.useTWAPProtection) {
            estimatedGas += 30000;
        }
        
        // Additional gas for burning old position
        if (params.burnOldPosition) {
            estimatedGas += 50000;
        }
        
        // Check if position has significant liquidity (affects gas)
        try nonfungiblePositionManager.positions(params.tokenId) returns (
            uint96, address, address, address, uint24, int24, int24, uint128 liquidity, uint256, uint256, uint128, uint128
        ) {
            if (liquidity > 1e18) {
                estimatedGas += 50000; // More gas for larger positions
            }
        } catch {
            // If we can't read the position, use base estimate
        }
    }
    
    /// @notice Validate TWAP for MEV protection
    function _validateTWAP(
        address token0,
        address token1,
        uint24 fee,
        uint32 twapSeconds,
        int24 maxTickDifference
    ) internal view {
        address poolAddress = IUniswapV3Factory(nonfungiblePositionManager.factory())
            .getPool(token0, token1, fee);
            
        require(poolAddress != address(0), "Pool not found");
        
        IUniswapV3Pool pool = IUniswapV3Pool(poolAddress);
        
        (bool success, ) = pool.verifyTWAP(
            twapSeconds,
            maxTickDifference,
            false // No AI override for transformers
        );
        
        if (!success) {
            revert TWAPValidationFailed();
        }
    }
    
    /// @notice Calculate optimal tick range based on current price and volatility
    /// @param token0 Token0 address
    /// @param token1 Token1 address
    /// @param fee Pool fee tier
    /// @param targetWidth Target width in ticks
    /// @return tickLower Optimal lower tick
    /// @return tickUpper Optimal upper tick
    function calculateOptimalRange(
        address token0,
        address token1,
        uint24 fee,
        int24 targetWidth
    ) external view returns (int24 tickLower, int24 tickUpper) {
        address poolAddress = IUniswapV3Factory(nonfungiblePositionManager.factory())
            .getPool(token0, token1, fee);
            
        require(poolAddress != address(0), "Pool not found");
        
        IUniswapV3Pool pool = IUniswapV3Pool(poolAddress);
        (, int24 currentTick, , , , , ) = pool.slot0();
        
        // Center range around current tick
        int24 halfWidth = targetWidth / 2;
        tickLower = currentTick - halfWidth;
        tickUpper = currentTick + halfWidth;
        
        // Align to tick spacing
        int24 tickSpacing = _getTickSpacing(fee);
        tickLower = (tickLower / tickSpacing) * tickSpacing;
        tickUpper = (tickUpper / tickSpacing) * tickSpacing;
        
        // Ensure upper > lower
        if (tickUpper <= tickLower) {
            tickUpper = tickLower + tickSpacing;
        }
    }
    
    /// @notice Get tick spacing for fee tier
    function _getTickSpacing(uint24 fee) internal pure returns (int24) {
        if (fee == 100) return 1;
        if (fee == 500) return 10;
        if (fee == 3000) return 60;
        if (fee == 10000) return 200;
        return 60; // Default to 0.3% tier spacing
    }
    
    /// @notice Emergency function to withdraw stuck tokens
    function emergencyWithdraw(address token, uint256 amount, address to) external {
        // Note: Should add access control in production
        IERC20(token).safeTransfer(to, amount);
    }
}