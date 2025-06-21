// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "./ITransformer.sol";
import "../libraries/TWAPOracle.sol";

/// @title CompoundTransformer
/// @notice Transformer for compounding position fees
/// @dev Implements ITransformer pattern for modular fee compounding
contract CompoundTransformer is ITransformer, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using TWAPOracle for IUniswapV3Pool;
    
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    
    struct CompoundParams {
        uint256 tokenId;
        uint256 amount0Min;
        uint256 amount1Min;
        uint256 deadline;
        bool useTWAPProtection;
        uint32 twapSeconds;
        int24 maxTickDifference;
        address recipient; // For reward collection
        uint256 rewardPercentage; // In basis points (100 = 1%)
    }
    
    error InvalidTokenId();
    error InsufficientFees();
    error DeadlineExpired();
    error TWAPValidationFailed();
    error InvalidRewardPercentage();
    
    event CompoundExecuted(
        uint256 indexed tokenId,
        uint256 amount0Compounded,
        uint256 amount1Compounded,
        uint256 rewardAmount0,
        uint256 rewardAmount1,
        address indexed recipient
    );
    
    constructor(INonfungiblePositionManager _nonfungiblePositionManager) {
        nonfungiblePositionManager = _nonfungiblePositionManager;
    }
    
    /// @notice Transform a position by compounding its fees
    /// @param transformParams Encoded CompoundParams
    /// @return success Whether the transformation succeeded
    /// @return newTokenId Always 0 (position modified in place)
    /// @return amount0 Token0 amount compounded
    /// @return amount1 Token1 amount compounded
    function transform(bytes calldata transformParams)
        external
        override
        nonReentrant
        returns (bool success, uint256 newTokenId, uint256 amount0, uint256 amount1)
    {
        CompoundParams memory params = abi.decode(transformParams, (CompoundParams));
        
        // Validate deadline
        if (params.deadline > 0 && block.timestamp > params.deadline) {
            revert DeadlineExpired();
        }
        
        // Validate reward percentage
        if (params.rewardPercentage > 1000) { // Max 10%
            revert InvalidRewardPercentage();
        }
        
        // Get position info
        (, , address token0, address token1, uint24 fee, , , uint128 liquidity, , , uint128 tokensOwed0, uint128 tokensOwed1) = 
            nonfungiblePositionManager.positions(params.tokenId);
            
        if (liquidity == 0) {
            revert InvalidTokenId();
        }
        
        if (tokensOwed0 == 0 && tokensOwed1 == 0) {
            revert InsufficientFees();
        }
        
        // TWAP validation if enabled
        if (params.useTWAPProtection) {
            _validateTWAP(token0, token1, fee, params.twapSeconds, params.maxTickDifference);
        }
        
        // Collect fees
        INonfungiblePositionManager.CollectParams memory collectParams = 
            INonfungiblePositionManager.CollectParams({
                tokenId: params.tokenId,
                recipient: address(this),
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            });
            
        (uint256 collected0, uint256 collected1) = nonfungiblePositionManager.collect(collectParams);
        
        // Calculate rewards
        uint256 reward0 = (collected0 * params.rewardPercentage) / 10000;
        uint256 reward1 = (collected1 * params.rewardPercentage) / 10000;
        
        // Amounts to compound
        amount0 = collected0 - reward0;
        amount1 = collected1 - reward1;
        
        // Compound the fees back into position
        if (amount0 > 0 || amount1 > 0) {
            // Approve tokens
            if (amount0 > 0) {
                IERC20(token0).safeApprove(address(nonfungiblePositionManager), amount0);
            }
            if (amount1 > 0) {
                IERC20(token1).safeApprove(address(nonfungiblePositionManager), amount1);
            }
            
            // Increase liquidity
            INonfungiblePositionManager.IncreaseLiquidityParams memory increaseParams = 
                INonfungiblePositionManager.IncreaseLiquidityParams({
                    tokenId: params.tokenId,
                    amount0Desired: amount0,
                    amount1Desired: amount1,
                    amount0Min: params.amount0Min,
                    amount1Min: params.amount1Min,
                    deadline: params.deadline
                });
                
            (, uint256 actualAmount0, uint256 actualAmount1) = nonfungiblePositionManager.increaseLiquidity(increaseParams);
            
            // Reset approvals
            if (amount0 > 0) {
                IERC20(token0).safeApprove(address(nonfungiblePositionManager), 0);
            }
            if (amount1 > 0) {
                IERC20(token1).safeApprove(address(nonfungiblePositionManager), 0);
            }
            
            // Send any leftover tokens to recipient
            if (amount0 > actualAmount0) {
                IERC20(token0).safeTransfer(params.recipient, amount0 - actualAmount0);
            }
            if (amount1 > actualAmount1) {
                IERC20(token1).safeTransfer(params.recipient, amount1 - actualAmount1);
            }
            
            amount0 = actualAmount0;
            amount1 = actualAmount1;
        }
        
        // Send rewards to recipient
        if (reward0 > 0) {
            IERC20(token0).safeTransfer(params.recipient, reward0);
        }
        if (reward1 > 0) {
            IERC20(token1).safeTransfer(params.recipient, reward1);
        }
        
        emit CompoundExecuted(
            params.tokenId,
            amount0,
            amount1,
            reward0,
            reward1,
            params.recipient
        );
        
        success = true;
        newTokenId = 0; // Position modified in place
    }
    
    /// @notice Validate transformation parameters
    /// @param transformParams Encoded CompoundParams
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
    function _decodeAndValidate(bytes calldata transformParams) external pure {
        CompoundParams memory params = abi.decode(transformParams, (CompoundParams));
        
        if (params.tokenId == 0) {
            revert("Invalid token ID");
        }
        
        if (params.rewardPercentage > 1000) {
            revert("Reward percentage too high");
        }
        
        if (params.deadline > 0 && params.deadline <= block.timestamp) {
            revert("Deadline expired");
        }
    }
    
    /// @notice Get transformation type identifier
    /// @return transformType String identifier for this transformer
    function getTransformType() external pure override returns (string memory transformType) {
        return "COMPOUND";
    }
    
    /// @notice Check if transformation requires approval
    /// @param transformParams Encoded transformation parameters
    /// @return requiresApproval Always true for compound operations
    function requiresApproval(bytes calldata transformParams)
        external
        pure
        override
        returns (bool requiresApproval)
    {
        // Compound operations always require approval since they modify positions
        return true;
    }
    
    /// @notice Estimate gas cost for transformation
    /// @param transformParams Encoded transformation parameters
    /// @return estimatedGas Estimated gas cost
    function estimateGas(bytes calldata transformParams)
        external
        pure
        override
        returns (uint256 estimatedGas)
    {
        // Base gas for compound operation
        estimatedGas = 180000;
        
        CompoundParams memory params = abi.decode(transformParams, (CompoundParams));
        
        // Additional gas for TWAP validation
        if (params.useTWAPProtection) {
            estimatedGas += 30000;
        }
        
        // Additional gas for reward distribution
        if (params.rewardPercentage > 0) {
            estimatedGas += 20000;
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
    
    /// @notice Emergency function to withdraw stuck tokens
    function emergencyWithdraw(address token, uint256 amount, address to) external {
        // Note: Should add access control in production
        IERC20(token).safeTransfer(to, amount);
    }
}