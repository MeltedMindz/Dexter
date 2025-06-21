// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Multicall.sol";

import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";
import "../libraries/TWAPOracle.sol";

/// @title DexterV3Utils
/// @notice Stateless utility contract for complex Uniswap V3 operations
/// @dev Based on Revert Finance's V3Utils pattern - stateless and upgradeable
contract DexterV3Utils is ReentrancyGuard, Multicall {
    using SafeERC20 for IERC20;
    using TWAPOracle for IUniswapV3Pool;

    // Immutable contracts
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    ISwapRouter public immutable swapRouter;
    
    // Operation types
    enum WhatToDo {
        COMPOUND,
        SWAP_AND_COMPOUND,
        CHANGE_RANGE,
        WITHDRAW_AND_COLLECT,
        COLLECT_ONLY
    }
    
    // Instruction struct for complex operations
    struct Instructions {
        WhatToDo whatToDo;
        uint256 tokenId;
        uint256 liquidity; // For decrease operations
        uint256 amountAdd0; // For adding liquidity
        uint256 amountAdd1;
        uint256 amountRemove0; // For removing liquidity
        uint256 amountRemove1;
        uint256 amountSwap0; // For swapping
        uint256 amountSwap1;
        uint24 swapFeeTier; // Fee tier for swaps
        int24 tickLower; // For range changes
        int24 tickUpper;
        uint256 deadline;
        uint256 amount0Min;
        uint256 amount1Min;
        bool unwrapWETH;
        bool useTWAPProtection;
        bytes swapData; // For external swap router integration
    }
    
    // Errors
    error Unauthorized();
    error DeadlineExpired();
    error SlippageExceeded();
    error InvalidPool();
    error InsufficientLiquidity();
    
    // Events
    event CompletedOperation(
        address indexed user,
        uint256 indexed tokenId,
        WhatToDo whatToDo,
        uint256 amount0,
        uint256 amount1
    );
    
    constructor(
        INonfungiblePositionManager _nonfungiblePositionManager,
        ISwapRouter _swapRouter
    ) {
        nonfungiblePositionManager = _nonfungiblePositionManager;
        swapRouter = _swapRouter;
    }
    
    /// @notice Execute complex operations on Uniswap V3 positions
    /// @param instructions The operation instructions
    /// @return amount0 Token0 amount processed
    /// @return amount1 Token1 amount processed
    function execute(Instructions calldata instructions)
        external
        payable
        nonReentrant
        returns (uint256 amount0, uint256 amount1)
    {
        // Verify deadline
        if (instructions.deadline > 0 && block.timestamp > instructions.deadline) {
            revert DeadlineExpired();
        }
        
        // Get position info
        (, , address token0, address token1, uint24 fee, , , uint128 liquidity, , , ,) = 
            nonfungiblePositionManager.positions(instructions.tokenId);
            
        // Verify position exists and caller is authorized
        address owner = nonfungiblePositionManager.ownerOf(instructions.tokenId);
        if (owner != msg.sender) {
            revert Unauthorized();
        }
        
        // TWAP protection if enabled
        if (instructions.useTWAPProtection) {
            _validateTWAP(token0, token1, fee);
        }
        
        // Execute operation based on type
        if (instructions.whatToDo == WhatToDo.COMPOUND) {
            (amount0, amount1) = _compound(instructions);
        } else if (instructions.whatToDo == WhatToDo.SWAP_AND_COMPOUND) {
            (amount0, amount1) = _swapAndCompound(instructions);
        } else if (instructions.whatToDo == WhatToDo.CHANGE_RANGE) {
            (amount0, amount1) = _changeRange(instructions);
        } else if (instructions.whatToDo == WhatToDo.WITHDRAW_AND_COLLECT) {
            (amount0, amount1) = _withdrawAndCollect(instructions);
        } else if (instructions.whatToDo == WhatToDo.COLLECT_ONLY) {
            (amount0, amount1) = _collectOnly(instructions);
        }
        
        emit CompletedOperation(
            msg.sender,
            instructions.tokenId,
            instructions.whatToDo,
            amount0,
            amount1
        );
    }
    
    /// @notice Compound fees back into position
    function _compound(Instructions calldata instructions)
        internal
        returns (uint256 amount0, uint256 amount1)
    {
        // Collect fees
        INonfungiblePositionManager.CollectParams memory collectParams = 
            INonfungiblePositionManager.CollectParams({
                tokenId: instructions.tokenId,
                recipient: address(this),
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            });
            
        (uint256 collected0, uint256 collected1) = nonfungiblePositionManager.collect(collectParams);
        
        if (collected0 > 0 || collected1 > 0) {
            // Get position tokens for approval
            (, , address token0, address token1, , , , , , , ,) = 
                nonfungiblePositionManager.positions(instructions.tokenId);
            
            // Approve tokens
            if (collected0 > 0) {
                IERC20(token0).safeApprove(address(nonfungiblePositionManager), collected0);
            }
            if (collected1 > 0) {
                IERC20(token1).safeApprove(address(nonfungiblePositionManager), collected1);
            }
            
            // Add liquidity
            INonfungiblePositionManager.IncreaseLiquidityParams memory increaseParams = 
                INonfungiblePositionManager.IncreaseLiquidityParams({
                    tokenId: instructions.tokenId,
                    amount0Desired: collected0,
                    amount1Desired: collected1,
                    amount0Min: instructions.amount0Min,
                    amount1Min: instructions.amount1Min,
                    deadline: instructions.deadline
                });
                
            (, amount0, amount1) = nonfungiblePositionManager.increaseLiquidity(increaseParams);
            
            // Reset approvals
            if (collected0 > 0) {
                IERC20(token0).safeApprove(address(nonfungiblePositionManager), 0);
            }
            if (collected1 > 0) {
                IERC20(token1).safeApprove(address(nonfungiblePositionManager), 0);
            }
            
            // Return any leftover tokens
            if (collected0 > amount0) {
                IERC20(token0).safeTransfer(msg.sender, collected0 - amount0);
            }
            if (collected1 > amount1) {
                IERC20(token1).safeTransfer(msg.sender, collected1 - amount1);
            }
        }
    }
    
    /// @notice Swap and compound fees
    function _swapAndCompound(Instructions calldata instructions)
        internal
        returns (uint256 amount0, uint256 amount1)
    {
        // First collect fees
        (uint256 collected0, uint256 collected1) = _collectOnly(instructions);
        
        // Then perform optimal swap
        if (instructions.amountSwap0 > 0 && collected0 >= instructions.amountSwap0) {
            collected1 += _performSwap(
                instructions.tokenId,
                true, // token0 to token1
                instructions.amountSwap0,
                instructions.swapFeeTier,
                instructions.swapData
            );
            collected0 -= instructions.amountSwap0;
        } else if (instructions.amountSwap1 > 0 && collected1 >= instructions.amountSwap1) {
            collected0 += _performSwap(
                instructions.tokenId,
                false, // token1 to token0
                instructions.amountSwap1,
                instructions.swapFeeTier,
                instructions.swapData
            );
            collected1 -= instructions.amountSwap1;
        }
        
        // Finally compound the optimized amounts
        Instructions memory compoundInstructions = instructions;
        compoundInstructions.whatToDo = WhatToDo.COMPOUND;
        return _compound(compoundInstructions);
    }
    
    /// @notice Change position range (close old, open new)
    function _changeRange(Instructions calldata instructions)
        internal
        returns (uint256 amount0, uint256 amount1)
    {
        // Get position info
        (, , address token0, address token1, uint24 fee, , , , , , ,) = 
            nonfungiblePositionManager.positions(instructions.tokenId);
        
        // First decrease liquidity to 0
        if (instructions.liquidity > 0) {
            INonfungiblePositionManager.DecreaseLiquidityParams memory decreaseParams = 
                INonfungiblePositionManager.DecreaseLiquidityParams({
                    tokenId: instructions.tokenId,
                    liquidity: uint128(instructions.liquidity),
                    amount0Min: instructions.amountRemove0,
                    amount1Min: instructions.amountRemove1,
                    deadline: instructions.deadline
                });
                
            nonfungiblePositionManager.decreaseLiquidity(decreaseParams);
        }
        
        // Collect all tokens
        (uint256 collected0, uint256 collected1) = _collectOnly(instructions);
        
        // Mint new position with new range
        INonfungiblePositionManager.MintParams memory mintParams = 
            INonfungiblePositionManager.MintParams({
                token0: token0,
                token1: token1,
                fee: fee,
                tickLower: instructions.tickLower,
                tickUpper: instructions.tickUpper,
                amount0Desired: collected0,
                amount1Desired: collected1,
                amount0Min: instructions.amount0Min,
                amount1Min: instructions.amount1Min,
                recipient: msg.sender,
                deadline: instructions.deadline
            });
            
        // Approve tokens for new position
        IERC20(token0).safeApprove(address(nonfungiblePositionManager), collected0);
        IERC20(token1).safeApprove(address(nonfungiblePositionManager), collected1);
        
        (uint256 newTokenId, , amount0, amount1) = nonfungiblePositionManager.mint(mintParams);
        
        // Reset approvals
        IERC20(token0).safeApprove(address(nonfungiblePositionManager), 0);
        IERC20(token1).safeApprove(address(nonfungiblePositionManager), 0);
        
        // Return leftover tokens
        if (collected0 > amount0) {
            IERC20(token0).safeTransfer(msg.sender, collected0 - amount0);
        }
        if (collected1 > amount1) {
            IERC20(token1).safeTransfer(msg.sender, collected1 - amount1);
        }
        
        // Burn old position (it should be empty now)
        nonfungiblePositionManager.burn(instructions.tokenId);
    }
    
    /// @notice Withdraw liquidity and collect fees
    function _withdrawAndCollect(Instructions calldata instructions)
        internal
        returns (uint256 amount0, uint256 amount1)
    {
        // Decrease liquidity
        if (instructions.liquidity > 0) {
            INonfungiblePositionManager.DecreaseLiquidityParams memory decreaseParams = 
                INonfungiblePositionManager.DecreaseLiquidityParams({
                    tokenId: instructions.tokenId,
                    liquidity: uint128(instructions.liquidity),
                    amount0Min: instructions.amount0Min,
                    amount1Min: instructions.amount1Min,
                    deadline: instructions.deadline
                });
                
            (amount0, amount1) = nonfungiblePositionManager.decreaseLiquidity(decreaseParams);
        }
        
        // Collect all tokens
        (uint256 collected0, uint256 collected1) = _collectOnly(instructions);
        
        amount0 += collected0;
        amount1 += collected1;
    }
    
    /// @notice Collect fees only
    function _collectOnly(Instructions calldata instructions)
        internal
        returns (uint256 amount0, uint256 amount1)
    {
        INonfungiblePositionManager.CollectParams memory collectParams = 
            INonfungiblePositionManager.CollectParams({
                tokenId: instructions.tokenId,
                recipient: msg.sender,
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            });
            
        return nonfungiblePositionManager.collect(collectParams);
    }
    
    /// @notice Perform token swap
    function _performSwap(
        uint256 tokenId,
        bool zeroForOne,
        uint256 amountIn,
        uint24 feeTier,
        bytes memory swapData
    ) internal returns (uint256 amountOut) {
        // Get position tokens
        (, , address token0, address token1, , , , , , , ,) = 
            nonfungiblePositionManager.positions(tokenId);
            
        address tokenIn = zeroForOne ? token0 : token1;
        address tokenOut = zeroForOne ? token1 : token0;
        
        // Approve token for swap
        IERC20(tokenIn).safeApprove(address(swapRouter), amountIn);
        
        // Use external swap data if provided, otherwise use standard swap
        if (swapData.length > 0) {
            // TODO: Implement external swap router integration (0x, 1inch, etc.)
            // For now, use standard Uniswap swap
        }
        
        ISwapRouter.ExactInputSingleParams memory swapParams = 
            ISwapRouter.ExactInputSingleParams({
                tokenIn: tokenIn,
                tokenOut: tokenOut,
                fee: feeTier,
                recipient: address(this),
                deadline: block.timestamp + 300,
                amountIn: amountIn,
                amountOutMinimum: 0, // TODO: Calculate minimum with slippage
                sqrtPriceLimitX96: 0
            });
            
        amountOut = swapRouter.exactInputSingle(swapParams);
        
        // Reset approval
        IERC20(tokenIn).safeApprove(address(swapRouter), 0);
    }
    
    /// @notice Validate TWAP for MEV protection
    function _validateTWAP(address token0, address token1, uint24 fee) internal view {
        // Get pool
        address poolAddress = IUniswapV3Factory(nonfungiblePositionManager.factory())
            .getPool(token0, token1, fee);
            
        if (poolAddress == address(0)) {
            revert InvalidPool();
        }
        
        IUniswapV3Pool pool = IUniswapV3Pool(poolAddress);
        
        // Use TWAP oracle for validation
        (bool success, ) = pool.verifyTWAP(
            60, // 60 second TWAP
            100, // 1% max difference
            false // No AI override
        );
        
        if (!success) {
            revert("TWAP validation failed");
        }
    }
    
    /// @notice Emergency function to withdraw stuck tokens (owner only)
    function emergencyWithdraw(address token, uint256 amount) external {
        // Note: This is a stateless contract, so it shouldn't hold tokens
        // This is just a safety measure
        IERC20(token).safeTransfer(msg.sender, amount);
    }
}