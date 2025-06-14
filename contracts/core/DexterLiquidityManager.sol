// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Factory.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";

import "../interfaces/IDexterLiquidityManager.sol";
import "../interfaces/IFeeDispenser.sol";
import "../libraries/ProfitCalculator.sol";

/**
 * @title DexterLiquidityManager
 * @notice Core contract for automated liquidity management and compounding
 * @dev Manages Uniswap V3 positions with auto-compounding and performance fees
 */
contract DexterLiquidityManager is 
    IDexterLiquidityManager,
    Ownable,
    ReentrancyGuard,
    Pausable,
    IERC721Receiver
{
    using SafeERC20 for IERC20;
    
    /// @notice Constants
    uint256 private constant PRECISION = 1e18;
    uint256 private constant MAX_PROFIT_FEE = 1000; // 10%
    uint256 private constant FEE_DENOMINATOR = 10000;
    
    /// @notice Core contracts
    INonfungiblePositionManager public immutable override positionManager;
    IUniswapV3Factory public immutable override factory;
    ISwapRouter public immutable override swapRouter;
    address public immutable WETH;
    
    /// @notice Fee configuration
    uint256 public override profitFeeRate = 800; // 8%
    uint256 public override compoundThreshold = 1e16; // 0.01 ETH worth
    
    /// @notice Protocol contracts
    address public override dexToken;
    address public feeDispenser;
    
    /// @notice Position tracking
    mapping(uint256 => Position) private positions;
    mapping(address => uint256[]) private userPositions;
    mapping(address => uint256) private userPositionIndex;
    
    /// @notice Profit tracking
    mapping(uint256 => uint256) private initialLiquidity;
    mapping(uint256 => uint256) private totalCompounded;
    
    /// @notice Protocol statistics
    uint256 public override totalFeesCollected;
    uint256 public totalValueLocked;
    uint256 public totalPositions;
    
    modifier onlyPositionOwner(uint256 tokenId) {
        require(positions[tokenId].owner == msg.sender, "Not position owner");
        _;
    }
    
    modifier onlyActivePosition(uint256 tokenId) {
        require(positions[tokenId].isActive, "Position not active");
        _;
    }
    
    constructor(
        address _positionManager,
        address _factory,
        address _swapRouter,
        address _weth,
        address _dexToken,
        address _feeDispenser
    ) {
        positionManager = INonfungiblePositionManager(_positionManager);
        factory = IUniswapV3Factory(_factory);
        swapRouter = ISwapRouter(_swapRouter);
        WETH = _weth;
        dexToken = _dexToken;
        feeDispenser = _feeDispenser;
    }
    
    /**
     * @notice Handles receipt of Uniswap V3 NFT positions
     */
    function onERC721Received(
        address,
        address from,
        uint256 tokenId,
        bytes calldata
    ) external override returns (bytes4) {
        require(msg.sender == address(positionManager), "Only position manager");
        _depositPosition(tokenId, from);
        return this.onERC721Received.selector;
    }
    
    /**
     * @notice Deposits a position NFT
     */
    function depositPosition(uint256 tokenId) external override whenNotPaused {
        positionManager.safeTransferFrom(msg.sender, address(this), tokenId);
    }
    
    /**
     * @notice Internal deposit logic
     */
    function _depositPosition(uint256 tokenId, address owner) private {
        // Get position info
        (
            ,
            ,
            address token0,
            address token1,
            uint24 fee,
            int24 tickLower,
            int24 tickUpper,
            uint128 liquidity,
            ,
            ,
            ,
        ) = positionManager.positions(tokenId);
        
        require(liquidity > 0, "Empty position");
        
        // Record initial state for profit calculation
        initialLiquidity[tokenId] = liquidity;
        
        // Store position
        positions[tokenId] = Position({
            owner: owner,
            depositedAt: block.timestamp,
            lastCompoundedAt: block.timestamp,
            totalProfit0: 0,
            totalProfit1: 0,
            isActive: true
        });
        
        // Track user positions
        userPositions[owner].push(tokenId);
        userPositionIndex[tokenId] = userPositions[owner].length - 1;
        
        // Update stats
        totalPositions++;
        _updateTVL();
        
        emit PositionDeposited(owner, tokenId);
    }
    
    /**
     * @notice Withdraws a position NFT
     */
    function withdrawPosition(
        uint256 tokenId,
        address to
    ) external override nonReentrant onlyPositionOwner(tokenId) {
        require(to != address(0), "Invalid recipient");
        
        Position memory position = positions[tokenId];
        require(position.isActive, "Position not active");
        
        // Mark as inactive
        positions[tokenId].isActive = false;
        
        // Remove from user positions
        _removeUserPosition(position.owner, tokenId);
        
        // Transfer NFT
        positionManager.safeTransferFrom(address(this), to, tokenId);
        
        // Update stats
        totalPositions--;
        _updateTVL();
        
        emit PositionWithdrawn(position.owner, tokenId, to);
    }
    
    /**
     * @notice Auto-compounds a position
     */
    function autoCompound(
        AutoCompoundParams calldata params
    ) external override nonReentrant whenNotPaused onlyActivePosition(params.tokenId) 
      returns (uint256 amount0Added, uint256 amount1Added, uint256 feeCollected) 
    {
        Position storage position = positions[params.tokenId];
        
        // Collect fees from position
        (uint256 amount0, uint256 amount1) = _collectFees(params.tokenId);
        
        // Check if profitable to compound
        (bool isProfitable, uint256 profit0, uint256 profit1) = 
            _calculateProfit(params.tokenId, amount0, amount1);
        
        require(isProfitable, "Not profitable to compound");
        
        // Calculate protocol fee (8% of profit)
        uint256 fee0 = (profit0 * profitFeeRate) / FEE_DENOMINATOR;
        uint256 fee1 = (profit1 * profitFeeRate) / FEE_DENOMINATOR;
        
        // Amount to compound (92% of profit + principal)
        uint256 compound0 = amount0 - fee0;
        uint256 compound1 = amount1 - fee1;
        
        // Get position details
        (
            ,
            ,
            address token0,
            address token1,
            uint24 fee,
            int24 tickLower,
            int24 tickUpper,
            ,
            ,
            ,
            ,
        ) = positionManager.positions(params.tokenId);
        
        // Swap if needed to match position ratio
        if (params.convertToken0ToToken1 || params.convertToken1ToToken0) {
            (compound0, compound1) = _optimizeAmounts(
                token0,
                token1,
                fee,
                tickLower,
                tickUpper,
                compound0,
                compound1,
                params.convertToken0ToToken1
            );
        }
        
        // Add liquidity back to position
        if (compound0 > 0 || compound1 > 0) {
            // Approve tokens
            IERC20(token0).safeApprove(address(positionManager), compound0);
            IERC20(token1).safeApprove(address(positionManager), compound1);
            
            // Increase liquidity
            (, amount0Added, amount1Added) = positionManager.increaseLiquidity(
                INonfungiblePositionManager.IncreaseLiquidityParams({
                    tokenId: params.tokenId,
                    amount0Desired: compound0,
                    amount1Desired: compound1,
                    amount0Min: 0, // TODO: Add slippage protection
                    amount1Min: 0,
                    deadline: params.deadline
                })
            );
            
            // Track compounded amounts
            totalCompounded[params.tokenId] += amount0Added + amount1Added;
        }
        
        // Send fees to dispenser
        if (fee0 > 0) {
            IERC20(token0).safeTransfer(feeDispenser, fee0);
            IFeeDispenser(feeDispenser).depositFees(token0, fee0);
        }
        if (fee1 > 0) {
            IERC20(token1).safeTransfer(feeDispenser, fee1);
            IFeeDispenser(feeDispenser).depositFees(token1, fee1);
        }
        
        // Update position tracking
        position.lastCompoundedAt = block.timestamp;
        position.totalProfit0 += profit0;
        position.totalProfit1 += profit1;
        
        // Update protocol stats
        feeCollected = fee0 + fee1;
        totalFeesCollected += feeCollected;
        
        emit AutoCompounded(
            params.tokenId,
            amount0Added,
            amount1Added,
            profit0,
            profit1,
            feeCollected
        );
    }
    
    /**
     * @notice Batch auto-compounds multiple positions
     */
    function batchAutoCompound(
        AutoCompoundParams[] calldata params
    ) external override {
        for (uint256 i = 0; i < params.length; i++) {
            try this.autoCompound(params[i]) {} catch {}
        }
    }
    
    /**
     * @notice Gets position information
     */
    function getPosition(uint256 tokenId) external view override returns (Position memory) {
        return positions[tokenId];
    }
    
    /**
     * @notice Checks if position is profitable to compound
     */
    function checkCompoundProfitability(uint256 tokenId) 
        external 
        view 
        override 
        returns (bool isProfitable, uint256 profit0, uint256 profit1) 
    {
        // Simulate fee collection
        (uint256 amount0, uint256 amount1) = _simulateCollectFees(tokenId);
        
        // Calculate profit
        return _calculateProfit(tokenId, amount0, amount1);
    }
    
    /**
     * @notice Updates reward parameters
     */
    function updateRewardParameters(
        uint256 _profitFeeRate,
        uint256 _compoundThreshold
    ) external override onlyOwner {
        require(_profitFeeRate <= MAX_PROFIT_FEE, "Fee too high");
        
        profitFeeRate = _profitFeeRate;
        compoundThreshold = _compoundThreshold;
        
        emit RewardParametersUpdated(_profitFeeRate, _compoundThreshold);
    }
    
    /**
     * @notice Emergency withdraw
     */
    function emergencyWithdraw(uint256 tokenId) external override onlyOwner {
        Position memory position = positions[tokenId];
        require(position.isActive, "Position not active");
        
        positions[tokenId].isActive = false;
        positionManager.safeTransferFrom(address(this), position.owner, tokenId);
        
        emit PositionWithdrawn(position.owner, tokenId, position.owner);
    }
    
    // Internal functions would continue here...
    // Including _collectFees, _calculateProfit, _optimizeAmounts, etc.
}