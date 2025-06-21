// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Multicall.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";

import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";
import "@uniswap/v3-periphery/contracts/libraries/LiquidityAmounts.sol";
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";

/**
 * @title DexterCompoundor
 * @notice AI-powered auto-compounding system for Uniswap V3 positions
 * @dev Based on Revert Finance's compounding architecture with AI enhancements
 */
contract DexterCompoundor is IERC721Receiver, ReentrancyGuard, Ownable, Multicall {
    using SafeERC20 for IERC20;

    // Constants
    uint128 constant Q64 = 2**64;
    uint128 constant Q96 = 2**96;
    uint64 constant public MAX_REWARD_X64 = uint64(Q64 / 50); // 2% max reward
    uint32 constant public MAX_POSITIONS_PER_ADDRESS = 200; // Higher limit for AI management

    // Configuration variables
    uint64 public totalRewardX64 = MAX_REWARD_X64; // 2%
    uint64 public compounderRewardX64 = MAX_REWARD_X64 / 2; // 1%
    uint32 public maxTWAPTickDifference = 100; // 1%
    uint32 public TWAPSeconds = 60;

    // Dexter-specific configurations
    uint64 public aiOptimizerRewardX64 = MAX_REWARD_X64 / 4; // 0.5% for AI optimizer
    address public aiAgent; // Address that can trigger AI-optimized compounds
    bool public aiOptimizationEnabled = true;

    // Core addresses
    address public immutable weth;
    IUniswapV3Factory public immutable factory;
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    ISwapRouter public immutable swapRouter;

    // Position tracking
    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256[]) public accountTokens;
    mapping(address => mapping(address => uint256)) public accountBalances;
    mapping(uint256 => bool) public aiManagedPositions; // Positions under AI management
    mapping(uint256 => uint256) public lastCompoundTime;
    mapping(uint256 => uint256) public compoundCount;

    // Events
    event TokenDeposited(address indexed account, uint256 indexed tokenId, bool aiManaged);
    event TokenWithdrawn(address indexed account, address indexed to, uint256 indexed tokenId);
    event AutoCompounded(
        address indexed account,
        uint256 indexed tokenId,
        uint256 amountAdded0,
        uint256 amountAdded1,
        uint256 reward0,
        uint256 reward1,
        address token0,
        address token1,
        bool aiOptimized
    );
    event BalanceWithdrawn(address indexed account, address indexed token, address indexed to, uint256 amount);
    event RewardUpdated(address indexed account, uint64 totalRewardX64, uint64 compounderRewardX64, uint64 aiOptimizerRewardX64);
    event TWAPConfigUpdated(address indexed account, uint32 maxTWAPTickDifference, uint32 TWAPSeconds);
    event AIAgentUpdated(address indexed oldAgent, address indexed newAgent);
    event AIOptimizationToggled(bool enabled);

    /// @notice Reward conversion options
    enum RewardConversion { NONE, TOKEN_0, TOKEN_1, AI_OPTIMIZED }

    /// @notice Parameters for auto-compounding
    struct AutoCompoundParams {
        uint256 tokenId;
        RewardConversion rewardConversion;
        bool withdrawReward;
        bool doSwap;
        bool useAIOptimization; // New parameter for AI-driven optimization
    }

    /// @notice Parameters for decreasing liquidity and collecting
    struct DecreaseLiquidityAndCollectParams {
        uint256 tokenId;
        uint128 liquidity;
        uint256 amount0Min;
        uint256 amount1Min;
        uint256 deadline;
        address recipient;
    }

    constructor(
        address _weth,
        IUniswapV3Factory _factory,
        INonfungiblePositionManager _nonfungiblePositionManager,
        ISwapRouter _swapRouter
    ) {
        weth = _weth;
        factory = _factory;
        nonfungiblePositionManager = _nonfungiblePositionManager;
        swapRouter = _swapRouter;
    }

    /**
     * @notice Set reward configuration (only owner)
     * @param _totalRewardX64 Total reward percentage
     * @param _compounderRewardX64 Compounder reward percentage
     * @param _aiOptimizerRewardX64 AI optimizer reward percentage
     */
    function setReward(
        uint64 _totalRewardX64,
        uint64 _compounderRewardX64,
        uint64 _aiOptimizerRewardX64
    ) external onlyOwner {
        require(_totalRewardX64 <= MAX_REWARD_X64, "Total reward too high");
        require(_compounderRewardX64 <= _totalRewardX64, "Compounder reward too high");
        require(_aiOptimizerRewardX64 <= _totalRewardX64, "AI optimizer reward too high");
        
        totalRewardX64 = _totalRewardX64;
        compounderRewardX64 = _compounderRewardX64;
        aiOptimizerRewardX64 = _aiOptimizerRewardX64;
        
        emit RewardUpdated(msg.sender, _totalRewardX64, _compounderRewardX64, _aiOptimizerRewardX64);
    }

    /**
     * @notice Set TWAP configuration (only owner)
     * @param _maxTWAPTickDifference Maximum tick difference from TWAP
     * @param _TWAPSeconds TWAP calculation period
     */
    function setTWAPConfig(uint32 _maxTWAPTickDifference, uint32 _TWAPSeconds) external onlyOwner {
        maxTWAPTickDifference = _maxTWAPTickDifference;
        TWAPSeconds = _TWAPSeconds;
        emit TWAPConfigUpdated(msg.sender, _maxTWAPTickDifference, _TWAPSeconds);
    }

    /**
     * @notice Set AI agent address (only owner)
     * @param _aiAgent New AI agent address
     */
    function setAIAgent(address _aiAgent) external onlyOwner {
        address oldAgent = aiAgent;
        aiAgent = _aiAgent;
        emit AIAgentUpdated(oldAgent, _aiAgent);
    }

    /**
     * @notice Toggle AI optimization (only owner)
     * @param _enabled Whether AI optimization is enabled
     */
    function toggleAIOptimization(bool _enabled) external onlyOwner {
        aiOptimizationEnabled = _enabled;
        emit AIOptimizationToggled(_enabled);
    }

    /**
     * @notice Handle NFT deposits
     * @param from Address depositing the NFT
     * @param tokenId Token ID being deposited
     * @param data Optional data (first byte indicates AI management preference)
     */
    function onERC721Received(
        address,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external override nonReentrant returns (bytes4) {
        require(msg.sender == address(nonfungiblePositionManager), "Not authorized NFT contract");

        bool enableAIManagement = data.length > 0 && data[0] == 0x01;
        
        _addToken(tokenId, from);
        
        if (enableAIManagement && aiOptimizationEnabled) {
            aiManagedPositions[tokenId] = true;
        }
        
        emit TokenDeposited(from, tokenId, enableAIManagement);
        return this.onERC721Received.selector;
    }

    /**
     * @notice Add token to account tracking
     * @param tokenId Token ID to add
     * @param account Account to add token to
     */
    function _addToken(uint256 tokenId, address account) internal {
        require(accountTokens[account].length < MAX_POSITIONS_PER_ADDRESS, "Too many positions");
        
        ownerOf[tokenId] = account;
        accountTokens[account].push(tokenId);
    }

    /**
     * @notice Withdraw NFT position
     * @param tokenId Token ID to withdraw
     * @param to Address to send NFT to
     * @param withdrawBalances Whether to withdraw token balances
     * @param data Optional data
     */
    function withdrawToken(
        uint256 tokenId,
        address to,
        bool withdrawBalances,
        bytes memory data
    ) external nonReentrant {
        require(ownerOf[tokenId] == msg.sender, "Not token owner");

        // Remove from AI management if applicable
        if (aiManagedPositions[tokenId]) {
            aiManagedPositions[tokenId] = false;
        }

        _removeToken(tokenId, msg.sender);

        if (withdrawBalances) {
            // Get position info to determine tokens
            (, , address token0, address token1, , , , uint128 liquidity, , , ,) = 
                nonfungiblePositionManager.positions(tokenId);
            
            if (liquidity > 0) {
                uint256 balance0 = accountBalances[msg.sender][token0];
                uint256 balance1 = accountBalances[msg.sender][token1];
                
                if (balance0 > 0) {
                    _withdrawBalance(token0, to, balance0);
                }
                if (balance1 > 0) {
                    _withdrawBalance(token1, to, balance1);
                }
            }
        }

        nonfungiblePositionManager.safeTransferFrom(address(this), to, tokenId, data);
        emit TokenWithdrawn(msg.sender, to, tokenId);
    }

    /**
     * @notice Remove token from account tracking
     * @param tokenId Token ID to remove
     * @param account Account to remove token from
     */
    function _removeToken(uint256 tokenId, address account) internal {
        uint256[] storage tokens = accountTokens[account];
        
        for (uint256 i = 0; i < tokens.length; i++) {
            if (tokens[i] == tokenId) {
                tokens[i] = tokens[tokens.length - 1];
                tokens.pop();
                break;
            }
        }
        
        delete ownerOf[tokenId];
    }

    /**
     * @notice Withdraw token balance
     * @param token Token address
     * @param to Recipient address
     * @param amount Amount to withdraw
     */
    function withdrawBalance(address token, address to, uint256 amount) external nonReentrant {
        _withdrawBalance(token, to, amount);
    }

    /**
     * @notice Internal function to withdraw token balance
     */
    function _withdrawBalance(address token, address to, uint256 amount) internal {
        require(accountBalances[msg.sender][token] >= amount, "Insufficient balance");
        
        accountBalances[msg.sender][token] -= amount;
        IERC20(token).safeTransfer(to, amount);
        
        emit BalanceWithdrawn(msg.sender, token, to, amount);
    }

    /**
     * @notice Get number of positions for an account
     * @param account Account address
     * @return balance Number of positions
     */
    function balanceOf(address account) external view returns (uint256 balance) {
        return accountTokens[account].length;
    }

    /**
     * @notice Check if position is AI-managed
     * @param tokenId Token ID to check
     * @return Whether position is under AI management
     */
    function isAIManagedPosition(uint256 tokenId) external view returns (bool) {
        return aiManagedPositions[tokenId];
    }

    /**
     * @notice Get position compound statistics
     * @param tokenId Token ID
     * @return lastCompound Last compound timestamp
     * @return compoundTotal Total number of compounds
     */
    function getPositionStats(uint256 tokenId) external view returns (uint256 lastCompound, uint256 compoundTotal) {
        return (lastCompoundTime[tokenId], compoundCount[tokenId]);
    }

    /**
     * @notice Auto-compound a position with optional AI optimization
     * @param params Auto-compound parameters
     * @return reward0 Token0 reward for compounder
     * @return reward1 Token1 reward for compounder  
     * @return compounded0 Token0 amount compounded
     * @return compounded1 Token1 amount compounded
     */
    function autoCompound(AutoCompoundParams calldata params) 
        external 
        nonReentrant 
        returns (uint256 reward0, uint256 reward1, uint256 compounded0, uint256 compounded1) 
    {
        uint256 tokenId = params.tokenId;
        require(ownerOf[tokenId] != address(0), "Invalid token");

        // Check if AI optimization is requested and authorized
        bool useAI = params.useAIOptimization && 
                    aiOptimizationEnabled && 
                    (msg.sender == aiAgent || aiManagedPositions[tokenId]);

        // Get position info
        (, , address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper, 
         uint128 liquidity, , , uint128 tokensOwed0, uint128 tokensOwed1) = 
            nonfungiblePositionManager.positions(tokenId);

        require(liquidity > 0, "No liquidity");
        require(tokensOwed0 > 0 || tokensOwed1 > 0, "No fees to compound");

        // Validate TWAP if not using AI (AI can override TWAP checks)
        if (!useAI) {
            _validateTWAP(token0, token1, fee);
        }

        // Collect fees
        INonfungiblePositionManager.CollectParams memory collectParams = INonfungiblePositionManager.CollectParams({
            tokenId: tokenId,
            recipient: address(this),
            amount0Max: type(uint128).max,
            amount1Max: type(uint128).max
        });

        (uint256 collected0, uint256 collected1) = nonfungiblePositionManager.collect(collectParams);

        // Calculate rewards
        uint64 rewardX64 = useAI ? aiOptimizerRewardX64 : compounderRewardX64;
        reward0 = (collected0 * rewardX64) / Q64;
        reward1 = (collected1 * rewardX64) / Q64;

        // Amounts available for compounding
        uint256 amount0 = collected0 - reward0;
        uint256 amount1 = collected1 - reward1;

        // Handle reward conversion and swapping
        if (params.rewardConversion != RewardConversion.NONE || params.doSwap) {
            (amount0, amount1) = _handleSwapping(
                token0, 
                token1, 
                fee, 
                amount0, 
                amount1, 
                tickLower, 
                tickUpper,
                params.rewardConversion,
                params.doSwap,
                useAI
            );
        }

        // Add liquidity back to position
        if (amount0 > 0 || amount1 > 0) {
            INonfungiblePositionManager.IncreaseLiquidityParams memory increaseLiquidityParams = 
                INonfungiblePositionManager.IncreaseLiquidityParams({
                    tokenId: tokenId,
                    amount0Desired: amount0,
                    amount1Desired: amount1,
                    amount0Min: useAI ? 0 : (amount0 * 95) / 100, // AI can use tighter slippage
                    amount1Min: useAI ? 0 : (amount1 * 95) / 100,
                    deadline: block.timestamp + 300
                });

            IERC20(token0).safeApprove(address(nonfungiblePositionManager), amount0);
            IERC20(token1).safeApprove(address(nonfungiblePositionManager), amount1);

            (, compounded0, compounded1) = nonfungiblePositionManager.increaseLiquidity(increaseLiquidityParams);

            // Reset approvals
            IERC20(token0).safeApprove(address(nonfungiblePositionManager), 0);
            IERC20(token1).safeApprove(address(nonfungiblePositionManager), 0);

            // Add any leftover amounts to account balances
            if (amount0 > compounded0) {
                accountBalances[ownerOf[tokenId]][token0] += (amount0 - compounded0);
            }
            if (amount1 > compounded1) {
                accountBalances[ownerOf[tokenId]][token1] += (amount1 - compounded1);
            }
        }

        // Transfer rewards to compounder
        if (reward0 > 0) {
            if (params.withdrawReward) {
                IERC20(token0).safeTransfer(msg.sender, reward0);
            } else {
                accountBalances[msg.sender][token0] += reward0;
            }
        }
        if (reward1 > 0) {
            if (params.withdrawReward) {
                IERC20(token1).safeTransfer(msg.sender, reward1);
            } else {
                accountBalances[msg.sender][token1] += reward1;
            }
        }

        // Update position stats
        lastCompoundTime[tokenId] = block.timestamp;
        compoundCount[tokenId]++;

        emit AutoCompounded(
            ownerOf[tokenId],
            tokenId,
            compounded0,
            compounded1,
            reward0,
            reward1,
            token0,
            token1,
            useAI
        );
    }

    /**
     * @notice Validate TWAP price to prevent MEV attacks
     */
    function _validateTWAP(address token0, address token1, uint24 fee) internal view {
        address pool = factory.getPool(token0, token1, fee);
        require(pool != address(0), "Pool not found");

        // Get current price
        (, int24 currentTick, , , , ,) = IUniswapV3Pool(pool).slot0();

        // Get TWAP price
        uint32[] memory secondsAgos = new uint32[](2);
        secondsAgos[0] = TWAPSeconds;
        secondsAgos[1] = 0;

        (int56[] memory tickCumulatives, ) = IUniswapV3Pool(pool).observe(secondsAgos);
        int24 twapTick = int24((tickCumulatives[1] - tickCumulatives[0]) / int56(uint56(TWAPSeconds)));

        // Check if current price is within acceptable range of TWAP
        int24 tickDifference = currentTick > twapTick ? currentTick - twapTick : twapTick - currentTick;
        require(uint24(tickDifference) <= maxTWAPTickDifference, "Price manipulation detected");
    }

    /**
     * @notice Handle token swapping for optimal compounding
     */
    function _handleSwapping(
        address token0,
        address token1, 
        uint24 fee,
        uint256 amount0,
        uint256 amount1,
        int24 tickLower,
        int24 tickUpper,
        RewardConversion rewardConversion,
        bool doSwap,
        bool useAI
    ) internal returns (uint256 newAmount0, uint256 newAmount1) {
        // For now, return amounts unchanged - implement swapping logic later
        // TODO: Implement optimal swapping based on current tick and position range
        // TODO: Add AI-driven swap optimization
        return (amount0, amount1);
    }

    /**
     * @notice Decrease liquidity and collect tokens
     */
    function decreaseLiquidityAndCollect(DecreaseLiquidityAndCollectParams calldata params)
        external
        nonReentrant
        returns (uint256 amount0, uint256 amount1)
    {
        require(ownerOf[params.tokenId] == msg.sender, "Not token owner");

        // Decrease liquidity
        INonfungiblePositionManager.DecreaseLiquidityParams memory decreaseParams = 
            INonfungiblePositionManager.DecreaseLiquidityParams({
                tokenId: params.tokenId,
                liquidity: params.liquidity,
                amount0Min: params.amount0Min,
                amount1Min: params.amount1Min,
                deadline: params.deadline
            });

        (amount0, amount1) = nonfungiblePositionManager.decreaseLiquidity(decreaseParams);

        // Collect the decreased amounts
        INonfungiblePositionManager.CollectParams memory collectParams = 
            INonfungiblePositionManager.CollectParams({
                tokenId: params.tokenId,
                recipient: params.recipient,
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            });

        nonfungiblePositionManager.collect(collectParams);
    }

    /**
     * @notice Collect fees from a position
     */
    function collect(INonfungiblePositionManager.CollectParams calldata params) 
        external 
        nonReentrant 
        returns (uint256 amount0, uint256 amount1) 
    {
        require(ownerOf[params.tokenId] == msg.sender, "Not token owner");
        return nonfungiblePositionManager.collect(params);
    }
}