// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Multicall.sol";

import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Factory.sol";
import "./vendor/uniswap/libraries/TickMath.sol";
import "./vendor/uniswap/libraries/LiquidityAmounts.sol";
import "./vendor/uniswap/interfaces/INonfungiblePositionManager.sol";
import "./vendor/uniswap/interfaces/IPeripheryImmutableState.sol";
import "./vendor/uniswap/interfaces/ISwapRouter.sol";

/**
 * @title DexterMVP
 * @notice Ultra-high frequency auto-compounding and bin-based rebalancing for Uniswap V3 positions
 * @dev Simplified version of DexterCompoundor with deterministic automation (no AI dependencies)
 *      Optimized for Base chain's low gas costs with 5-minute compound intervals
 */
contract DexterMVP is IERC721Receiver, ReentrancyGuard, Pausable, Ownable, Multicall {
    using SafeERC20 for IERC20;

    // ============ CONSTANTS ============
    
    uint128 constant Q64 = 2**64;
    uint128 constant Q96 = 2**96;
    uint64 constant public MAX_REWARD_X64 = uint64(Q64 / 50); // 2% max reward
    uint32 constant public MAX_POSITIONS_PER_ADDRESS = 200;
    
    // MVP-specific ultra-frequent settings
    uint256 constant public MIN_COMPOUND_THRESHOLD_USD = 5e17; // $0.50 minimum (Base optimized)
    uint256 constant public ULTRA_FREQUENT_INTERVAL = 5 minutes; // 5-minute compounds
    uint256 constant public MAX_BIN_DRIFT = 3; // Max bins from price before rebalance
    uint256 constant public DEFAULT_CONCENTRATION_LEVEL = 5; // 1-10 scale

    // ============ STATE VARIABLES ============
    
    // Core Uniswap interfaces
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    ISwapRouter public immutable swapRouter;
    address public immutable weth;
    
    // Position management
    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256[]) public accountTokens;
    mapping(address => mapping(address => uint256)) public accountBalances;
    
    // MVP Automation Settings
    struct AutomationSettings {
        bool autoCompoundEnabled;
        bool autoRebalanceEnabled;
        uint256 compoundThresholdUSD;    // Min fees to trigger compound
        uint256 maxBinsFromPrice;        // Max bins before rebalance
        uint256 concentrationLevel;      // 1-10 concentration scale
        uint256 lastCompoundTime;
        uint256 lastRebalanceTime;
    }
    
    mapping(uint256 => AutomationSettings) public positionAutomation;
    
    // Bin tracking for rebalancing
    struct BinPosition {
        int24 currentTick;
        int24 positionTickLower;
        int24 positionTickUpper;
        uint256 binsFromPrice;
        bool inRange;
    }
    
    // Keeper authorization
    mapping(address => bool) public authorizedKeepers;
    
    // Performance tracking
    mapping(uint256 => uint256) public compoundCount;
    mapping(uint256 => uint256) public rebalanceCount;
    mapping(uint256 => uint256) public totalFeesCompounded;
    
    // ============ EVENTS ============
    
    event PositionDeposited(address indexed account, uint256 indexed tokenId, AutomationSettings settings);
    event PositionWithdrawn(address indexed account, uint256 indexed tokenId);
    event UltraFrequentCompound(
        uint256 indexed tokenId,
        uint256 amountAdded0,
        uint256 amountAdded1,
        uint256 feesUSD,
        uint256 compoundNumber
    );
    event BinBasedRebalance(
        uint256 indexed oldTokenId,
        uint256 indexed newTokenId,
        uint256 binsFromPrice,
        int24 newTickLower,
        int24 newTickUpper
    );
    event AutomationSettingsUpdated(uint256 indexed tokenId, AutomationSettings settings);
    event KeeperAuthorized(address indexed keeper, bool authorized);
    
    // ============ MODIFIERS ============
    
    modifier onlyAuthorizedKeeper() {
        require(authorizedKeepers[msg.sender] || msg.sender == owner(), "Not authorized keeper");
        _;
    }
    
    modifier validPosition(uint256 tokenId) {
        require(ownerOf[tokenId] != address(0), "Position not deposited");
        _;
    }
    
    // ============ CONSTRUCTOR ============
    
    constructor(
        INonfungiblePositionManager _nonfungiblePositionManager,
        ISwapRouter _swapRouter,
        address _weth
    ) {
        nonfungiblePositionManager = _nonfungiblePositionManager;
        swapRouter = _swapRouter;
        weth = _weth;
    }
    
    // ============ POSITION MANAGEMENT ============
    
    /**
     * @notice Deposit a Uniswap V3 position for MVP automation
     * @param tokenId The NFT token ID
     * @param settings Automation configuration
     */
    function depositPosition(
        uint256 tokenId,
        AutomationSettings memory settings
    ) external nonReentrant whenNotPaused {
        // Enforce position limit per address (RISK-003 fix)
        require(
            accountTokens[msg.sender].length < MAX_POSITIONS_PER_ADDRESS,
            "Position limit exceeded"
        );

        // Transfer NFT to this contract
        nonfungiblePositionManager.safeTransferFrom(msg.sender, address(this), tokenId);
        
        // Validate and set automation settings
        _validateAutomationSettings(settings);
        settings.lastCompoundTime = block.timestamp;
        settings.lastRebalanceTime = block.timestamp;
        
        // Store position data
        ownerOf[tokenId] = msg.sender;
        accountTokens[msg.sender].push(tokenId);
        positionAutomation[tokenId] = settings;
        
        emit PositionDeposited(msg.sender, tokenId, settings);
    }
    
    /**
     * @notice Withdraw a position from MVP automation
     * @param tokenId The NFT token ID
     */
    function withdrawPosition(uint256 tokenId) external nonReentrant validPosition(tokenId) {
        require(ownerOf[tokenId] == msg.sender, "Not position owner");
        
        // Remove from tracking
        _removeFromAccountTokens(msg.sender, tokenId);
        delete ownerOf[tokenId];
        delete positionAutomation[tokenId];
        
        // Transfer NFT back to owner
        nonfungiblePositionManager.safeTransferFrom(address(this), msg.sender, tokenId);
        
        emit PositionWithdrawn(msg.sender, tokenId);
    }
    
    // ============ ULTRA-FREQUENT COMPOUNDING ============
    
    /**
     * @notice Check if position should be compounded (5-minute interval or fee threshold)
     * @param tokenId The position to check
     * @return shouldCompound Whether position needs compounding
     */
    function shouldCompound(uint256 tokenId) public view validPosition(tokenId) returns (bool) {
        AutomationSettings memory settings = positionAutomation[tokenId];
        if (!settings.autoCompoundEnabled) return false;
        
        // Check time interval (5 minutes)
        if (block.timestamp >= settings.lastCompoundTime + ULTRA_FREQUENT_INTERVAL) {
            return true;
        }
        
        // Check fee threshold
        uint256 feesUSD = _getUnclaimedFeesUSD(tokenId);
        return feesUSD >= settings.compoundThresholdUSD;
    }
    
    /**
     * @notice Execute ultra-frequent compound for a single position
     * @param tokenId The position to compound
     */
    function executeCompound(uint256 tokenId) external onlyAuthorizedKeeper validPosition(tokenId) whenNotPaused {
        require(shouldCompound(tokenId), "Compound not needed");
        
        AutomationSettings storage settings = positionAutomation[tokenId];
        
        // Collect fees
        (uint256 amount0, uint256 amount1) = nonfungiblePositionManager.collect(
            INonfungiblePositionManager.CollectParams({
                tokenId: tokenId,
                recipient: address(this),
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            })
        );
        
        if (amount0 > 0 || amount1 > 0) {
            // Increase liquidity with collected fees
            nonfungiblePositionManager.increaseLiquidity(
                INonfungiblePositionManager.IncreaseLiquidityParams({
                    tokenId: tokenId,
                    amount0Desired: amount0,
                    amount1Desired: amount1,
                    amount0Min: 0,
                    amount1Min: 0,
                    deadline: block.timestamp + 300
                })
            );
            
            // Update tracking
            settings.lastCompoundTime = block.timestamp;
            compoundCount[tokenId]++;
            totalFeesCompounded[tokenId] += _calculateFeesUSD(amount0, amount1, tokenId);
            
            emit UltraFrequentCompound(
                tokenId,
                amount0,
                amount1,
                _calculateFeesUSD(amount0, amount1, tokenId),
                compoundCount[tokenId]
            );
        }
    }
    
    /**
     * @notice Batch compound multiple positions for gas efficiency
     * @param tokenIds Array of positions to compound
     */
    function batchCompound(uint256[] calldata tokenIds) external onlyAuthorizedKeeper whenNotPaused {
        uint256 successCount = 0;
        
        for (uint256 i = 0; i < tokenIds.length; i++) {
            if (ownerOf[tokenIds[i]] != address(0) && shouldCompound(tokenIds[i])) {
                try this.executeCompound(tokenIds[i]) {
                    successCount++;
                } catch {
                    // Continue with next position if one fails
                    continue;
                }
            }
        }
        
        require(successCount > 0, "No positions compounded");
    }
    
    // ============ BIN-BASED REBALANCING ============
    
    /**
     * @notice Calculate current bin position for a token
     * @param tokenId The position to analyze
     * @return binData Current bin position data
     */
    function calculateBinPosition(uint256 tokenId) public view validPosition(tokenId) returns (BinPosition memory) {
        (,, address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper,,,,,) = 
            nonfungiblePositionManager.positions(tokenId);
        
        address factoryAddress = IPeripheryImmutableState(address(nonfungiblePositionManager)).factory();
        IUniswapV3Pool pool = IUniswapV3Pool(
            IUniswapV3Factory(factoryAddress).getPool(token0, token1, fee)
        );
        
        (, int24 currentTick,,,,,) = pool.slot0();
        
        uint256 binsFromPrice = 0;
        bool inRange = currentTick >= tickLower && currentTick <= tickUpper;
        
        if (!inRange) {
            int24 tickSpacing = pool.tickSpacing();
            if (currentTick < tickLower) {
                binsFromPrice = uint256(int256(tickLower - currentTick)) / uint256(int256(tickSpacing));
            } else {
                binsFromPrice = uint256(int256(currentTick - tickUpper)) / uint256(int256(tickSpacing));
            }
        }
        
        return BinPosition({
            currentTick: currentTick,
            positionTickLower: tickLower,
            positionTickUpper: tickUpper,
            binsFromPrice: binsFromPrice,
            inRange: inRange
        });
    }
    
    /**
     * @notice Check if position should be rebalanced based on bin drift
     * @param tokenId The position to check
     * @return shouldRebalance Whether position needs rebalancing
     */
    function shouldRebalance(uint256 tokenId) public view validPosition(tokenId) returns (bool) {
        AutomationSettings memory settings = positionAutomation[tokenId];
        if (!settings.autoRebalanceEnabled) return false;
        
        BinPosition memory binData = calculateBinPosition(tokenId);
        
        // Rebalance if out of range or too many bins away
        return !binData.inRange || binData.binsFromPrice >= settings.maxBinsFromPrice;
    }
    
    /**
     * @notice Execute bin-based rebalance for concentrated liquidity
     * @param tokenId The position to rebalance
     */
    function executeRebalance(uint256 tokenId) external onlyAuthorizedKeeper validPosition(tokenId) whenNotPaused {
        require(shouldRebalance(tokenId), "Rebalance not needed");
        
        BinPosition memory binData = calculateBinPosition(tokenId);
        AutomationSettings storage settings = positionAutomation[tokenId];
        
        // Close current position
        (uint256 amount0, uint256 amount1) = _closePosition(tokenId);
        
        // Calculate new concentrated range
        (int24 newTickLower, int24 newTickUpper) = _calculateConcentratedRange(
            binData.currentTick,
            settings.concentrationLevel,
            tokenId
        );
        
        // Open new position
        uint256 newTokenId = _openNewPosition(
            tokenId,
            newTickLower,
            newTickUpper,
            amount0,
            amount1
        );
        
        // Update tracking
        settings.lastRebalanceTime = block.timestamp;
        rebalanceCount[tokenId]++;
        
        emit BinBasedRebalance(
            tokenId,
            newTokenId,
            binData.binsFromPrice,
            newTickLower,
            newTickUpper
        );
    }
    
    // ============ CONFIGURATION ============
    
    /**
     * @notice Update automation settings for a position
     * @param tokenId The position to update
     * @param settings New automation settings
     */
    function updateAutomationSettings(
        uint256 tokenId,
        AutomationSettings memory settings
    ) external validPosition(tokenId) {
        require(ownerOf[tokenId] == msg.sender, "Not position owner");
        
        _validateAutomationSettings(settings);
        
        // Preserve timestamps
        settings.lastCompoundTime = positionAutomation[tokenId].lastCompoundTime;
        settings.lastRebalanceTime = positionAutomation[tokenId].lastRebalanceTime;
        
        positionAutomation[tokenId] = settings;
        
        emit AutomationSettingsUpdated(tokenId, settings);
    }
    
    /**
     * @notice Authorize or revoke keeper access
     * @param keeper Address to authorize/revoke
     * @param authorized Whether to authorize or revoke
     */
    function setKeeperAuthorization(address keeper, bool authorized) external onlyOwner {
        authorizedKeepers[keeper] = authorized;
        emit KeeperAuthorized(keeper, authorized);
    }

    // ============ EMERGENCY CONTROLS (RISK-006 fix) ============

    /**
     * @notice Pause all position operations in case of emergency
     * @dev Can only be called by owner
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @notice Unpause position operations after emergency is resolved
     * @dev Can only be called by owner
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    // ============ INTERNAL FUNCTIONS ============
    
    function _validateAutomationSettings(AutomationSettings memory settings) internal pure {
        require(settings.compoundThresholdUSD >= MIN_COMPOUND_THRESHOLD_USD / 10, "Threshold too low");
        require(settings.compoundThresholdUSD <= 100e18, "Threshold too high");
        require(settings.maxBinsFromPrice >= 1 && settings.maxBinsFromPrice <= 10, "Invalid bin drift");
        require(settings.concentrationLevel >= 1 && settings.concentrationLevel <= 10, "Invalid concentration");
    }
    
    /**
     * @notice PLACEHOLDER: Get unclaimed fees in USD value
     * @dev CRITICAL (RISK-001): This function returns a hardcoded $1 USD value.
     *      Production implementation requires:
     *      1. Integration with price oracle (Chainlink, Uniswap TWAP, etc.)
     *      2. Fetching actual unclaimed fees from position
     *      3. Converting token amounts to USD using oracle prices
     * @param tokenId The position to check (currently unused)
     * @return feesUSD Always returns 1e18 (placeholder $1 USD)
     */
    function _getUnclaimedFeesUSD(uint256 tokenId) internal view returns (uint256) {
        // PLACEHOLDER: Returns $1 USD regardless of actual fees
        // TODO: Integrate price oracle for real USD conversion
        return 1e18;
    }

    /**
     * @notice PLACEHOLDER: Calculate fees in USD value
     * @dev CRITICAL (RISK-001): This function returns a hardcoded $1 USD value.
     *      Production implementation requires:
     *      1. Price oracle integration for token0 and token1
     *      2. Proper decimal handling for different tokens
     *      3. Slippage/spread consideration
     * @param amount0 Amount of token0 (currently unused)
     * @param amount1 Amount of token1 (currently unused)
     * @param tokenId The position ID (currently unused)
     * @return feesUSD Always returns 1e18 (placeholder $1 USD)
     */
    function _calculateFeesUSD(uint256 amount0, uint256 amount1, uint256 tokenId) internal view returns (uint256) {
        // PLACEHOLDER: Returns $1 USD regardless of actual amounts
        // TODO: Integrate price oracle for real USD conversion
        return 1e18;
    }

    /**
     * @notice PLACEHOLDER: Close position and retrieve liquidity
     * @dev CRITICAL (RISK-001): This function returns (0, 0) without removing liquidity.
     *      Production implementation requires:
     *      1. Call decreaseLiquidity on position manager
     *      2. Collect all tokens from position
     *      3. Handle edge cases (single-sided positions, dust amounts)
     *      4. Proper slippage protection
     * @param tokenId The position to close (currently unused)
     * @return amount0 Always returns 0 (placeholder)
     * @return amount1 Always returns 0 (placeholder)
     */
    function _closePosition(uint256 tokenId) internal returns (uint256 amount0, uint256 amount1) {
        // PLACEHOLDER: Returns (0, 0) without actually closing position
        // TODO: Implement proper liquidity removal
        return (0, 0);
    }
    
    function _calculateConcentratedRange(
        int24 currentTick,
        uint256 concentrationLevel,
        uint256 tokenId
    ) internal view returns (int24 tickLower, int24 tickUpper) {
        // Get tick spacing from position
        (,, address token0, address token1, uint24 fee,,,,,,,) = nonfungiblePositionManager.positions(tokenId);
        address factoryAddress = IPeripheryImmutableState(address(nonfungiblePositionManager)).factory();
        IUniswapV3Pool pool = IUniswapV3Pool(
            IUniswapV3Factory(factoryAddress).getPool(token0, token1, fee)
        );
        int24 tickSpacing = pool.tickSpacing();
        
        // Calculate range based on concentration level (1-10 scale)
        int24 halfRange = int24(tickSpacing * (11 - int256(concentrationLevel)));
        
        tickLower = ((currentTick - halfRange) / tickSpacing) * tickSpacing;
        tickUpper = ((currentTick + halfRange) / tickSpacing) * tickSpacing;
        
        // Ensure minimum range
        if (tickUpper - tickLower < tickSpacing) {
            tickUpper = tickLower + tickSpacing;
        }
    }
    
    /**
     * @notice PLACEHOLDER: Open new position with given range and amounts
     * @dev CRITICAL (RISK-001): This function returns oldTokenId without creating position.
     *      Production implementation requires:
     *      1. Approve tokens to position manager
     *      2. Call mint on position manager with new tick range
     *      3. Handle leftover tokens from optimal swap
     *      4. Update internal tracking to new token ID
     *      5. Transfer ownership of old position to owner
     * @param oldTokenId The previous position ID (returned as placeholder)
     * @param tickLower New lower tick (currently unused)
     * @param tickUpper New upper tick (currently unused)
     * @param amount0 Amount of token0 to deposit (currently unused)
     * @param amount1 Amount of token1 to deposit (currently unused)
     * @return newTokenId Always returns oldTokenId (placeholder)
     */
    function _openNewPosition(
        uint256 oldTokenId,
        int24 tickLower,
        int24 tickUpper,
        uint256 amount0,
        uint256 amount1
    ) internal returns (uint256 newTokenId) {
        // PLACEHOLDER: Returns oldTokenId without creating new position
        // TODO: Implement proper position minting
        return oldTokenId;
    }
    
    function _removeFromAccountTokens(address account, uint256 tokenId) internal {
        uint256[] storage tokens = accountTokens[account];
        for (uint256 i = 0; i < tokens.length; i++) {
            if (tokens[i] == tokenId) {
                tokens[i] = tokens[tokens.length - 1];
                tokens.pop();
                break;
            }
        }
    }
    
    // ============ VIEW FUNCTIONS ============
    
    /**
     * @notice Get automation settings for a position
     * @param tokenId The position to query
     * @return settings Current automation settings
     */
    function getAutomationSettings(uint256 tokenId) external view returns (AutomationSettings memory) {
        return positionAutomation[tokenId];
    }
    
    /**
     * @notice Get position performance metrics
     * @param tokenId The position to query
     * @return compoundCount_ Number of compounds executed
     * @return rebalanceCount_ Number of rebalances executed
     * @return totalFeesCompounded_ Total fees compounded in USD
     */
    function getPositionMetrics(uint256 tokenId) external view returns (
        uint256 compoundCount_,
        uint256 rebalanceCount_,
        uint256 totalFeesCompounded_
    ) {
        return (
            compoundCount[tokenId],
            rebalanceCount[tokenId],
            totalFeesCompounded[tokenId]
        );
    }
    
    /**
     * @notice Get positions owned by an account
     * @param account The account to query
     * @return tokenIds Array of token IDs owned by account
     */
    function getAccountPositions(address account) external view returns (uint256[] memory) {
        return accountTokens[account];
    }
    
    // ============ REQUIRED OVERRIDES ============
    
    function onERC721Received(
        address,
        address,
        uint256,
        bytes calldata
    ) external pure override returns (bytes4) {
        return IERC721Receiver.onERC721Received.selector;
    }
}