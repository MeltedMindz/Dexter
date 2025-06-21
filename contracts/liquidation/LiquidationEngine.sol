// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Multicall.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/interfaces/IERC3156FlashLender.sol";
import "@openzeppelin/contracts/interfaces/IERC3156FlashBorrower.sol";

import "../lending/DexterVault.sol";
import "../oracles/PriceAggregator.sol";
import "../governance/EmergencyAdmin.sol";

/// @title LiquidationEngine
/// @notice Automated liquidation system with MEV protection and AI optimization
/// @dev Based on Revert Finance's liquidator with flashloan integration
contract LiquidationEngine is Ownable, ReentrancyGuard, Multicall, IERC3156FlashBorrower {
    using SafeERC20 for IERC20;

    struct LiquidationTarget {
        address borrower;
        address vault;
        uint256 maxRepayAmount;
        uint256 minProfitBps; // Minimum profit in basis points
        uint256 deadline;
        bool useFlashloan;
        bytes flashloanData;
    }
    
    struct LiquidationResult {
        bool success;
        uint256 repaidAmount;
        uint256 collateralReceived;
        uint256 profit;
        uint256 gasUsed;
        string errorMessage;
    }
    
    struct LiquidatorConfig {
        uint256 minProfitBps; // Minimum profit required (100 = 1%)
        uint256 maxGasPrice; // Maximum gas price willing to pay
        uint256 maxSlippageBps; // Maximum slippage tolerance
        bool flashloanEnabled; // Whether to use flashloans
        bool aiOptimizationEnabled; // Whether to use AI for optimization
    }
    
    // Core contracts
    PriceAggregator public priceAggregator;
    EmergencyAdmin public emergencyAdmin;
    IERC3156FlashLender public flashLender;
    
    // Liquidator configurations
    mapping(address => LiquidatorConfig) public liquidatorConfigs;
    mapping(address => bool) public authorizedLiquidators;
    
    // Supported vaults for liquidation
    mapping(address => bool) public supportedVaults;
    
    // AI optimization
    address public aiLiquidationOptimizer;
    bool public aiOptimizationEnabled = true;
    
    // MEV protection
    uint256 public maxBlockDelay = 3; // Maximum blocks between liquidation discovery and execution
    mapping(bytes32 => uint256) public liquidationDiscoveryBlock;
    
    // Performance tracking
    mapping(address => uint256) public liquidatorProfits;
    mapping(address => uint256) public liquidatorGasUsed;
    mapping(address => uint256) public successfulLiquidations;
    mapping(address => uint256) public failedLiquidations;
    
    // Flash loan callback data
    bytes32 private constant CALLBACK_SUCCESS = keccak256("ERC3156FlashBorrower.onFlashLoan");
    mapping(bytes32 => bool) private activeFlashLoans;
    
    // Constants
    uint256 constant BPS_SCALE = 10000;
    uint256 constant MAX_SLIPPAGE_BPS = 1000; // 10% max slippage
    uint256 constant MIN_PROFIT_BPS = 50; // 0.5% minimum profit
    
    // Events
    event LiquidationExecuted(
        address indexed liquidator,
        address indexed borrower,
        address indexed vault,
        uint256 repaidAmount,
        uint256 collateralReceived,
        uint256 profit,
        bool usedFlashloan
    );
    
    event LiquidationFailed(
        address indexed liquidator,
        address indexed borrower,
        address indexed vault,
        string reason,
        uint256 gasUsed
    );
    
    event LiquidatorAuthorized(address indexed liquidator, bool authorized);
    event VaultSupported(address indexed vault, bool supported);
    event AIOptimizationUsed(address indexed borrower, uint256 optimizedProfit, string strategy);
    event MEVProtectionTriggered(address indexed borrower, uint256 blockDelay);
    
    error UnauthorizedLiquidator();
    error UnsupportedVault();
    error InsufficientProfit();
    error MEVProtectionActive();
    error InvalidFlashloanCallback();
    error LiquidationFailed();
    error SlippageExceeded();
    
    modifier onlyAuthorizedLiquidator() {
        if (!authorizedLiquidators[msg.sender]) {
            revert UnauthorizedLiquidator();
        }
        _;
    }
    
    modifier supportedVault(address vault) {
        if (!supportedVaults[vault]) {
            revert UnsupportedVault();
        }
        _;
    }
    
    constructor(
        PriceAggregator _priceAggregator,
        IERC3156FlashLender _flashLender
    ) {
        priceAggregator = _priceAggregator;
        flashLender = _flashLender;
        
        // Authorize deployer as initial liquidator
        authorizedLiquidators[msg.sender] = true;
    }
    
    /// @notice Execute liquidation with optional flashloan
    /// @param target Liquidation target parameters
    /// @return result Liquidation execution result
    function executeLiquidation(LiquidationTarget calldata target)
        external
        nonReentrant
        onlyAuthorizedLiquidator
        supportedVault(target.vault)
        returns (LiquidationResult memory result)
    {
        uint256 gasStart = gasleft();
        
        // Check deadline
        require(block.timestamp <= target.deadline, "Liquidation expired");
        
        // MEV protection check
        bytes32 targetHash = keccak256(abi.encodePacked(target.borrower, target.vault));
        uint256 discoveryBlock = liquidationDiscoveryBlock[targetHash];
        if (discoveryBlock > 0 && block.number > discoveryBlock + maxBlockDelay) {
            emit MEVProtectionTriggered(target.borrower, block.number - discoveryBlock);
            revert MEVProtectionActive();
        }
        
        // Get liquidator config
        LiquidatorConfig memory config = liquidatorConfigs[msg.sender];
        
        // Check minimum profit requirement
        if (target.minProfitBps < config.minProfitBps) {
            revert InsufficientProfit();
        }
        
        try this._executeLiquidationInternal(target, config) returns (
            uint256 repaidAmount,
            uint256 collateralReceived,
            uint256 profit
        ) {
            result = LiquidationResult({
                success: true,
                repaidAmount: repaidAmount,
                collateralReceived: collateralReceived,
                profit: profit,
                gasUsed: gasStart - gasleft(),
                errorMessage: ""
            });
            
            // Update statistics
            liquidatorProfits[msg.sender] += profit;
            liquidatorGasUsed[msg.sender] += result.gasUsed;
            successfulLiquidations[msg.sender]++;
            
            emit LiquidationExecuted(
                msg.sender,
                target.borrower,
                target.vault,
                repaidAmount,
                collateralReceived,
                profit,
                target.useFlashloan
            );
            
        } catch Error(string memory reason) {
            result = LiquidationResult({
                success: false,
                repaidAmount: 0,
                collateralReceived: 0,
                profit: 0,
                gasUsed: gasStart - gasleft(),
                errorMessage: reason
            });
            
            failedLiquidations[msg.sender]++;
            
            emit LiquidationFailed(msg.sender, target.borrower, target.vault, reason, result.gasUsed);
        }
    }
    
    /// @notice Internal liquidation execution (enables try/catch)
    function _executeLiquidationInternal(
        LiquidationTarget calldata target,
        LiquidatorConfig memory config
    ) external returns (uint256 repaidAmount, uint256 collateralReceived, uint256 profit) {
        require(msg.sender == address(this), "Internal function");
        
        if (target.useFlashloan && config.flashloanEnabled) {
            // Execute with flashloan
            return _executeFlashloanLiquidation(target, config);
        } else {
            // Execute direct liquidation
            return _executeDirectLiquidation(target, config);
        }
    }
    
    /// @notice Execute liquidation with flashloan
    function _executeFlashloanLiquidation(
        LiquidationTarget calldata target,
        LiquidatorConfig memory config
    ) internal returns (uint256 repaidAmount, uint256 collateralReceived, uint256 profit) {
        
        DexterVault vault = DexterVault(target.vault);
        IERC20 asset = IERC20(vault.asset());
        
        // Calculate flashloan amount needed
        uint256 flashAmount = target.maxRepayAmount;
        
        // Prepare flashloan data
        bytes memory data = abi.encode(target, config);
        bytes32 loanId = keccak256(abi.encodePacked(target.borrower, block.timestamp));
        activeFlashLoans[loanId] = true;
        
        // Execute flashloan
        uint256 fee = flashLender.flashFee(address(asset), flashAmount);
        flashLender.flashLoan(this, address(asset), flashAmount, data);
        
        // Results would be set during flashloan callback
        // This is simplified - in practice, would need proper state management
        activeFlashLoans[loanId] = false;
        
        return (repaidAmount, collateralReceived, profit);
    }
    
    /// @notice Execute direct liquidation (without flashloan)
    function _executeDirectLiquidation(
        LiquidationTarget calldata target,
        LiquidatorConfig memory config
    ) internal returns (uint256 repaidAmount, uint256 collateralReceived, uint256 profit) {
        
        DexterVault vault = DexterVault(target.vault);
        IERC20 asset = IERC20(vault.asset());
        
        // Check liquidator has enough balance
        uint256 liquidatorBalance = asset.balanceOf(msg.sender);
        require(liquidatorBalance >= target.maxRepayAmount, "Insufficient balance");
        
        // Transfer funds from liquidator
        asset.safeTransferFrom(msg.sender, address(this), target.maxRepayAmount);
        
        // Approve vault for repayment
        asset.safeApprove(target.vault, target.maxRepayAmount);
        
        // Execute liquidation
        uint256 balanceBefore = asset.balanceOf(address(this));
        vault.liquidate(target.borrower, target.maxRepayAmount);
        uint256 balanceAfter = asset.balanceOf(address(this));
        
        // Calculate results
        repaidAmount = target.maxRepayAmount;
        collateralReceived = balanceAfter - balanceBefore + target.maxRepayAmount;
        
        if (collateralReceived > repaidAmount) {
            profit = collateralReceived - repaidAmount;
        }
        
        // Check minimum profit
        uint256 requiredProfit = (repaidAmount * target.minProfitBps) / BPS_SCALE;
        if (profit < requiredProfit) {
            revert InsufficientProfit();
        }
        
        // Return funds to liquidator
        asset.safeTransfer(msg.sender, collateralReceived);
        
        // Reset approval
        asset.safeApprove(target.vault, 0);
    }
    
    /// @notice Flash loan callback
    function onFlashLoan(
        address initiator,
        address token,
        uint256 amount,
        uint256 fee,
        bytes calldata data
    ) external override returns (bytes32) {
        // Verify callback is from authorized flash lender
        require(msg.sender == address(flashLender), "Invalid flash lender");
        require(initiator == address(this), "Invalid initiator");
        
        // Decode liquidation data
        (LiquidationTarget memory target, LiquidatorConfig memory config) = 
            abi.decode(data, (LiquidationTarget, LiquidatorConfig));
        
        // Execute liquidation with borrowed funds
        IERC20 asset = IERC20(token);
        DexterVault vault = DexterVault(target.vault);
        
        // Approve vault for liquidation
        asset.safeApprove(target.vault, amount);
        
        // Execute liquidation
        vault.liquidate(target.borrower, target.maxRepayAmount);
        
        // Check we received enough to repay loan + fee + profit
        uint256 totalNeeded = amount + fee;
        uint256 currentBalance = asset.balanceOf(address(this));
        
        require(currentBalance >= totalNeeded, "Insufficient liquidation proceeds");
        
        // Approve repayment
        asset.safeApprove(address(flashLender), totalNeeded);
        
        return CALLBACK_SUCCESS;
    }
    
    /// @notice Discover liquidation opportunity (MEV protection)
    /// @param borrower Borrower address
    /// @param vault Vault address
    function discoverLiquidation(address borrower, address vault) external {
        bytes32 targetHash = keccak256(abi.encodePacked(borrower, vault));
        liquidationDiscoveryBlock[targetHash] = block.number;
    }
    
    /// @notice AI-optimized liquidation strategy
    /// @param targets Array of liquidation targets
    /// @return optimizedTargets Optimized targets with AI recommendations
    function optimizeLiquidationStrategy(LiquidationTarget[] calldata targets)
        external
        view
        returns (LiquidationTarget[] memory optimizedTargets)
    {
        if (!aiOptimizationEnabled || aiLiquidationOptimizer == address(0)) {
            return targets;
        }
        
        // AI optimization logic would go here
        // This would integrate with the AI system to optimize:
        // - Liquidation order
        // - Profit maximization
        // - Gas optimization
        // - MEV protection
        
        optimizedTargets = targets; // Simplified for now
    }
    
    /// @notice Batch liquidation execution
    /// @param targets Array of liquidation targets
    /// @return results Array of liquidation results
    function batchLiquidate(LiquidationTarget[] calldata targets)
        external
        onlyAuthorizedLiquidator
        returns (LiquidationResult[] memory results)
    {
        results = new LiquidationResult[](targets.length);
        
        for (uint256 i = 0; i < targets.length; i++) {
            // Skip if vault not supported
            if (!supportedVaults[targets[i].vault]) {
                results[i].success = false;
                results[i].errorMessage = "Unsupported vault";
                continue;
            }
            
            try this.executeLiquidation(targets[i]) returns (LiquidationResult memory result) {
                results[i] = result;
            } catch Error(string memory reason) {
                results[i].success = false;
                results[i].errorMessage = reason;
            }
        }
    }
    
    /// @notice Check if position is liquidatable
    /// @param borrower Borrower address
    /// @param vault Vault address
    /// @return isLiquidatable Whether position can be liquidated
    /// @return healthFactor Current health factor
    /// @return maxRepayAmount Maximum amount that can be repaid
    function checkLiquidationEligibility(address borrower, address vault)
        external
        view
        returns (bool isLiquidatable, uint256 healthFactor, uint256 maxRepayAmount)
    {
        // This would integrate with the vault to check loan health
        // Simplified implementation
        return (false, 0, 0);
    }
    
    /// @notice Calculate liquidation profit
    /// @param borrower Borrower address
    /// @param vault Vault address
    /// @param repayAmount Amount to repay
    /// @return profit Expected profit from liquidation
    /// @return collateralValue Value of collateral to be received
    function calculateLiquidationProfit(
        address borrower,
        address vault,
        uint256 repayAmount
    ) external view returns (uint256 profit, uint256 collateralValue) {
        // Implementation would calculate based on vault's liquidation logic
        return (0, 0);
    }
    
    /// @notice Set liquidator configuration
    /// @param liquidator Liquidator address
    /// @param config Configuration parameters
    function setLiquidatorConfig(address liquidator, LiquidatorConfig calldata config) external {
        require(msg.sender == liquidator || msg.sender == owner(), "Unauthorized");
        require(config.minProfitBps >= MIN_PROFIT_BPS, "Profit too low");
        require(config.maxSlippageBps <= MAX_SLIPPAGE_BPS, "Slippage too high");
        
        liquidatorConfigs[liquidator] = config;
    }
    
    /// @notice Authorize liquidator
    /// @param liquidator Liquidator address
    /// @param authorized Whether to authorize
    function authorizeLiquidator(address liquidator, bool authorized) external onlyOwner {
        authorizedLiquidators[liquidator] = authorized;
        emit LiquidatorAuthorized(liquidator, authorized);
    }
    
    /// @notice Add supported vault
    /// @param vault Vault address
    /// @param supported Whether vault is supported
    function setSupportedVault(address vault, bool supported) external onlyOwner {
        supportedVaults[vault] = supported;
        emit VaultSupported(vault, supported);
    }
    
    /// @notice Set AI liquidation optimizer
    /// @param _aiLiquidationOptimizer AI optimizer address
    function setAILiquidationOptimizer(address _aiLiquidationOptimizer) external onlyOwner {
        aiLiquidationOptimizer = _aiLiquidationOptimizer;
    }
    
    /// @notice Toggle AI optimization
    /// @param _enabled Whether AI optimization is enabled
    function toggleAIOptimization(bool _enabled) external onlyOwner {
        aiOptimizationEnabled = _enabled;
    }
    
    /// @notice Set MEV protection parameters
    /// @param _maxBlockDelay Maximum block delay for MEV protection
    function setMEVProtection(uint256 _maxBlockDelay) external onlyOwner {
        require(_maxBlockDelay <= 10, "Block delay too high");
        maxBlockDelay = _maxBlockDelay;
    }
    
    /// @notice Get liquidator statistics
    /// @param liquidator Liquidator address
    /// @return totalProfit Total profit earned
    /// @return totalGasUsed Total gas used
    /// @return successCount Number of successful liquidations
    /// @return failureCount Number of failed liquidations
    function getLiquidatorStats(address liquidator)
        external
        view
        returns (
            uint256 totalProfit,
            uint256 totalGasUsed,
            uint256 successCount,
            uint256 failureCount
        )
    {
        return (
            liquidatorProfits[liquidator],
            liquidatorGasUsed[liquidator],
            successfulLiquidations[liquidator],
            failedLiquidations[liquidator]
        );
    }
    
    /// @notice Emergency pause liquidations
    function emergencyPause() external {
        require(
            msg.sender == owner() ||
            (address(emergencyAdmin) != address(0) && 
             emergencyAdmin.hasRole(emergencyAdmin.EMERGENCY_ADMIN_ROLE(), msg.sender)),
            "Unauthorized"
        );
        // Implementation would pause liquidations
    }
    
    /// @notice Withdraw stuck tokens (emergency)
    /// @param token Token address
    /// @param amount Amount to withdraw
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        IERC20(token).safeTransfer(msg.sender, amount);
    }
}