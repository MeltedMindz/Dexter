// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/extensions/ERC4626.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Multicall.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";

import "../transformers/ITransformer.sol";
import "../transformers/TransformerRegistry.sol";
import "../oracles/PriceAggregator.sol";
import "../governance/EmergencyAdmin.sol";

/// @title DexterVault
/// @notice Lending vault using Uniswap V3 positions as collateral
/// @dev Based on Revert Finance's V3Vault with AI enhancements
contract DexterVault is ERC4626, Ownable, ReentrancyGuard, Multicall {
    using SafeERC20 for IERC20;

    struct CollateralConfig {
        uint256 collateralFactorX32; // Collateral factor (32-bit fixed point)
        uint256 liquidationPenaltyX32; // Liquidation penalty
        uint256 reserveFactorX32; // Reserve factor for protocol
        bool isActive; // Whether this collateral type is active
    }
    
    struct LoanData {
        uint256 debtShares; // Shares of debt
        uint256 collateralTokenId; // NFT token ID used as collateral
        address collateralToken0; // Token0 of the collateral position
        address collateralToken1; // Token1 of the collateral position
        uint256 liquidationPrice; // Price at which position can be liquidated
        uint256 lastUpdateTime; // Last time loan was updated
        bool isAIManaged; // Whether loan is under AI management
    }
    
    struct VaultConfig {
        uint256 reserveProtectionFactorX32; // Reserve protection factor
        uint256 dailyLendIncreaseLimitMin; // Daily increase limit in minutes
        uint256 dailyDebtIncreaseLimitMin; // Daily debt increase limit in minutes
        uint256 maxCollateralCount; // Maximum collateral positions per loan
        uint256 interestRateX96; // Current interest rate (96-bit fixed point)
        bool transformEnabled; // Whether transformations are enabled
    }
    
    // Core state
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    TransformerRegistry public transformerRegistry;
    PriceAggregator public priceAggregator;
    EmergencyAdmin public emergencyAdmin;
    
    // Vault configuration
    VaultConfig public vaultConfig;
    
    // Collateral configurations by token pair hash
    mapping(bytes32 => CollateralConfig) public collateralConfigs;
    
    // Loan data by borrower
    mapping(address => LoanData) public loans;
    
    // NFT token approvals for vault transformations
    mapping(uint256 => address) public transformApprovals;
    
    // Daily limits tracking
    mapping(uint256 => uint256) public dailyLendIncrease; // day => amount
    mapping(uint256 => uint256) public dailyDebtIncrease; // day => amount
    
    // AI management
    address public aiLoanManager; // AI contract for loan optimization
    bool public aiOptimizationEnabled = true;
    
    // Reserve tracking
    uint256 public totalReserves;
    uint256 public lastAccrualTime;
    
    // Constants
    uint256 constant Q32 = 2**32;
    uint256 constant Q96 = 2**96;
    uint256 constant YEAR_SECS = 365 days;
    uint256 constant MAX_COLLATERAL_FACTOR = Q32 * 90 / 100; // 90% max
    uint256 constant MIN_COLLATERAL_FACTOR = Q32 * 10 / 100; // 10% min
    
    // Events
    event CollateralConfigured(
        bytes32 indexed pairHash,
        uint256 collateralFactor,
        uint256 liquidationPenalty,
        uint256 reserveFactor
    );
    
    event LoanCreated(
        address indexed borrower,
        uint256 indexed tokenId,
        uint256 debtAmount,
        uint256 collateralValue
    );
    
    event LoanRepaid(
        address indexed borrower,
        uint256 repayAmount,
        uint256 remainingDebt
    );
    
    event Liquidation(
        address indexed borrower,
        address indexed liquidator,
        uint256 indexed tokenId,
        uint256 debtAmount,
        uint256 liquidationBonus
    );
    
    event TransformApproval(
        uint256 indexed tokenId,
        address indexed transformer,
        bool approved
    );
    
    event AILoanOptimization(
        address indexed borrower,
        uint256 oldDebt,
        uint256 newDebt,
        string action
    );
    
    error InsufficientCollateral();
    error InvalidCollateralFactor();
    error LoanNotFound();
    error Unauthorized();
    error ExceedsCollateralLimit();
    error DailyLimitExceeded();
    error LiquidationNotAllowed();
    error TransformNotApproved();
    
    constructor(
        IERC20 asset,
        string memory name,
        string memory symbol,
        INonfungiblePositionManager _nonfungiblePositionManager,
        TransformerRegistry _transformerRegistry,
        PriceAggregator _priceAggregator
    ) ERC4626(asset) ERC20(name, symbol) {
        nonfungiblePositionManager = _nonfungiblePositionManager;
        transformerRegistry = _transformerRegistry;
        priceAggregator = _priceAggregator;
        
        // Initialize vault config with safe defaults
        vaultConfig = VaultConfig({
            reserveProtectionFactorX32: Q32 * 5 / 100, // 5% reserve protection
            dailyLendIncreaseLimitMin: 1000, // 1000 minute limit
            dailyDebtIncreaseLimitMin: 1000,
            maxCollateralCount: 10, // Max 10 positions per loan
            interestRateX96: Q96 * 5 / 100, // 5% annual interest
            transformEnabled: true
        });
        
        lastAccrualTime = block.timestamp;
    }
    
    /// @notice Configure collateral parameters for a token pair
    /// @param token0 First token of the pair
    /// @param token1 Second token of the pair
    /// @param collateralFactorX32 Collateral factor (0-90%)
    /// @param liquidationPenaltyX32 Liquidation penalty
    /// @param reserveFactorX32 Reserve factor for protocol
    function configureCollateral(
        address token0,
        address token1,
        uint256 collateralFactorX32,
        uint256 liquidationPenaltyX32,
        uint256 reserveFactorX32
    ) external onlyOwner {
        if (collateralFactorX32 > MAX_COLLATERAL_FACTOR || 
            collateralFactorX32 < MIN_COLLATERAL_FACTOR) {
            revert InvalidCollateralFactor();
        }
        
        bytes32 pairHash = keccak256(abi.encodePacked(token0, token1));
        
        collateralConfigs[pairHash] = CollateralConfig({
            collateralFactorX32: collateralFactorX32,
            liquidationPenaltyX32: liquidationPenaltyX32,
            reserveFactorX32: reserveFactorX32,
            isActive: true
        });
        
        emit CollateralConfigured(
            pairHash,
            collateralFactorX32,
            liquidationPenaltyX32,
            reserveFactorX32
        );
    }
    
    /// @notice Create a loan using Uniswap V3 position as collateral
    /// @param tokenId NFT token ID to use as collateral
    /// @param borrowAmount Amount to borrow
    /// @param enableAIManagement Whether to enable AI management for this loan
    function createLoan(
        uint256 tokenId,
        uint256 borrowAmount,
        bool enableAIManagement
    ) external nonReentrant {
        // Accrue interest before creating loan
        _accrueInterest();
        
        // Check daily limits
        uint256 today = block.timestamp / 1 days;
        if (dailyDebtIncrease[today] + borrowAmount > _getDailyDebtLimit()) {
            revert DailyLimitExceeded();
        }
        
        // Get position info
        (, , address token0, address token1, , , , uint128 liquidity, , , ,) = 
            nonfungiblePositionManager.positions(tokenId);
            
        require(liquidity > 0, "Position has no liquidity");
        
        // Check collateral configuration
        bytes32 pairHash = keccak256(abi.encodePacked(token0, token1));
        CollateralConfig memory collateralConfig = collateralConfigs[pairHash];
        require(collateralConfig.isActive, "Collateral not supported");
        
        // Get collateral value
        uint256 collateralValue = _getPositionValue(tokenId, token0, token1);
        uint256 maxBorrow = (collateralValue * collateralConfig.collateralFactorX32) / Q32;
        
        if (borrowAmount > maxBorrow) {
            revert InsufficientCollateral();
        }
        
        // Check existing loan
        LoanData storage loan = loans[msg.sender];
        require(loan.collateralTokenId == 0, "Loan already exists");
        
        // Transfer NFT to vault
        nonfungiblePositionManager.safeTransferFrom(msg.sender, address(this), tokenId);
        
        // Calculate debt shares
        uint256 debtShares = _convertToShares(borrowAmount, totalSupply(), totalAssets());
        
        // Create loan
        loan.debtShares = debtShares;
        loan.collateralTokenId = tokenId;
        loan.collateralToken0 = token0;
        loan.collateralToken1 = token1;
        loan.liquidationPrice = _calculateLiquidationPrice(collateralValue, borrowAmount, collateralConfig);
        loan.lastUpdateTime = block.timestamp;
        loan.isAIManaged = enableAIManagement && aiOptimizationEnabled;
        
        // Update daily limits
        dailyDebtIncrease[today] += borrowAmount;
        
        // Mint debt tokens to borrower
        _mint(msg.sender, debtShares);
        
        // Transfer borrowed assets
        IERC20(asset()).safeTransfer(msg.sender, borrowAmount);
        
        emit LoanCreated(msg.sender, tokenId, borrowAmount, collateralValue);
    }
    
    /// @notice Repay loan and optionally withdraw collateral
    /// @param repayAmount Amount to repay (0 = pay all)
    /// @param withdrawCollateral Whether to withdraw collateral after repay
    function repayLoan(uint256 repayAmount, bool withdrawCollateral) external nonReentrant {
        _accrueInterest();
        
        LoanData storage loan = loans[msg.sender];
        if (loan.collateralTokenId == 0) {
            revert LoanNotFound();
        }
        
        uint256 currentDebt = _convertToAssets(loan.debtShares, totalSupply(), totalAssets());
        
        // If repayAmount is 0, repay full debt
        if (repayAmount == 0) {
            repayAmount = currentDebt;
        }
        
        require(repayAmount <= currentDebt, "Repay amount exceeds debt");
        
        // Calculate shares to burn
        uint256 sharesToBurn = _convertToShares(repayAmount, totalSupply(), totalAssets());
        
        // Transfer repayment from user
        IERC20(asset()).safeTransferFrom(msg.sender, address(this), repayAmount);
        
        // Burn debt shares
        _burn(msg.sender, sharesToBurn);
        
        // Update loan
        loan.debtShares -= sharesToBurn;
        loan.lastUpdateTime = block.timestamp;
        
        uint256 remainingDebt = loan.debtShares > 0 ? 
            _convertToAssets(loan.debtShares, totalSupply(), totalAssets()) : 0;
        
        // If fully repaid and withdrawal requested, return collateral
        if (remainingDebt == 0 && withdrawCollateral) {
            uint256 tokenId = loan.collateralTokenId;
            
            // Clear loan data
            delete loans[msg.sender];
            
            // Return NFT
            nonfungiblePositionManager.safeTransferFrom(address(this), msg.sender, tokenId);
        }
        
        emit LoanRepaid(msg.sender, repayAmount, remainingDebt);
    }
    
    /// @notice Liquidate an undercollateralized loan
    /// @param borrower Address of borrower to liquidate
    /// @param maxRepayAmount Maximum amount liquidator is willing to repay
    function liquidate(address borrower, uint256 maxRepayAmount) external nonReentrant {
        _accrueInterest();
        
        LoanData storage loan = loans[borrower];
        if (loan.collateralTokenId == 0) {
            revert LoanNotFound();
        }
        
        // Check if liquidation is allowed
        if (!_isLiquidationAllowed(borrower)) {
            revert LiquidationNotAllowed();
        }
        
        uint256 currentDebt = _convertToAssets(loan.debtShares, totalSupply(), totalAssets());
        uint256 repayAmount = maxRepayAmount < currentDebt ? maxRepayAmount : currentDebt;
        
        // Get collateral config
        bytes32 pairHash = keccak256(abi.encodePacked(loan.collateralToken0, loan.collateralToken1));
        CollateralConfig memory collateralConfig = collateralConfigs[pairHash];
        
        // Calculate liquidation bonus
        uint256 liquidationBonus = (repayAmount * collateralConfig.liquidationPenaltyX32) / Q32;
        
        // Transfer repayment from liquidator
        IERC20(asset()).safeTransferFrom(msg.sender, address(this), repayAmount);
        
        // Calculate shares to burn
        uint256 sharesToBurn = _convertToShares(repayAmount, totalSupply(), totalAssets());
        
        // Burn debt shares
        _burn(borrower, sharesToBurn);
        
        // Update loan
        loan.debtShares -= sharesToBurn;
        loan.lastUpdateTime = block.timestamp;
        
        // If fully liquidated, transfer collateral to liquidator
        if (loan.debtShares == 0) {
            uint256 tokenId = loan.collateralTokenId;
            delete loans[borrower];
            nonfungiblePositionManager.safeTransferFrom(address(this), msg.sender, tokenId);
        }
        
        // Transfer liquidation bonus to liquidator
        if (liquidationBonus > 0) {
            IERC20(asset()).safeTransfer(msg.sender, liquidationBonus);
        }
        
        emit Liquidation(borrower, msg.sender, loan.collateralTokenId, repayAmount, liquidationBonus);
    }
    
    /// @notice Approve transformer for position management
    /// @param tokenId NFT token ID
    /// @param transformer Transformer contract address
    /// @param approved Whether to approve or revoke
    function approveTransformer(uint256 tokenId, address transformer, bool approved) external {
        LoanData storage loan = loans[msg.sender];
        require(loan.collateralTokenId == tokenId, "Not your collateral");
        require(transformerRegistry.isRegisteredTransformer(transformer), "Invalid transformer");
        
        if (approved) {
            transformApprovals[tokenId] = transformer;
        } else {
            delete transformApprovals[tokenId];
        }
        
        emit TransformApproval(tokenId, transformer, approved);
    }
    
    /// @notice Execute transformation on collateral position
    /// @param tokenId NFT token ID
    /// @param transformType Type of transformation
    /// @param transformData Encoded transformation parameters
    function executeTransform(
        uint256 tokenId,
        string calldata transformType,
        bytes calldata transformData
    ) external nonReentrant {
        // Check authorization
        address transformer = transformerRegistry.getTransformer(transformType);
        require(transformer != address(0), "Transformer not found");
        require(transformApprovals[tokenId] == transformer, "Transform not approved");
        
        // Execute transformation
        ITransformer(transformer).transform(transformData);
        
        // Update loan liquidation price after transformation
        address borrower = _getTokenOwner(tokenId);
        if (borrower != address(0)) {
            _updateLiquidationPrice(borrower);
        }
    }
    
    /// @notice AI-driven loan optimization
    /// @param borrower Borrower address
    /// @param action Optimization action
    /// @param params Action parameters
    function optimizeLoan(
        address borrower,
        string calldata action,
        bytes calldata params
    ) external {
        require(msg.sender == aiLoanManager, "Only AI manager");
        require(aiOptimizationEnabled, "AI optimization disabled");
        
        LoanData storage loan = loans[borrower];
        require(loan.isAIManaged, "Loan not AI managed");
        
        uint256 oldDebt = _convertToAssets(loan.debtShares, totalSupply(), totalAssets());
        
        // Execute AI optimization logic here
        // This would integrate with the AI system for loan management
        
        uint256 newDebt = _convertToAssets(loan.debtShares, totalSupply(), totalAssets());
        
        emit AILoanOptimization(borrower, oldDebt, newDebt, action);
    }
    
    /// @notice Accrue interest on all loans
    function _accrueInterest() internal {
        uint256 timeDelta = block.timestamp - lastAccrualTime;
        if (timeDelta == 0) return;
        
        uint256 interestRate = vaultConfig.interestRateX96;
        uint256 interestAccrued = (totalAssets() * interestRate * timeDelta) / (Q96 * YEAR_SECS);
        
        // Add to reserves
        uint256 reserveIncrease = (interestAccrued * vaultConfig.reserveProtectionFactorX32) / Q32;
        totalReserves += reserveIncrease;
        
        lastAccrualTime = block.timestamp;
    }
    
    /// @notice Get position value in vault asset terms
    function _getPositionValue(uint256 tokenId, address token0, address token1) internal view returns (uint256) {
        // Get position liquidity and fee data
        (, , , , , , , uint128 liquidity, , , uint128 tokensOwed0, uint128 tokensOwed1) = 
            nonfungiblePositionManager.positions(tokenId);
            
        if (liquidity == 0) return 0;
        
        // Get current prices from price aggregator
        (uint256 price0, uint256 confidence0, bool valid0) = priceAggregator.getValidatedPrice(token0, address(asset()));
        (uint256 price1, uint256 confidence1, bool valid1) = priceAggregator.getValidatedPrice(token1, address(asset()));
        
        require(valid0 && valid1 && confidence0 >= 60 && confidence1 >= 60, "Invalid price data");
        
        // Calculate total value (simplified - would need proper liquidity calculation)
        uint256 value0 = (uint256(liquidity) * price0) / 1e18; // Simplified
        uint256 value1 = (uint256(liquidity) * price1) / 1e18; // Simplified
        uint256 feeValue0 = (uint256(tokensOwed0) * price0) / 1e18;
        uint256 feeValue1 = (uint256(tokensOwed1) * price1) / 1e18;
        
        return value0 + value1 + feeValue0 + feeValue1;
    }
    
    /// @notice Calculate liquidation price for a loan
    function _calculateLiquidationPrice(
        uint256 collateralValue,
        uint256 debtAmount,
        CollateralConfig memory config
    ) internal pure returns (uint256) {
        // Liquidation occurs when debt exceeds collateral * collateral factor
        return (debtAmount * Q32) / config.collateralFactorX32;
    }
    
    /// @notice Check if loan can be liquidated
    function _isLiquidationAllowed(address borrower) internal view returns (bool) {
        LoanData storage loan = loans[borrower];
        if (loan.collateralTokenId == 0) return false;
        
        uint256 currentDebt = _convertToAssets(loan.debtShares, totalSupply(), totalAssets());
        uint256 collateralValue = _getPositionValue(
            loan.collateralTokenId,
            loan.collateralToken0,
            loan.collateralToken1
        );
        
        bytes32 pairHash = keccak256(abi.encodePacked(loan.collateralToken0, loan.collateralToken1));
        CollateralConfig memory config = collateralConfigs[pairHash];
        
        uint256 maxDebt = (collateralValue * config.collateralFactorX32) / Q32;
        
        return currentDebt > maxDebt;
    }
    
    /// @notice Update liquidation price after collateral changes
    function _updateLiquidationPrice(address borrower) internal {
        LoanData storage loan = loans[borrower];
        if (loan.collateralTokenId == 0) return;
        
        uint256 collateralValue = _getPositionValue(
            loan.collateralTokenId,
            loan.collateralToken0,
            loan.collateralToken1
        );
        
        uint256 currentDebt = _convertToAssets(loan.debtShares, totalSupply(), totalAssets());
        
        bytes32 pairHash = keccak256(abi.encodePacked(loan.collateralToken0, loan.collateralToken1));
        CollateralConfig memory config = collateralConfigs[pairHash];
        
        loan.liquidationPrice = _calculateLiquidationPrice(collateralValue, currentDebt, config);
    }
    
    /// @notice Get daily debt limit
    function _getDailyDebtLimit() internal view returns (uint256) {
        return totalAssets() / vaultConfig.dailyDebtIncreaseLimitMin;
    }
    
    /// @notice Get token owner by token ID
    function _getTokenOwner(uint256 tokenId) internal view returns (address) {
        // Find borrower by scanning loans (in production, use more efficient mapping)
        // This is simplified for demonstration
        return address(0);
    }
    
    /// @notice Handle NFT deposits
    function onERC721Received(
        address,
        address from,
        uint256 tokenId,
        bytes calldata
    ) external returns (bytes4) {
        require(msg.sender == address(nonfungiblePositionManager), "Invalid NFT");
        return this.onERC721Received.selector;
    }
    
    /// @notice Set AI loan manager
    function setAILoanManager(address _aiLoanManager) external onlyOwner {
        aiLoanManager = _aiLoanManager;
    }
    
    /// @notice Toggle AI optimization
    function toggleAIOptimization(bool _enabled) external onlyOwner {
        aiOptimizationEnabled = _enabled;
    }
    
    /// @notice Withdraw reserves (only owner)
    function withdrawReserves(uint256 amount) external onlyOwner {
        require(amount <= totalReserves, "Insufficient reserves");
        totalReserves -= amount;
        IERC20(asset()).safeTransfer(msg.sender, amount);
    }
}