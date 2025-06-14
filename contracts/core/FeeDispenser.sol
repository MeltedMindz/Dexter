// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";

import "../interfaces/IFeeDispenser.sol";
import "../interfaces/IDEXStaking.sol";

/**
 * @title FeeDispenser
 * @notice Collects protocol fees, converts to WETH, and distributes to stakers
 * @dev All fees are converted to WETH for consistent distribution
 */
contract FeeDispenser is IFeeDispenser, Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    
    /// @notice Core addresses
    address public immutable override WETH;
    address public override stakingContract;
    ISwapRouter public immutable override swapRouter;
    
    /// @notice Authorized depositors (LiquidityManager contracts)
    mapping(address => bool) public authorizedDepositors;
    
    /// @notice Fee tracking
    mapping(address => uint256) private pendingFeeBalances;
    address[] private tokensWithFees;
    mapping(address => bool) private hasBalance;
    
    /// @notice Distribution configuration
    uint256 public override distributionThreshold = 0.1 ether; // 0.1 WETH
    
    /// @notice Events
    event DepositorAuthorized(address indexed depositor, bool authorized);
    
    modifier onlyAuthorized() {
        require(authorizedDepositors[msg.sender], "Not authorized");
        _;
    }
    
    constructor(
        address _weth,
        address _swapRouter,
        address _stakingContract
    ) {
        WETH = _weth;
        swapRouter = ISwapRouter(_swapRouter);
        stakingContract = _stakingContract;
    }
    
    /**
     * @notice Authorizes or removes a depositor
     */
    function setAuthorizedDepositor(address depositor, bool authorized) external onlyOwner {
        authorizedDepositors[depositor] = authorized;
        emit DepositorAuthorized(depositor, authorized);
    }
    
    /**
     * @notice Updates the staking contract address
     */
    function setStakingContract(address _stakingContract) external onlyOwner {
        require(_stakingContract != address(0), "Invalid address");
        stakingContract = _stakingContract;
    }
    
    /**
     * @notice Deposits fees from liquidity manager
     */
    function depositFees(
        address token,
        uint256 amount
    ) external override onlyAuthorized {
        require(token != address(0), "Invalid token");
        require(amount > 0, "Invalid amount");
        
        // Transfer tokens from sender
        IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        
        // Track the balance
        if (!hasBalance[token]) {
            tokensWithFees.push(token);
            hasBalance[token] = true;
        }
        pendingFeeBalances[token] += amount;
        
        // If it's already WETH, check for distribution
        if (token == WETH) {
            emit FeesDeposited(token, amount, amount);
            _tryDistribute();
        } else {
            emit FeesDeposited(token, amount, 0);
        }
    }
    
    /**
     * @notice Converts accumulated tokens to WETH
     */
    function convertToWETH(
        address[] calldata tokens,
        uint256[] calldata minAmountsOut
    ) external override nonReentrant {
        require(tokens.length == minAmountsOut.length, "Length mismatch");
        
        for (uint256 i = 0; i < tokens.length; i++) {
            address token = tokens[i];
            uint256 balance = pendingFeeBalances[token];
            
            if (balance == 0 || token == WETH) continue;
            
            // Approve swap router
            IERC20(token).safeApprove(address(swapRouter), balance);
            
            // Prepare swap parameters
            ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
                tokenIn: token,
                tokenOut: WETH,
                fee: 3000, // 0.3% pool (could be optimized)
                recipient: address(this),
                deadline: block.timestamp,
                amountIn: balance,
                amountOutMinimum: minAmountsOut[i],
                sqrtPriceLimitX96: 0
            });
            
            // Execute swap
            try swapRouter.exactInputSingle(params) returns (uint256 amountOut) {
                pendingFeeBalances[token] = 0;
                pendingFeeBalances[WETH] += amountOut;
                
                emit FeesConverted(token, balance, amountOut);
            } catch {
                // Revert approval if swap fails
                IERC20(token).safeApprove(address(swapRouter), 0);
            }
        }
        
        _tryDistribute();
    }
    
    /**
     * @notice Distributes WETH to stakers if threshold is met
     */
    function distribute() external override returns (uint256 distributed) {
        return _distribute();
    }
    
    /**
     * @notice Force distribution regardless of threshold
     */
    function forceDistribute() external override onlyOwner returns (uint256 distributed) {
        return _distribute();
    }
    
    /**
     * @notice Internal distribution logic
     */
    function _distribute() private nonReentrant returns (uint256 distributed) {
        distributed = pendingFeeBalances[WETH];
        require(distributed > 0, "No WETH to distribute");
        
        // Clear balance before transfer (reentrancy protection)
        pendingFeeBalances[WETH] = 0;
        
        // Get total staked for event
        uint256 totalStaked = IDEXStaking(stakingContract).totalStaked();
        
        // Transfer WETH to staking contract
        IERC20(WETH).safeTransfer(stakingContract, distributed);
        
        // Notify staking contract
        IDEXStaking(stakingContract).receiveWETHRewards(distributed);
        
        emit FeesDistributed(distributed, totalStaked);
    }
    
    /**
     * @notice Tries to distribute if threshold is met
     */
    function _tryDistribute() private {
        if (pendingFeeBalances[WETH] >= distributionThreshold) {
            _distribute();
        }
    }
    
    /**
     * @notice Updates the distribution threshold
     */
    function updateThreshold(uint256 newThreshold) external override onlyOwner {
        distributionThreshold = newThreshold;
        emit ThresholdUpdated(newThreshold);
    }
    
    /**
     * @notice Gets current WETH balance pending distribution
     */
    function pendingDistribution() external view override returns (uint256) {
        return pendingFeeBalances[WETH];
    }
    
    /**
     * @notice Gets pending fees for a specific token
     */
    function pendingFees(address token) external view override returns (uint256) {
        return pendingFeeBalances[token];
    }
    
    /**
     * @notice Gets all tokens with pending fees
     */
    function getTokensWithFees() external view returns (address[] memory) {
        uint256 count = 0;
        for (uint256 i = 0; i < tokensWithFees.length; i++) {
            if (pendingFeeBalances[tokensWithFees[i]] > 0) {
                count++;
            }
        }
        
        address[] memory result = new address[](count);
        uint256 index = 0;
        for (uint256 i = 0; i < tokensWithFees.length; i++) {
            if (pendingFeeBalances[tokensWithFees[i]] > 0) {
                result[index++] = tokensWithFees[i];
            }
        }
        
        return result;
    }
    
    /**
     * @notice Rescue stuck tokens (emergency only)
     */
    function rescueTokens(
        address token,
        uint256 amount
    ) external override onlyOwner {
        require(token != WETH || amount <= pendingFeeBalances[WETH], "Cannot rescue pending WETH");
        IERC20(token).safeTransfer(owner(), amount);
    }
}