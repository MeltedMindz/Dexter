// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Multicall.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

import "./DexterCompoundor.sol";

/// @title DexterMultiCompoundor
/// @notice Batch operations for multiple positions with gas optimization
/// @dev Based on Revert Finance's MultiCompoundor pattern
contract DexterMultiCompoundor is ReentrancyGuard, Ownable, Multicall {
    using SafeERC20 for IERC20;

    DexterCompoundor public immutable compoundor;
    
    // Maximum positions per batch to prevent gas limit issues
    uint256 public constant MAX_BATCH_SIZE = 50;
    
    // Batch compound parameters
    struct BatchCompoundParams {
        uint256[] tokenIds;
        DexterCompoundor.RewardConversion[] rewardConversions;
        bool[] withdrawRewards;
        bool[] doSwaps;
        bool[] useAIOptimizations;
        uint256 maxGasPerPosition; // Gas limit per position
        bool stopOnFailure; // Whether to stop batch on first failure
    }
    
    // Batch result for each position
    struct CompoundResult {
        uint256 tokenId;
        bool success;
        uint256 reward0;
        uint256 reward1;
        uint256 compounded0;
        uint256 compounded1;
        string errorMessage;
        uint256 gasUsed;
    }
    
    // Events
    event BatchCompoundCompleted(
        address indexed user,
        uint256 totalPositions,
        uint256 successfulCompounds,
        uint256 totalGasUsed
    );
    
    event PositionCompoundFailed(
        uint256 indexed tokenId,
        string reason,
        uint256 gasUsed
    );
    
    error BatchTooLarge();
    error InvalidBatchParams();
    error UnauthorizedCompound();
    
    constructor(DexterCompoundor _compoundor) {
        compoundor = _compoundor;
    }
    
    /// @notice Compound multiple positions in a single transaction
    /// @param params Batch compound parameters
    /// @return results Array of compound results for each position
    function batchCompound(BatchCompoundParams calldata params)
        external
        nonReentrant
        returns (CompoundResult[] memory results)
    {
        uint256 positionCount = params.tokenIds.length;
        
        // Validate batch size
        if (positionCount > MAX_BATCH_SIZE) {
            revert BatchTooLarge();
        }
        
        // Validate parameter arrays match
        if (positionCount != params.rewardConversions.length ||
            positionCount != params.withdrawRewards.length ||
            positionCount != params.doSwaps.length ||
            positionCount != params.useAIOptimizations.length) {
            revert InvalidBatchParams();
        }
        
        results = new CompoundResult[](positionCount);
        uint256 successfulCompounds = 0;
        uint256 totalGasUsed = 0;
        
        for (uint256 i = 0; i < positionCount; i++) {
            uint256 gasStart = gasleft();
            
            // Check if user owns the position
            if (compoundor.ownerOf(params.tokenIds[i]) != msg.sender) {
                results[i] = CompoundResult({
                    tokenId: params.tokenIds[i],
                    success: false,
                    reward0: 0,
                    reward1: 0,
                    compounded0: 0,
                    compounded1: 0,
                    errorMessage: "Not token owner",
                    gasUsed: 0
                });
                continue;
            }
            
            try this._compoundSingle(
                params.tokenIds[i],
                params.rewardConversions[i],
                params.withdrawRewards[i],
                params.doSwaps[i],
                params.useAIOptimizations[i],
                params.maxGasPerPosition
            ) returns (uint256 reward0, uint256 reward1, uint256 compounded0, uint256 compounded1) {
                uint256 gasUsed = gasStart - gasleft();
                totalGasUsed += gasUsed;
                
                results[i] = CompoundResult({
                    tokenId: params.tokenIds[i],
                    success: true,
                    reward0: reward0,
                    reward1: reward1,
                    compounded0: compounded0,
                    compounded1: compounded1,
                    errorMessage: "",
                    gasUsed: gasUsed
                });
                
                successfulCompounds++;
                
            } catch Error(string memory reason) {
                uint256 gasUsed = gasStart - gasleft();
                totalGasUsed += gasUsed;
                
                results[i] = CompoundResult({
                    tokenId: params.tokenIds[i],
                    success: false,
                    reward0: 0,
                    reward1: 0,
                    compounded0: 0,
                    compounded1: 0,
                    errorMessage: reason,
                    gasUsed: gasUsed
                });
                
                emit PositionCompoundFailed(params.tokenIds[i], reason, gasUsed);
                
                if (params.stopOnFailure) {
                    break;
                }
                
            } catch (bytes memory lowLevelData) {
                uint256 gasUsed = gasStart - gasleft();
                totalGasUsed += gasUsed;
                
                results[i] = CompoundResult({
                    tokenId: params.tokenIds[i],
                    success: false,
                    reward0: 0,
                    reward1: 0,
                    compounded0: 0,
                    compounded1: 0,
                    errorMessage: "Low level error",
                    gasUsed: gasUsed
                });
                
                emit PositionCompoundFailed(params.tokenIds[i], "Low level error", gasUsed);
                
                if (params.stopOnFailure) {
                    break;
                }
            }
            
            // Check gas limit per position
            if (params.maxGasPerPosition > 0 && 
                (gasStart - gasleft()) > params.maxGasPerPosition) {
                break;
            }
        }
        
        emit BatchCompoundCompleted(msg.sender, positionCount, successfulCompounds, totalGasUsed);
    }
    
    /// @notice Internal function to compound a single position
    /// @dev This function is called externally to enable try/catch
    function _compoundSingle(
        uint256 tokenId,
        DexterCompoundor.RewardConversion rewardConversion,
        bool withdrawReward,
        bool doSwap,
        bool useAIOptimization,
        uint256 maxGas
    ) external returns (uint256 reward0, uint256 reward1, uint256 compounded0, uint256 compounded1) {
        // Only allow calls from this contract
        require(msg.sender == address(this), "Only self calls allowed");
        
        uint256 gasStart = gasleft();
        
        DexterCompoundor.AutoCompoundParams memory compoundParams = DexterCompoundor.AutoCompoundParams({
            tokenId: tokenId,
            rewardConversion: rewardConversion,
            withdrawReward: withdrawReward,
            doSwap: doSwap,
            useAIOptimization: useAIOptimization
        });
        
        (reward0, reward1, compounded0, compounded1) = compoundor.autoCompound(compoundParams);
        
        // Check gas usage doesn't exceed limit
        if (maxGas > 0 && (gasStart - gasleft()) > maxGas) {
            revert("Gas limit exceeded");
        }
    }
    
    /// @notice Get compound opportunities for multiple positions
    /// @param tokenIds Array of token IDs to check
    /// @return opportunities Array indicating which positions have fees to compound
    /// @return estimatedGas Array of estimated gas costs per position
    function getCompoundOpportunities(uint256[] calldata tokenIds)
        external
        view
        returns (bool[] memory opportunities, uint256[] memory estimatedGas)
    {
        opportunities = new bool[](tokenIds.length);
        estimatedGas = new uint256[](tokenIds.length);
        
        for (uint256 i = 0; i < tokenIds.length; i++) {
            try compoundor.nonfungiblePositionManager().positions(tokenIds[i]) returns (
                uint96,
                address,
                address,
                address,
                uint24,
                int24,
                int24,
                uint128 liquidity,
                uint256,
                uint256,
                uint128 tokensOwed0,
                uint128 tokensOwed1
            ) {
                // Position has compound opportunity if it has liquidity and owed tokens
                opportunities[i] = liquidity > 0 && (tokensOwed0 > 0 || tokensOwed1 > 0);
                
                // Estimate gas based on complexity
                if (opportunities[i]) {
                    estimatedGas[i] = _estimateCompoundGas(tokensOwed0, tokensOwed1);
                }
            } catch {
                opportunities[i] = false;
                estimatedGas[i] = 0;
            }
        }
    }
    
    /// @notice Estimate gas cost for compounding a position
    function _estimateCompoundGas(uint128 tokensOwed0, uint128 tokensOwed1) 
        internal 
        pure 
        returns (uint256 estimatedGas) 
    {
        // Base gas for compound operation
        estimatedGas = 150000;
        
        // Additional gas if both tokens need swapping
        if (tokensOwed0 > 0 && tokensOwed1 > 0) {
            estimatedGas += 50000;
        }
        
        // Additional gas for larger amounts (more complex calculations)
        if (tokensOwed0 > 1e18 || tokensOwed1 > 1e18) {
            estimatedGas += 25000;
        }
    }
    
    /// @notice Compound all eligible positions for a user
    /// @param maxPositions Maximum number of positions to compound
    /// @param rewardConversion Default reward conversion method
    /// @param withdrawReward Whether to withdraw rewards
    /// @return results Array of compound results
    function compoundAllEligible(
        uint256 maxPositions,
        DexterCompoundor.RewardConversion rewardConversion,
        bool withdrawReward
    ) external nonReentrant returns (CompoundResult[] memory results) {
        // Get user's positions
        uint256 userBalance = compoundor.balanceOf(msg.sender);
        uint256 positionsToProcess = userBalance > maxPositions ? maxPositions : userBalance;
        
        if (positionsToProcess == 0) {
            return new CompoundResult[](0);
        }
        
        // Get eligible positions
        uint256[] memory eligibleTokens = new uint256[](positionsToProcess);
        bool[] memory opportunities;
        uint256[] memory estimatedGas;
        uint256 eligibleCount = 0;
        
        // Get user's token IDs (this would need to be implemented in DexterCompoundor)
        // For now, we'll assume we can get them
        
        // Create batch parameters
        BatchCompoundParams memory batchParams = BatchCompoundParams({
            tokenIds: eligibleTokens,
            rewardConversions: new DexterCompoundor.RewardConversion[](eligibleCount),
            withdrawRewards: new bool[](eligibleCount),
            doSwaps: new bool[](eligibleCount),
            useAIOptimizations: new bool[](eligibleCount),
            maxGasPerPosition: 200000, // 200k gas per position
            stopOnFailure: false
        });
        
        // Set default parameters for all positions
        for (uint256 i = 0; i < eligibleCount; i++) {
            batchParams.rewardConversions[i] = rewardConversion;
            batchParams.withdrawRewards[i] = withdrawReward;
            batchParams.doSwaps[i] = false; // Default to no swapping
            batchParams.useAIOptimizations[i] = true; // Default to AI optimization
        }
        
        return batchCompound(batchParams);
    }
    
    /// @notice Calculate optimal batch size based on gas limit
    /// @param gasLimit Target gas limit for the batch
    /// @param avgGasPerPosition Average gas per position
    /// @return optimalBatchSize Recommended batch size
    function calculateOptimalBatchSize(uint256 gasLimit, uint256 avgGasPerPosition)
        external
        pure
        returns (uint256 optimalBatchSize)
    {
        if (avgGasPerPosition == 0) {
            return 0;
        }
        
        // Reserve 100k gas for batch overhead
        uint256 availableGas = gasLimit > 100000 ? gasLimit - 100000 : 0;
        optimalBatchSize = availableGas / avgGasPerPosition;
        
        // Cap at maximum batch size
        if (optimalBatchSize > MAX_BATCH_SIZE) {
            optimalBatchSize = MAX_BATCH_SIZE;
        }
    }
    
    /// @notice Emergency function to pause batch operations
    function emergencyPause() external onlyOwner {
        // Implementation would include pausing mechanism
        // For now, this is a placeholder
    }
}