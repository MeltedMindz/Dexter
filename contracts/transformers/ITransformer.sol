// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";

/// @title ITransformer
/// @notice Interface for position transformation operations
/// @dev Based on Revert Finance's transformer pattern for modular position management
interface ITransformer {
    /// @notice Transform a position with specific parameters
    /// @param transformParams Encoded transformation parameters
    /// @return success Whether the transformation succeeded
    /// @return newTokenId New token ID if position was replaced (0 if modified in place)
    /// @return amount0 Token0 amount processed
    /// @return amount1 Token1 amount processed
    function transform(bytes calldata transformParams) 
        external 
        returns (bool success, uint256 newTokenId, uint256 amount0, uint256 amount1);
    
    /// @notice Validate transformation parameters before execution
    /// @param transformParams Encoded transformation parameters
    /// @return isValid Whether parameters are valid
    /// @return errorMessage Error message if invalid
    function validateTransform(bytes calldata transformParams) 
        external 
        view 
        returns (bool isValid, string memory errorMessage);
    
    /// @notice Get transformation type identifier
    /// @return transformType String identifier for this transformer
    function getTransformType() external pure returns (string memory transformType);
    
    /// @notice Check if transformation requires approval
    /// @param transformParams Encoded transformation parameters
    /// @return requiresApproval Whether user approval is needed
    function requiresApproval(bytes calldata transformParams) 
        external 
        view 
        returns (bool requiresApproval);
    
    /// @notice Estimate gas cost for transformation
    /// @param transformParams Encoded transformation parameters
    /// @return estimatedGas Estimated gas cost
    function estimateGas(bytes calldata transformParams) 
        external 
        view 
        returns (uint256 estimatedGas);
}

/// @title ITransformerRegistry
/// @notice Registry for managing approved transformers
interface ITransformerRegistry {
    /// @notice Register a new transformer
    /// @param transformer Address of transformer contract
    /// @param transformType Type identifier for the transformer
    function registerTransformer(address transformer, string calldata transformType) external;
    
    /// @notice Unregister a transformer
    /// @param transformType Type identifier to remove
    function unregisterTransformer(string calldata transformType) external;
    
    /// @notice Get transformer by type
    /// @param transformType Type identifier
    /// @return transformer Address of transformer contract
    function getTransformer(string calldata transformType) external view returns (address transformer);
    
    /// @notice Check if transformer is registered
    /// @param transformer Address to check
    /// @return isRegistered Whether transformer is registered
    function isRegisteredTransformer(address transformer) external view returns (bool isRegistered);
    
    /// @notice Get all registered transformer types
    /// @return types Array of registered type identifiers
    function getRegisteredTypes() external view returns (string[] memory types);
    
    // Events
    event TransformerRegistered(address indexed transformer, string transformType);
    event TransformerUnregistered(string transformType);
}