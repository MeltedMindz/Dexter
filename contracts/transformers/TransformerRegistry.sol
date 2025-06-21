// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./ITransformer.sol";

/// @title TransformerRegistry
/// @notice Registry for managing approved position transformers
/// @dev Allows modular addition of new transformation capabilities
contract TransformerRegistry is ITransformerRegistry, Ownable, ReentrancyGuard {
    
    // Mapping from transform type to transformer address
    mapping(string => address) public transformers;
    
    // Mapping to check if address is a registered transformer
    mapping(address => bool) public isRegistered;
    
    // Array of all registered types (for enumeration)
    string[] public registeredTypes;
    
    // Mapping to track type index in array (for O(1) removal)
    mapping(string => uint256) private typeIndex;
    
    error TransformerAlreadyRegistered();
    error TransformerNotRegistered();
    error InvalidTransformer();
    error TypeAlreadyExists();
    
    /// @notice Register a new transformer
    /// @param transformer Address of transformer contract
    /// @param transformType Type identifier for the transformer
    function registerTransformer(address transformer, string calldata transformType) 
        external 
        override 
        onlyOwner 
        nonReentrant 
    {
        if (transformer == address(0)) {
            revert InvalidTransformer();
        }
        
        if (transformers[transformType] != address(0)) {
            revert TypeAlreadyExists();
        }
        
        if (isRegistered[transformer]) {
            revert TransformerAlreadyRegistered();
        }
        
        // Validate that contract implements ITransformer
        try ITransformer(transformer).getTransformType() returns (string memory returnedType) {
            // Verify the returned type matches
            if (keccak256(bytes(returnedType)) != keccak256(bytes(transformType))) {
                revert InvalidTransformer();
            }
        } catch {
            revert InvalidTransformer();
        }
        
        // Register the transformer
        transformers[transformType] = transformer;
        isRegistered[transformer] = true;
        
        // Add to enumeration
        typeIndex[transformType] = registeredTypes.length;
        registeredTypes.push(transformType);
        
        emit TransformerRegistered(transformer, transformType);
    }
    
    /// @notice Unregister a transformer
    /// @param transformType Type identifier to remove
    function unregisterTransformer(string calldata transformType) 
        external 
        override 
        onlyOwner 
        nonReentrant 
    {
        address transformer = transformers[transformType];
        if (transformer == address(0)) {
            revert TransformerNotRegistered();
        }
        
        // Remove from mappings
        delete transformers[transformType];
        isRegistered[transformer] = false;
        
        // Remove from enumeration array
        uint256 indexToRemove = typeIndex[transformType];
        uint256 lastIndex = registeredTypes.length - 1;
        
        if (indexToRemove != lastIndex) {
            string memory lastType = registeredTypes[lastIndex];
            registeredTypes[indexToRemove] = lastType;
            typeIndex[lastType] = indexToRemove;
        }
        
        registeredTypes.pop();
        delete typeIndex[transformType];
        
        emit TransformerUnregistered(transformType);
    }
    
    /// @notice Get transformer by type
    /// @param transformType Type identifier
    /// @return transformer Address of transformer contract
    function getTransformer(string calldata transformType) 
        external 
        view 
        override 
        returns (address transformer) 
    {
        return transformers[transformType];
    }
    
    /// @notice Check if transformer is registered
    /// @param transformer Address to check
    /// @return isRegisteredTransformer Whether transformer is registered
    function isRegisteredTransformer(address transformer) 
        external 
        view 
        override 
        returns (bool isRegisteredTransformer) 
    {
        return isRegistered[transformer];
    }
    
    /// @notice Get all registered transformer types
    /// @return types Array of registered type identifiers
    function getRegisteredTypes() 
        external 
        view 
        override 
        returns (string[] memory types) 
    {
        return registeredTypes;
    }
    
    /// @notice Get total number of registered transformers
    /// @return count Number of registered transformers
    function getTransformerCount() external view returns (uint256 count) {
        return registeredTypes.length;
    }
    
    /// @notice Get transformer info by index
    /// @param index Index in the registered types array
    /// @return transformType The type identifier
    /// @return transformer The transformer address
    function getTransformerByIndex(uint256 index) 
        external 
        view 
        returns (string memory transformType, address transformer) 
    {
        require(index < registeredTypes.length, "Index out of bounds");
        transformType = registeredTypes[index];
        transformer = transformers[transformType];
    }
    
    /// @notice Batch register multiple transformers
    /// @param transformerAddresses Array of transformer addresses
    /// @param transformTypes Array of type identifiers
    function batchRegisterTransformers(
        address[] calldata transformerAddresses,
        string[] calldata transformTypes
    ) external onlyOwner {
        require(
            transformerAddresses.length == transformTypes.length,
            "Array length mismatch"
        );
        
        for (uint256 i = 0; i < transformerAddresses.length; i++) {
            registerTransformer(transformerAddresses[i], transformTypes[i]);
        }
    }
}