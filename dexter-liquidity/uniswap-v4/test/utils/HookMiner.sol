// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title HookMiner
 * @notice Utility contract for mining hook addresses with specific permissions
 * @dev Used in tests to generate valid hook addresses
 */
library HookMiner {
    
    // Salt generation constants
    uint256 internal constant SALT_MASK = 0x000000000000000000000000000000000000000000000000000000000000FFFF;
    
    /**
     * @notice Find a salt that produces a hook address with the desired permissions
     * @param deployer Address that will deploy the hook
     * @param flags Permission flags required for the hook
     * @param creationCode The creation bytecode of the hook contract
     * @param constructorArgs ABI-encoded constructor arguments
     * @return hookAddress The computed hook address
     * @return salt The salt value that produces the desired address
     */
    function find(
        address deployer,
        uint160 flags,
        bytes memory creationCode,
        bytes memory constructorArgs
    ) internal pure returns (address hookAddress, bytes32 salt) {
        bytes memory bytecode = abi.encodePacked(creationCode, constructorArgs);
        
        for (uint256 i = 0; i < type(uint16).max; i++) {
            salt = bytes32(i);
            
            // Compute CREATE2 address
            hookAddress = computeAddress(deployer, salt, bytecode);
            
            // Check if the address has the required permissions
            if (uint160(hookAddress) & flags == flags) {
                break;
            }
        }
        
        require(uint160(hookAddress) & flags == flags, "HookMiner: Could not find valid address");
    }
    
    /**
     * @notice Compute CREATE2 address
     * @param deployer Address that will deploy the contract
     * @param salt Salt value
     * @param bytecode Complete bytecode including constructor args
     * @return The computed address
     */
    function computeAddress(
        address deployer,
        bytes32 salt,
        bytes memory bytecode
    ) internal pure returns (address) {
        bytes32 hash = keccak256(
            abi.encodePacked(
                bytes1(0xff),
                deployer,
                salt,
                keccak256(bytecode)
            )
        );
        return address(uint160(uint256(hash)));
    }
}