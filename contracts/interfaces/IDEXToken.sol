// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title IDEXToken
 * @notice Interface for the DEX governance token
 */
interface IDEXToken is IERC20 {
    
    /// @notice Mints tokens to a specified address (restricted)
    function mint(address to, uint256 amount) external;
    
    /// @notice Burns tokens from msg.sender
    function burn(uint256 amount) external;
    
    /// @notice Returns the maximum supply cap
    function maxSupply() external view returns (uint256);
    
    /// @notice Returns the current circulating supply
    function circulatingSupply() external view returns (uint256);
}