// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

import "../interfaces/IDEXToken.sol";

/**
 * @title DEXToken
 * @notice The governance and revenue-sharing token of Dexter Protocol
 * @dev ERC20 token with burn functionality and supply cap
 */
contract DEXToken is IDEXToken, ERC20, ERC20Burnable, Ownable, Pausable {
    
    /// @notice Maximum supply cap (100 million tokens)
    uint256 public constant override maxSupply = 100_000_000 * 10**18;
    
    /// @notice Minting permissions
    mapping(address => bool) public minters;
    
    /// @notice Events
    event MinterUpdated(address indexed minter, bool authorized);
    
    modifier onlyMinter() {
        require(minters[msg.sender] || msg.sender == owner(), "Not authorized to mint");
        _;
    }
    
    constructor(
        string memory name,
        string memory symbol
    ) ERC20(name, symbol) {
        // Mint initial supply to deployer for distribution
        // This could be adjusted based on tokenomics design
        _mint(msg.sender, 10_000_000 * 10**18); // 10M initial supply
    }
    
    /**
     * @notice Updates minter authorization
     * @param minter Address to update
     * @param authorized Whether the address can mint
     */
    function setMinter(address minter, bool authorized) external onlyOwner {
        minters[minter] = authorized;
        emit MinterUpdated(minter, authorized);
    }
    
    /**
     * @notice Mints new tokens (restricted to authorized minters)
     * @param to Recipient address
     * @param amount Amount to mint
     */
    function mint(address to, uint256 amount) external override onlyMinter whenNotPaused {
        require(totalSupply() + amount <= maxSupply, "Would exceed max supply");
        _mint(to, amount);
    }
    
    /**
     * @notice Burns tokens from caller's balance
     * @param amount Amount to burn
     */
    function burn(uint256 amount) public override(IDEXToken, ERC20Burnable) {
        super.burn(amount);
    }
    
    /**
     * @notice Returns current circulating supply
     */
    function circulatingSupply() external view override returns (uint256) {
        return totalSupply();
    }
    
    /**
     * @notice Pauses all token transfers (emergency use only)
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @notice Unpauses token transfers
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @notice Hook that is called before any token transfer
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }
}