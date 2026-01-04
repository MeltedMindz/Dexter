# Contract Build Issues

## Known Issue: OpenZeppelin Version Conflict

### Problem
The contracts use OpenZeppelin v4 (`security/ReentrancyGuard`), but `@uniswap/v3-periphery@1.4.4` requires OpenZeppelin v3 (`token/ERC721/IERC721Enumerable`).

### Current Status
- ✅ Dependencies installed
- ✅ Contracts moved to `contracts/` directory
- ✅ Hardhat configured
- ⚠️  Compilation fails due to version conflict

### Error
```
Error HH404: File @openzeppelin/contracts/token/ERC721/IERC721Metadata.sol, 
imported from @uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol, not found.
```

### Solutions (in order of preference)

1. **Update Uniswap packages** (Recommended)
   - Check if newer versions of `@uniswap/v3-periphery` support OpenZeppelin v4
   - Update package.json if available

2. **Use Hardhat resolver configuration**
   - Configure Hardhat to resolve OpenZeppelin v3 for Uniswap imports
   - Keep OpenZeppelin v4 for our contracts

3. **Downgrade contracts to OpenZeppelin v3**
   - Update contract imports from `security/ReentrancyGuard` to `utils/ReentrancyGuard`
   - Less ideal but ensures compatibility

### Temporary Workaround
For now, contracts can be compiled individually or using Foundry if available.

### Next Steps
1. Research if `@uniswap/v3-periphery` has a version compatible with OpenZeppelin v4
2. If not, implement Hardhat resolver configuration
3. Update this document when resolved

