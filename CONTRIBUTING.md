# Contributing to Dexter Protocol

Thank you for your interest in contributing to Dexter Protocol! This document provides guidelines and information for contributors.

## 🎯 **Ways to Contribute**

### **🐛 Bug Reports**
- Use GitHub Issues to report bugs
- Provide detailed reproduction steps
- Include system information and error logs
- Label with `bug` tag

### **✨ Feature Requests**
- Propose new features via GitHub Issues
- Describe the problem your feature solves
- Include implementation considerations
- Label with `enhancement` tag

### **📝 Code Contributions**
- Fork the repository
- Create feature branches from `main`
- Follow code style guidelines
- Include comprehensive tests
- Submit pull requests with clear descriptions

### **📖 Documentation**
- Improve technical documentation
- Add code comments and examples
- Update README and guides
- Translate documentation

## 🛠️ **Development Setup**

### **Prerequisites**
```bash
# Required tools
Node.js 18+ and npm
Python 3.9+
Git
Foundry (for smart contracts)
Docker (optional, for full stack)
```

### **1. Fork and Clone**
```bash
git clone https://github.com/YOUR-USERNAME/dexter-protocol.git
cd dexter-protocol
git remote add upstream https://github.com/dexter-protocol/dexter-protocol.git
```

### **2. Environment Setup**
```bash
# Frontend
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local with your API keys

# Backend
cd ../backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Smart Contracts
cd ../contracts
forge install
```

### **3. Running Tests**
```bash
# Frontend tests
cd frontend && npm test

# Backend tests  
cd backend && python -m pytest

# Smart contract tests
cd contracts && forge test

# Full test suite
npm run test:all
```

## 📋 **Code Style Guidelines**

### **TypeScript/JavaScript**
- Use ESLint and Prettier configurations
- Follow Next.js best practices
- Use TypeScript strict mode
- Prefer functional components with hooks

```typescript
// Good
export function ComponentName({ prop }: Props) {
  const [state, setState] = useState<Type>(initialValue)
  
  return (
    <div className="component-styles">
      {/* JSX content */}
    </div>
  )
}

// Avoid
class ComponentName extends React.Component {
  // Class components for new code
}
```

### **Python**
- Follow PEP 8 style guide
- Use type hints for all functions
- Use Black for code formatting
- Use isort for import organization

```python
# Good
def calculate_yield(
    liquidity: Decimal,
    fees_collected: Decimal,
    time_period: int
) -> Decimal:
    """Calculate annualized yield percentage."""
    return (fees_collected / liquidity) * (365 / time_period) * 100

# Use descriptive variable names
pool_liquidity = get_pool_liquidity(pool_address)
```

### **Solidity**
- Follow Solidity style guide
- Use NatSpec comments for all public functions
- Implement comprehensive error handling
- Optimize for gas efficiency

```solidity
// Good
/**
 * @notice Compounds fees for a Uniswap V3 position
 * @param tokenId The NFT token ID of the position
 * @return amount0 The amount of token0 fees collected
 * @return amount1 The amount of token1 fees collected
 */
function compoundFees(uint256 tokenId) 
    external 
    returns (uint256 amount0, uint256 amount1) 
{
    if (tokenId == 0) revert InvalidTokenId();
    // Implementation
}
```

## 🧪 **Testing Requirements**

### **Frontend Tests**
- Unit tests for utility functions
- Component tests with React Testing Library
- Integration tests for API interactions
- E2E tests for critical user flows

### **Backend Tests**
- Unit tests for all business logic
- Integration tests for API endpoints
- Mock external dependencies
- Test error scenarios and edge cases

### **Smart Contract Tests**
- Unit tests for all public functions
- Integration tests for contract interactions
- Gas optimization verification
- Security scenario testing

```solidity
// Example test structure
contract DexterCompoundorTest is Test {
    DexterCompoundor compoundor;
    
    function setUp() public {
        compoundor = new DexterCompoundor();
    }
    
    function testCompoundFees() public {
        // Test implementation
    }
    
    function testCompoundFeesWithInvalidInput() public {
        // Error scenario testing
    }
}
```

## 📝 **Pull Request Process**

### **1. Before Submitting**
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] Documentation is updated
- [ ] Commits are well-structured
- [ ] Branch is up to date with main

### **2. PR Description Template**
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Code comments added
- [ ] README updated if needed
- [ ] API docs updated if needed

## Checklist
- [ ] Self-review completed
- [ ] No console.log statements
- [ ] No hardcoded values
- [ ] Error handling implemented
```

### **3. Review Process**
- Automated checks must pass
- Code review by maintainers
- Security review for smart contracts
- Performance impact assessment

## 🔐 **Security Guidelines**

### **General Security**
- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all user inputs
- Follow principle of least privilege

### **Smart Contract Security**
- Implement reentrancy guards
- Use SafeMath for arithmetic operations
- Add proper access controls
- Consider MEV and front-running protection

### **Frontend Security**
- Sanitize user inputs
- Use HTTPS for all API calls
- Implement proper error handling
- Protect against XSS and CSRF

## 🏷️ **Issue Labels**

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to docs
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `security` - Security-related issue
- `performance` - Performance improvement
- `frontend` - Frontend-related
- `backend` - Backend-related
- `contracts` - Smart contract related

## 🤝 **Community Guidelines**

### **Code of Conduct**
- Be respectful and inclusive
- Focus on technical discussions
- Provide constructive feedback
- Help newcomers learn

### **Communication**
- Use clear, descriptive commit messages
- Respond to feedback promptly
- Ask questions when unclear
- Share knowledge and insights

### **Recognition**
Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Community showcase for innovative features

## 📞 **Getting Help**

### **Development Questions**
- GitHub Discussions for general questions
- GitHub Issues for bug reports
- Discord community for real-time chat

### **Contact Information**
- Technical questions: dev@dexteragent.com
- Security issues: security@dexteragent.com
- General inquiries: hello@dexteragent.com

## 📚 **Resources**

### **Documentation**
- [Architecture Guide](docs/architecture/)
- [API Documentation](docs/api/)
- [Smart Contract Specifications](contracts/README.md)

### **External Resources**
- [Solidity Documentation](https://docs.soliditylang.org/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Uniswap V3 Documentation](https://docs.uniswap.org/protocol/V3/introduction)

---

**Thank you for contributing to Dexter Protocol!** 🚀

Together, we're building the future of AI-powered DeFi infrastructure.