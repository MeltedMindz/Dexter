# Dexter Protocol 🤖

**AI-Powered Liquidity Management for Decentralized Finance**

Dexter Protocol is an advanced DeFi infrastructure platform that leverages artificial intelligence to optimize liquidity provision, automate position management, and maximize yields across decentralized exchanges.

[![License: Source Available](https://img.shields.io/badge/License-Source%20Available-blue.svg)](LICENSE)
[![Contracts: Solidity](https://img.shields.io/badge/Contracts-Solidity-gray)](contracts/)
[![AI: Python](https://img.shields.io/badge/AI-Python-green)](backend/)
[![Website: Next.js](https://img.shields.io/badge/Website-dexter--website-blue)](https://github.com/MeltedMindz/dexter-website)

## 🚀 **Core Features**

### **🧠 AI-Powered Position Management**
- **Machine Learning Models**: LSTM price prediction, volatility analysis, yield optimization
- **Real-Time Strategy**: Dynamic range adjustment based on market conditions
- **Risk Assessment**: Conservative, moderate, and aggressive strategy frameworks
- **Automated Compounding**: Gas-efficient fee collection and reinvestment

### **🏦 ERC4626 Vault Infrastructure**
- **Standard Compliance**: Full ERC4626 vault implementation for institutional integration
- **Multi-Strategy Support**: Manual, AI-assisted, and fully automated vault modes
- **Gamma-Style Positioning**: Dual-position strategies with base + limit ranges
- **Advanced Fee Management**: Tiered structure (Retail/Premium/Institutional/VIP)

### **⚡ Uniswap V4 Integration**
- **Dynamic Hook System**: AI-powered fee optimization (0.01%-100% range)
- **Capital Efficiency**: Real-time position rebalancing and concentration monitoring
- **Emergency Controls**: Volatility-based safety mechanisms and emergency mode
- **Gas Optimization**: Sophisticated batching and execution efficiency

### **🌐 Website Integration**
- **Separate Repository**: [dexter-website](https://github.com/MeltedMindz/dexter-website) for user interface
- **API Integration**: RESTful APIs for frontend communication
- **Real-Time Data**: WebSocket connections for live updates
- **Independent Deployment**: Frontend and backend deploy separately for optimal performance

## 🏗️ **Protocol Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│              Website (dexter-website repo)                 │
│        https://github.com/MeltedMindz/dexter-website        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐   │
│  │  Vault Factory  │ │ Position Manager│ │   AI Chat    │   │
│  └─────────────────┘ └─────────────────┘ └──────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │ API Integration
┌─────────────────────────────┴───────────────────────────────┐
│                 Smart Contracts (Solidity)                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐   │
│  │  ERC4626 Vaults │ │  V4 Hooks       │ │ AI Strategies│   │
│  └─────────────────┘ └─────────────────┘ └──────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                   AI Engine (Python)                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐   │
│  │ DexBrain Hub    │ │ ML Pipeline     │ │  Data Sources │  │
│  └─────────────────┘ └─────────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ **Technology Stack**

### **Website** ([dexter-website](https://github.com/MeltedMindz/dexter-website))
- **Framework**: Next.js 15 with TypeScript
- **Web3 Integration**: Wagmi + Viem for blockchain connectivity
- **UI/UX**: Tailwind CSS with Neo-brutalism design
- **Deployment**: Vercel with automatic deployments

### **Smart Contracts**
- **Language**: Solidity ^0.8.19
- **Standards**: ERC4626 vaults, ERC721 position NFTs
- **Testing**: Foundry test suite with gas optimization
- **Security**: Battle-tested patterns from Revert Finance

### **AI/ML Backend**
- **Language**: Python 3.9+
- **ML Framework**: scikit-learn, PyTorch for LSTM models
- **Data Pipeline**: Real-time blockchain data processing
- **APIs**: FastAPI for AI service endpoints

### **Infrastructure**
- **Blockchain**: Base Network (Ethereum L2)
- **Data Sources**: Alchemy, Uniswap Subgraph
- **Deployment**: Docker containerization
- **Monitoring**: Prometheus + Grafana dashboards

## 🚀 **Quick Start**

### **Prerequisites**
- Node.js 18+ and npm
- Python 3.9+
- Git

### **1. Clone Repository**
```bash
git clone https://github.com/your-org/dexter-protocol.git
cd dexter-protocol
```

### **2. AI Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python -m dexbrain.core
```

### **3. Website Setup**
The user interface is maintained in a separate repository for independent deployment:

```bash
# Clone the website repository
git clone https://github.com/MeltedMindz/dexter-website.git
cd dexter-website
npm install
npm run dev
```

See the [dexter-website repository](https://github.com/MeltedMindz/dexter-website) for complete setup instructions.

### **4. Smart Contract Development**
```bash
cd contracts
# Install Foundry if not already installed
forge install
forge test
```

## 📖 **Documentation**

- **[Architecture Guide](docs/architecture/)** - Technical system design
- **[API Documentation](docs/api/)** - Backend service APIs  
- **[Smart Contract Docs](contracts/README.md)** - Contract specifications
- **[Deployment Guide](docs/deployment/)** - Production setup instructions
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow

## 🔐 **Security**

Dexter Protocol implements enterprise-grade security practices:

- **Smart Contract Audits**: Professional security reviews
- **Rate Limiting**: API abuse protection and cost management
- **Environment Isolation**: Secure credential management
- **Input Validation**: Comprehensive data sanitization
- **Access Controls**: Multi-role permission systems

## 📊 **Performance**

### **AI Model Accuracy**
- **Price Prediction**: LSTM models with >85% directional accuracy
- **Volatility Forecasting**: Real-time regime detection
- **Yield Optimization**: 15-25% APR improvement over manual strategies

### **Gas Efficiency**
- **Batch Operations**: 70-90% gas savings on multiple positions
- **Smart Rebalancing**: Optimal timing to minimize transaction costs
- **V4 Hook Integration**: Sub-1% gas overhead for AI features

## 🤝 **Contributing**

We welcome contributions from the DeFi community! Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup and workflows
- Code style and testing requirements  
- Pull request and review process
- Community guidelines and standards

## 📄 **License**

This project is released under the **Source Available License** - see the [LICENSE](LICENSE) file for details.

- ✅ **Open Source**: View, study, and learn from the code
- ✅ **Non-Commercial**: Use for personal and educational purposes
- ❌ **Commercial Use**: Requires separate licensing agreement

For commercial licensing inquiries, contact: license@dexteragent.com

## 🌟 **Community**

- **Website**: [dexteragent.com](https://dexteragent.com)
- **Twitter**: [@DexterProtocol](https://twitter.com/DexterProtocol)
- **Discord**: [Community Server](https://discord.gg/dexter-protocol)
- **Documentation**: [docs.dexteragent.com](https://docs.dexteragent.com)

---

**Built with ❤️ by the Dexter Protocol team**

*Democratizing sophisticated liquidity management through artificial intelligence*