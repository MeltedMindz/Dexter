// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/proxy/Clones.sol";
import "@openzeppelin/contracts/utils/Address.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Factory.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";

import "../vaults/DexterVault.sol";
import "../vaults/IDexterVault.sol";
import "../strategies/StrategyManager.sol";
import "../fees/FeeManager.sol";
import "../validation/VaultClearing.sol";
import "../ranges/MultiRangeManager.sol";

/// @title VaultFactory - Factory for creating and managing Dexter vaults with integrated strategies
/// @notice Creates minimal proxy vaults with configurable strategies, fees, and risk management
contract VaultFactory is Ownable, ReentrancyGuard {
    using Address for address;
    using Clones for address;

    // ============ CONSTANTS ============
    
    uint256 public constant MAX_VAULTS_PER_POOL = 10;
    uint256 public constant MIN_INITIAL_LIQUIDITY = 1000; // Minimum initial liquidity
    
    // ============ ENUMS ============
    
    enum VaultTemplate {
        BASIC,              // Basic vault with standard features
        GAMMA_STYLE,        // Gamma-inspired dual position vault
        AI_OPTIMIZED,       // AI-powered optimization vault
        INSTITUTIONAL,      // Institutional-grade vault with advanced features
        CUSTOM              // Custom vault with specific configuration
    }
    
    enum DeploymentStatus {
        PENDING,
        DEPLOYED,
        FAILED,
        DEPRECATED
    }

    // ============ STRUCTS ============
    
    struct VaultTemplate {
        string name;
        string description;
        address implementation;
        IDexterVault.VaultConfig defaultConfig;
        uint256 minDeposit;
        uint256 maxDeposit;
        bool requiresWhitelist;
        bool aiEnabled;
        uint256 managementFee;
        uint256 performanceFee;
    }
    
    struct VaultInfo {
        address vaultAddress;
        address pool;
        address token0;
        address token1;
        uint24 fee;
        VaultTemplate templateType;
        DeploymentStatus status;
        address creator;
        uint256 createdAt;
        uint256 totalValueLocked;
        uint256 totalShares;
        string name;
        string symbol;
        bytes32 configHash;
    }
    
    struct VaultDeploymentParams {
        address token0;
        address token1;
        uint24 fee;
        VaultTemplate templateType;
        IDexterVault.VaultConfig vaultConfig;
        string name;
        string symbol;
        uint256 initialDeposit0;
        uint256 initialDeposit1;
        bool createPool;            // Create pool if it doesn't exist
        bool enableWhitelist;       // Enable whitelist functionality
        address[] initialWhitelist; // Initial whitelist addresses
        bytes customData;           // Custom configuration data
    }
    
    struct FactoryMetrics {
        uint256 totalVaultsCreated;
        uint256 totalValueLocked;
        uint256 totalFeesCollected;
        uint256 successfulDeployments;
        uint256 failedDeployments;
        mapping(VaultTemplate => uint256) templateUsage;
        mapping(address => uint256) creatorCounts;
    }

    // ============ STATE VARIABLES ============
    
    // Core contracts
    IUniswapV3Factory public immutable uniswapFactory;
    address public immutable compoundor;
    address public immutable aiManager;
    StrategyManager public immutable strategyManager;
    FeeManager public immutable feeManager;
    VaultClearing public immutable vaultClearing;
    MultiRangeManager public immutable rangeManager;
    
    // Templates and implementations
    mapping(VaultTemplate => VaultTemplate) public vaultTemplates;
    address public defaultImplementation;
    
    // Vault registry
    mapping(bytes32 => address) public getVault; // pool hash => vault address
    mapping(address => VaultInfo) public vaultInfo;
    mapping(address => address[]) public userVaults;
    mapping(address => mapping(address => mapping(uint24 => address[]))) public poolVaults; // token0 => token1 => fee => vaults[]
    address[] public allVaults;
    
    // Access control
    mapping(address => bool) public authorizedDeployers;
    mapping(VaultTemplate => bool) public templateEnabled;
    mapping(address => bool) public whitelistedTokens;
    bool public publicDeploymentEnabled = true;
    
    // Fee structure
    uint256 public deploymentFee = 0.01 ether;
    uint256 public protocolFeeShare = 1000; // 10% of vault fees
    address public treasury;
    
    // Metrics
    FactoryMetrics public metrics;
    
    // Configuration
    uint256 public maxVaultsPerUser = 50;
    uint256 public minTimeBetweenDeployments = 1 minutes;
    mapping(address => uint256) public lastDeploymentTime;

    // ============ EVENTS ============
    
    event VaultCreated(
        address indexed vault,
        address indexed pool,
        address indexed creator,
        address token0,
        address token1,
        uint24 fee,
        VaultTemplate templateType
    );
    event VaultTemplateAdded(VaultTemplate templateType, string name, address implementation);
    event VaultTemplateUpdated(VaultTemplate templateType, address newImplementation);
    event DeploymentFeeUpdated(uint256 oldFee, uint256 newFee);
    event AuthorizedDeployerSet(address indexed deployer, bool authorized);
    event VaultInitialized(address indexed vault, uint256 initialLiquidity, uint256 shares);
    event VaultDeprecated(address indexed vault, string reason);

    // ============ MODIFIERS ============
    
    modifier onlyAuthorizedDeployer() {
        require(
            publicDeploymentEnabled || 
            authorizedDeployers[msg.sender] || 
            msg.sender == owner(),
            "Not authorized to deploy"
        );
        _;
    }
    
    modifier validTemplate(VaultTemplate templateType) {
        require(templateEnabled[templateType], "Template not enabled");
        require(vaultTemplates[templateType].implementation != address(0), "Template not configured");
        _;
    }
    
    modifier deploymentCooldown() {
        require(
            block.timestamp >= lastDeploymentTime[msg.sender] + minTimeBetweenDeployments,
            "Deployment cooldown active"
        );
        _;
    }

    // ============ CONSTRUCTOR ============
    
    constructor(
        address _uniswapFactory,
        address _compoundor,
        address _aiManager,
        address _strategyManager,
        address _feeManager,
        address _vaultClearing,
        address _rangeManager,
        address _treasury
    ) {
        require(_uniswapFactory != address(0), "Invalid Uniswap factory");
        require(_compoundor != address(0), "Invalid compoundor");
        require(_aiManager != address(0), "Invalid AI manager");
        require(_treasury != address(0), "Invalid treasury");
        
        uniswapFactory = IUniswapV3Factory(_uniswapFactory);
        compoundor = _compoundor;
        aiManager = _aiManager;
        strategyManager = StrategyManager(_strategyManager);
        feeManager = FeeManager(_feeManager);
        vaultClearing = VaultClearing(_vaultClearing);
        rangeManager = MultiRangeManager(_rangeManager);
        treasury = _treasury;
        
        // Initialize default templates
        _initializeTemplates();
        
        // Set owner as authorized deployer
        authorizedDeployers[owner()] = true;
    }

    // ============ VAULT CREATION ============
    
    function createVault(VaultDeploymentParams calldata params)
        external
        payable
        onlyAuthorizedDeployer
        deploymentCooldown
        validTemplate(params.templateType)
        nonReentrant
        returns (address vault)
    {
        require(msg.value >= deploymentFee, "Insufficient deployment fee");
        require(userVaults[msg.sender].length < maxVaultsPerUser, "Too many vaults");
        require(params.token0 != params.token1, "Identical tokens");
        require(params.token0 != address(0) && params.token1 != address(0), "Zero address token");
        
        // Sort tokens
        (address token0, address token1) = params.token0 < params.token1 ? 
            (params.token0, params.token1) : (params.token1, params.token0);
        
        // Check token whitelist if enabled
        if (!publicDeploymentEnabled) {
            require(whitelistedTokens[token0] && whitelistedTokens[token1], "Tokens not whitelisted");
        }
        
        // Get or create pool
        address pool = _getOrCreatePool(token0, token1, params.fee, params.createPool);
        require(pool != address(0), "Pool creation failed");
        
        // Check vault limit per pool
        bytes32 poolHash = keccak256(abi.encodePacked(token0, token1, params.fee));
        require(poolVaults[token0][token1][params.fee].length < MAX_VAULTS_PER_POOL, "Too many vaults for pool");
        require(getVault[poolHash] == address(0), "Vault already exists");
        
        // Deploy vault
        vault = _deployVault(pool, params);
        
        // Initialize vault
        _initializeVault(vault, params);
        
        // Register vault
        _registerVault(vault, pool, token0, token1, params);
        
        // Collect deployment fee
        if (msg.value > 0) {
            payable(treasury).transfer(msg.value);
        }
        
        // Update metrics
        metrics.totalVaultsCreated++;
        metrics.successfulDeployments++;
        metrics.templateUsage[params.templateType]++;
        metrics.creatorCounts[msg.sender]++;
        
        lastDeploymentTime[msg.sender] = block.timestamp;
        
        emit VaultCreated(vault, pool, msg.sender, token0, token1, params.fee, params.templateType);
    }
    
    function createMultipleVaults(VaultDeploymentParams[] calldata paramsArray)
        external
        payable
        onlyAuthorizedDeployer
        nonReentrant
        returns (address[] memory vaults)
    {
        require(paramsArray.length > 0 && paramsArray.length <= 10, "Invalid batch size");
        require(msg.value >= deploymentFee * paramsArray.length, "Insufficient deployment fee");
        
        vaults = new address[](paramsArray.length);
        
        for (uint256 i = 0; i < paramsArray.length; i++) {
            // Note: Individual vault creation will handle most validations
            vaults[i] = this.createVault{value: deploymentFee}(paramsArray[i]);
        }
        
        // Refund excess
        uint256 excess = msg.value - (deploymentFee * paramsArray.length);
        if (excess > 0) {
            payable(msg.sender).transfer(excess);
        }
    }

    // ============ TEMPLATE MANAGEMENT ============
    
    function addVaultTemplate(
        VaultTemplate templateType,
        VaultTemplate calldata template
    ) external onlyOwner {
        require(template.implementation != address(0), "Invalid implementation");
        require(template.implementation.isContract(), "Implementation not a contract");
        
        vaultTemplates[templateType] = template;
        templateEnabled[templateType] = true;
        
        emit VaultTemplateAdded(templateType, template.name, template.implementation);
    }
    
    function updateVaultTemplate(
        VaultTemplate templateType,
        address newImplementation
    ) external onlyOwner {
        require(newImplementation != address(0), "Invalid implementation");
        require(newImplementation.isContract(), "Implementation not a contract");
        
        vaultTemplates[templateType].implementation = newImplementation;
        
        emit VaultTemplateUpdated(templateType, newImplementation);
    }
    
    function setTemplateEnabled(VaultTemplate templateType, bool enabled) external onlyOwner {
        templateEnabled[templateType] = enabled;
    }
    
    function setDefaultImplementation(address implementation) external onlyOwner {
        require(implementation != address(0), "Invalid implementation");
        require(implementation.isContract(), "Implementation not a contract");
        defaultImplementation = implementation;
    }

    // ============ ACCESS CONTROL ============
    
    function setAuthorizedDeployer(address deployer, bool authorized) external onlyOwner {
        authorizedDeployers[deployer] = authorized;
        emit AuthorizedDeployerSet(deployer, authorized);
    }
    
    function setPublicDeploymentEnabled(bool enabled) external onlyOwner {
        publicDeploymentEnabled = enabled;
    }
    
    function setWhitelistedToken(address token, bool whitelisted) external onlyOwner {
        whitelistedTokens[token] = whitelisted;
    }

    // ============ FEE MANAGEMENT ============
    
    function setDeploymentFee(uint256 newFee) external onlyOwner {
        uint256 oldFee = deploymentFee;
        deploymentFee = newFee;
        emit DeploymentFeeUpdated(oldFee, newFee);
    }
    
    function setProtocolFeeShare(uint256 share) external onlyOwner {
        require(share <= 2000, "Share too high"); // Max 20%
        protocolFeeShare = share;
    }
    
    function setTreasury(address newTreasury) external onlyOwner {
        require(newTreasury != address(0), "Invalid treasury");
        treasury = newTreasury;
    }

    // ============ CONFIGURATION ============
    
    function setMaxVaultsPerUser(uint256 max) external onlyOwner {
        require(max > 0 && max <= 1000, "Invalid max");
        maxVaultsPerUser = max;
    }
    
    function setMinTimeBetweenDeployments(uint256 time) external onlyOwner {
        require(time <= 1 hours, "Time too long");
        minTimeBetweenDeployments = time;
    }

    // ============ VAULT MANAGEMENT ============
    
    function deprecateVault(address vault, string calldata reason) external onlyOwner {
        require(vaultInfo[vault].vaultAddress != address(0), "Vault not found");
        
        vaultInfo[vault].status = DeploymentStatus.DEPRECATED;
        
        emit VaultDeprecated(vault, reason);
    }
    
    function upgradeVaultImplementation(address vault, address newImplementation) 
        external 
        onlyOwner 
    {
        require(vaultInfo[vault].vaultAddress != address(0), "Vault not found");
        require(newImplementation != address(0), "Invalid implementation");
        require(newImplementation.isContract(), "Implementation not a contract");
        
        // This would require upgradeable proxy pattern
        // Implementation depends on vault upgrade mechanism
    }

    // ============ VIEW FUNCTIONS ============
    
    function getVaultInfo(address vault) external view returns (VaultInfo memory) {
        return vaultInfo[vault];
    }
    
    function getUserVaults(address user) external view returns (address[] memory) {
        return userVaults[user];
    }
    
    function getPoolVaults(address token0, address token1, uint24 fee) 
        external 
        view 
        returns (address[] memory) 
    {
        return poolVaults[token0][token1][fee];
    }
    
    function getAllVaults() external view returns (address[] memory) {
        return allVaults;
    }
    
    function getVaultTemplate(VaultTemplate templateType) 
        external 
        view 
        returns (VaultTemplate memory) 
    {
        return vaultTemplates[templateType];
    }
    
    function getFactoryMetrics() 
        external 
        view 
        returns (
            uint256 totalVaults,
            uint256 totalTVL,
            uint256 totalFees,
            uint256 successRate
        ) 
    {
        totalVaults = metrics.totalVaultsCreated;
        totalTVL = metrics.totalValueLocked;
        totalFees = metrics.totalFeesCollected;
        
        if (metrics.totalVaultsCreated > 0) {
            successRate = (metrics.successfulDeployments * 10000) / metrics.totalVaultsCreated;
        }
    }
    
    function predictVaultAddress(
        address token0,
        address token1,
        uint24 fee,
        uint256 salt
    ) external view returns (address) {
        address implementation = defaultImplementation;
        bytes32 bytecodeHash = keccak256(abi.encodePacked(
            type(DexterVault).creationCode,
            abi.encode(token0, token1, fee, salt)
        ));
        
        return Clones.predictDeterministicAddress(implementation, bytecodeHash, address(this));
    }
    
    function computePoolAddress(address token0, address token1, uint24 fee) 
        external 
        view 
        returns (address pool) 
    {
        return uniswapFactory.getPool(token0, token1, fee);
    }

    // ============ INTERNAL FUNCTIONS ============
    
    function _initializeTemplates() internal {
        // Basic template
        vaultTemplates[VaultTemplate.BASIC] = VaultTemplate({
            name: "Basic Vault",
            description: "Standard automated liquidity management",
            implementation: defaultImplementation,
            defaultConfig: IDexterVault.VaultConfig({
                mode: IDexterVault.StrategyMode.AI_ASSISTED,
                positionType: IDexterVault.PositionType.SINGLE_RANGE,
                aiOptimizationEnabled: false,
                autoCompoundEnabled: true,
                rebalanceThreshold: 1000, // 10%
                maxSlippageBps: 100       // 1%
            }),
            minDeposit: 100 * 1e18,      // $100
            maxDeposit: type(uint256).max,
            requiresWhitelist: false,
            aiEnabled: false,
            managementFee: 50,           // 0.5%
            performanceFee: 1000         // 10%
        });
        
        templateEnabled[VaultTemplate.BASIC] = true;
        
        // AI Optimized template
        vaultTemplates[VaultTemplate.AI_OPTIMIZED] = VaultTemplate({
            name: "AI Optimized Vault",
            description: "Advanced AI-powered optimization",
            implementation: defaultImplementation,
            defaultConfig: IDexterVault.VaultConfig({
                mode: IDexterVault.StrategyMode.FULLY_AUTOMATED,
                positionType: IDexterVault.PositionType.AI_OPTIMIZED,
                aiOptimizationEnabled: true,
                autoCompoundEnabled: true,
                rebalanceThreshold: 200,  // 2%
                maxSlippageBps: 300       // 3%
            }),
            minDeposit: 1000 * 1e18,     // $1000
            maxDeposit: type(uint256).max,
            requiresWhitelist: false,
            aiEnabled: true,
            managementFee: 75,           // 0.75%
            performanceFee: 1500         // 15%
        });
        
        templateEnabled[VaultTemplate.AI_OPTIMIZED] = true;
        
        // Institutional template
        vaultTemplates[VaultTemplate.INSTITUTIONAL] = VaultTemplate({
            name: "Institutional Vault",
            description: "Enterprise-grade vault with advanced features",
            implementation: defaultImplementation,
            defaultConfig: IDexterVault.VaultConfig({
                mode: IDexterVault.StrategyMode.AI_ASSISTED,
                positionType: IDexterVault.PositionType.MULTI_RANGE,
                aiOptimizationEnabled: true,
                autoCompoundEnabled: true,
                rebalanceThreshold: 500,  // 5%
                maxSlippageBps: 50        // 0.5%
            }),
            minDeposit: 100000 * 1e18,   // $100k
            maxDeposit: type(uint256).max,
            requiresWhitelist: true,
            aiEnabled: true,
            managementFee: 25,           // 0.25%
            performanceFee: 750          // 7.5%
        });
        
        templateEnabled[VaultTemplate.INSTITUTIONAL] = true;
    }
    
    function _getOrCreatePool(
        address token0,
        address token1,
        uint24 fee,
        bool createPool
    ) internal returns (address pool) {
        pool = uniswapFactory.getPool(token0, token1, fee);
        
        if (pool == address(0) && createPool) {
            pool = uniswapFactory.createPool(token0, token1, fee);
        }
    }
    
    function _deployVault(
        address pool,
        VaultDeploymentParams calldata params
    ) internal returns (address vault) {
        VaultTemplate memory template = vaultTemplates[params.templateType];
        
        // Use template implementation or default
        address implementation = template.implementation != address(0) ? 
            template.implementation : defaultImplementation;
        
        // Create deterministic salt
        bytes32 salt = keccak256(abi.encodePacked(
            msg.sender,
            pool,
            params.name,
            block.timestamp
        ));
        
        // Deploy minimal proxy
        vault = Clones.cloneDeterministic(implementation, salt);
    }
    
    function _initializeVault(
        address vault,
        VaultDeploymentParams calldata params
    ) internal {
        VaultTemplate memory template = vaultTemplates[params.templateType];
        
        // Merge template config with custom config
        IDexterVault.VaultConfig memory finalConfig = params.vaultConfig;
        if (template.aiEnabled) {
            finalConfig.aiOptimizationEnabled = true;
        }
        
        // Initialize vault with compoundor, AI manager, etc.
        DexterVault(vault).initialize(
            IUniswapV3Pool(vaultInfo[vault].pool),
            compoundor,
            aiManager,
            params.name,
            params.symbol,
            finalConfig
        );
        
        // Setup fee configuration
        IDexterVault.FeeConfiguration memory feeConfig = IDexterVault.FeeConfiguration({
            managementFeeBps: template.managementFee,
            performanceFeeBps: template.performanceFee,
            aiOptimizationFeeBps: 25,
            feeRecipient: treasury,
            strategistShareBps: 2000,
            strategist: msg.sender
        });
        
        IDexterVault(vault).setFeeConfiguration(feeConfig);
        
        // Setup clearing and validation
        vaultClearing.authorizeVault(vault, true);
        rangeManager.authorizeVault(vault, true);
        strategyManager.authorizeVault(vault, true);
        
        // Initialize with liquidity if provided
        if (params.initialDeposit0 > 0 || params.initialDeposit1 > 0) {
            _initializeWithLiquidity(vault, params);
        }
        
        emit VaultInitialized(vault, 0, 0); // Would pass actual values
    }
    
    function _initializeWithLiquidity(
        address vault,
        VaultDeploymentParams calldata params
    ) internal {
        // Transfer initial liquidity from creator
        if (params.initialDeposit0 > 0) {
            IERC20(vaultInfo[vault].token0).transferFrom(
                msg.sender, 
                vault, 
                params.initialDeposit0
            );
        }
        
        if (params.initialDeposit1 > 0) {
            IERC20(vaultInfo[vault].token1).transferFrom(
                msg.sender, 
                vault, 
                params.initialDeposit1
            );
        }
        
        // Perform initial deposit
        uint256 shares = IDexterVault(vault).deposit(
            params.initialDeposit0 + params.initialDeposit1,
            msg.sender
        );
        
        // Update metrics
        metrics.totalValueLocked += params.initialDeposit0 + params.initialDeposit1;
    }
    
    function _registerVault(
        address vault,
        address pool,
        address token0,
        address token1,
        VaultDeploymentParams calldata params
    ) internal {
        bytes32 poolHash = keccak256(abi.encodePacked(token0, token1, params.fee));
        
        // Register in main mapping
        getVault[poolHash] = vault;
        
        // Store vault info
        vaultInfo[vault] = VaultInfo({
            vaultAddress: vault,
            pool: pool,
            token0: token0,
            token1: token1,
            fee: params.fee,
            templateType: params.templateType,
            status: DeploymentStatus.DEPLOYED,
            creator: msg.sender,
            createdAt: block.timestamp,
            totalValueLocked: 0,
            totalShares: 0,
            name: params.name,
            symbol: params.symbol,
            configHash: keccak256(abi.encode(params.vaultConfig))
        });
        
        // Add to arrays
        allVaults.push(vault);
        userVaults[msg.sender].push(vault);
        poolVaults[token0][token1][params.fee].push(vault);
    }
}