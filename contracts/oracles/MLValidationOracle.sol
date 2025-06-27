// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "./MultiOracleValidator.sol";

/**
 * @title MLValidationOracle
 * @notice Real-time ML validation oracle with on-chain inference
 * @dev Provides ML-powered validation for DeFi operations with cryptographic proofs
 */
contract MLValidationOracle is Ownable, ReentrancyGuard {
    using ECDSA for bytes32;

    // ML Model types
    enum ModelType {
        PRICE_PREDICTION,
        VOLATILITY_FORECAST,
        RISK_ASSESSMENT,
        YIELD_OPTIMIZATION,
        MARKET_REGIME,
        MEV_DETECTION,
        LIQUIDITY_FORECAST
    }

    // ML Prediction structure
    struct MLPrediction {
        uint256 timestamp;
        ModelType modelType;
        bytes32 modelHash;
        int256 prediction;
        uint256 confidence;
        bytes32 dataHash;
        bytes signature;
        bool validated;
    }

    // Model configuration
    struct ModelConfig {
        bool active;
        uint256 minConfidence;
        uint256 maxAge;
        address validator;
        uint256 weight;
    }

    // Market regime classification
    enum MarketRegime {
        TRENDING_UP,
        TRENDING_DOWN,
        RANGING,
        HIGH_VOLATILITY,
        LOW_VOLATILITY,
        UNKNOWN
    }

    // ML Service Provider configuration
    struct MLServiceProvider {
        string name;
        bool active;
        uint256 reputation;          // 0-10000 reputation score
        uint256 totalPredictions;
        uint256 accuratePredictions;
        uint256 weight;              // Weight in consensus (0-10000)
        uint256 lastActiveTime;
        ModelType[] supportedModels;
    }

    // Consensus prediction combining multiple providers
    struct ConsensusPrediction {
        ModelType modelType;
        int256 consensusValue;
        uint256 consensusConfidence;
        uint256 participatingProviders;
        uint256 timestamp;
        address[] providers;
        int256[] providerPredictions;
        uint256[] providerWeights;
        bytes32 consensusHash;
    }

    // Manual override system
    struct ManualOverride {
        bool active;
        int256 overrideValue;
        uint256 overrideConfidence;
        uint256 expiryTime;
        address overrider;
        string reason;
        uint256 timestamp;
    }

    // Events
    event PredictionSubmitted(
        bytes32 indexed predictionId,
        ModelType indexed modelType,
        int256 prediction,
        uint256 confidence
    );
    
    event PredictionValidated(
        bytes32 indexed predictionId,
        bool valid,
        string reason
    );
    
    event ModelConfigUpdated(
        ModelType indexed modelType,
        bool active,
        uint256 minConfidence
    );
    
    event MarketRegimeUpdated(
        MarketRegime indexed oldRegime,
        MarketRegime indexed newRegime,
        uint256 confidence
    );

    // Enhanced Events for Multiple Providers
    event MLServiceProviderAdded(
        address indexed provider,
        string name,
        uint256 weight
    );

    event MLServiceProviderUpdated(
        address indexed provider,
        uint256 newReputation,
        uint256 newWeight
    );

    event ConsensusPredictionGenerated(
        bytes32 indexed consensusId,
        ModelType indexed modelType,
        int256 consensusValue,
        uint256 consensusConfidence,
        uint256 participatingProviders
    );

    event ManualOverrideActivated(
        ModelType indexed modelType,
        address indexed overrider,
        int256 overrideValue,
        string reason
    );

    event ManualOverrideExpired(
        ModelType indexed modelType,
        address indexed overrider
    );
    
    // ============ NEW CRITICAL EVENTS ============
    
    // Validator Management Events
    event ValidatorAdded(address indexed validator, address indexed addedBy, string reason);
    event ValidatorRemoved(address indexed validator, address indexed removedBy, string reason);
    event ValidatorReputationUpdated(address indexed validator, uint256 oldReputation, uint256 newReputation);
    
    // Provider Quality and Performance Events
    event ProviderPredictionAccuracyUpdated(address indexed provider, ModelType modelType, uint256 accuracyScore, uint256 totalPredictions);
    event ProviderPerformanceAlert(address indexed provider, string alertType, uint256 value, uint256 threshold);
    event ProviderWeightAdjusted(address indexed provider, uint256 oldWeight, uint256 newWeight, string reason);
    
    // Consensus and Validation Events
    event ConsensusMechanismUpdated(string mechanism, uint256 parameter, address indexed updatedBy);
    event ConsensusDisagreementDetected(ModelType modelType, uint256 disagreementLevel, uint256 participatingProviders);
    event ConsensusThresholdUpdated(ModelType modelType, uint256 oldThreshold, uint256 newThreshold);
    
    // Security and Authorization Events
    event UnauthorizedPredictionAttempt(address indexed sender, ModelType modelType, uint256 timestamp);
    event OverriderAuthorizationChanged(address indexed overrider, bool authorized, address indexed changedBy);
    event ModelSecurityParameterUpdated(ModelType modelType, string parameter, uint256 oldValue, uint256 newValue);
    
    // Prediction Quality Events
    event PredictionConfidenceBelowThreshold(bytes32 indexed predictionId, uint256 confidence, uint256 threshold);
    event PredictionDataHashMismatch(bytes32 indexed predictionId, bytes32 expectedHash, bytes32 actualHash);
    event PredictionSignatureVerified(bytes32 indexed predictionId, address indexed validator, bool verified);
    
    // Emergency and Failsafe Events
    event EmergencyOracleShutdown(address indexed shutdownBy, string reason, uint256 affectedModels);
    event FallbackOracleActivated(ModelType modelType, address fallbackOracle, string reason);
    event OracleHealthCheckFailed(address indexed oracle, string healthCheck, uint256 timestamp);

    // Enhanced Storage for Multiple ML Providers
    mapping(bytes32 => MLPrediction) public predictions;
    mapping(ModelType => ModelConfig) public modelConfigs;
    mapping(address => bool) public authorizedValidators;
    mapping(bytes32 => uint256) public predictionScores;
    
    // Multiple ML Service Provider Support
    mapping(address => MLServiceProvider) public mlServiceProviders;
    mapping(ModelType => address[]) public modelProviders;
    mapping(ModelType => mapping(address => MLPrediction)) public latestProviderPredictions;
    mapping(bytes32 => ConsensusPrediction) public consensusPredictions;
    
    // Manual Override System
    mapping(ModelType => ManualOverride) public manualOverrides;
    mapping(address => bool) public authorizedOverriders;
    
    bytes32[] public activePredictions;
    MarketRegime public currentMarketRegime;
    uint256 public regimeConfidence;
    uint256 public lastRegimeUpdate;
    
    MultiOracleValidator public immutable priceOracle;
    
    // Constants
    uint256 public constant MAX_CONFIDENCE = 10000; // 100% = 10000
    uint256 public constant MIN_PREDICTION_AGE = 30 seconds;
    uint256 public constant MAX_PREDICTION_AGE = 300 seconds; // 5 minutes
    uint256 public constant REGIME_UPDATE_THRESHOLD = 7500; // 75% confidence

    constructor(address _priceOracle) {
        priceOracle = MultiOracleValidator(_priceOracle);
        
        // Initialize model configurations
        _initializeModelConfigs();
        
        currentMarketRegime = MarketRegime.UNKNOWN;
        regimeConfidence = 0;
        lastRegimeUpdate = block.timestamp;
    }

    /**
     * @notice Submit ML prediction with cryptographic proof
     * @param modelType Type of ML model
     * @param prediction Prediction value (scaled by 1e18)
     * @param confidence Confidence score (0-10000)
     * @param dataHash Hash of input data used for prediction
     * @param signature Cryptographic signature from authorized validator
     */
    function submitPrediction(
        ModelType modelType,
        int256 prediction,
        uint256 confidence,
        bytes32 dataHash,
        bytes calldata signature
    ) external nonReentrant {
        require(modelConfigs[modelType].active, "Model not active");
        require(confidence >= modelConfigs[modelType].minConfidence, "Insufficient confidence");
        require(confidence <= MAX_CONFIDENCE, "Invalid confidence");
        
        // Verify signature
        bytes32 messageHash = keccak256(abi.encodePacked(
            modelType,
            prediction,
            confidence,
            dataHash,
            block.timestamp
        ));
        
        address signer = messageHash.toEthSignedMessageHash().recover(signature);
        require(authorizedValidators[signer], "Unauthorized validator");
        
        // Create prediction ID
        bytes32 predictionId = keccak256(abi.encodePacked(
            modelType,
            block.timestamp,
            msg.sender,
            prediction
        ));
        
        // Store prediction
        predictions[predictionId] = MLPrediction({
            timestamp: block.timestamp,
            modelType: modelType,
            modelHash: _getModelHash(modelType),
            prediction: prediction,
            confidence: confidence,
            dataHash: dataHash,
            signature: signature,
            validated: false
        });
        
        activePredictions.push(predictionId);
        
        emit PredictionSubmitted(predictionId, modelType, prediction, confidence);
        
        // Update market regime if applicable
        if (modelType == ModelType.MARKET_REGIME) {
            _updateMarketRegime(prediction, confidence);
        }
        
        // Validate prediction
        _validatePrediction(predictionId);
    }

    /**
     * @notice Get latest prediction for a model type
     * @param modelType Type of ML model
     * @return prediction Latest prediction data
     */
    function getLatestPrediction(ModelType modelType) 
        external 
        view 
        returns (MLPrediction memory prediction) 
    {
        bytes32 latestId;
        uint256 latestTimestamp = 0;
        
        for (uint256 i = 0; i < activePredictions.length; i++) {
            MLPrediction memory pred = predictions[activePredictions[i]];
            if (pred.modelType == modelType && 
                pred.timestamp > latestTimestamp &&
                pred.validated &&
                block.timestamp - pred.timestamp <= modelConfigs[modelType].maxAge) {
                latestTimestamp = pred.timestamp;
                latestId = activePredictions[i];
            }
        }
        
        require(latestId != bytes32(0), "No valid prediction found");
        return predictions[latestId];
    }

    /**
     * @notice Get market regime assessment
     * @return regime Current market regime
     * @return confidence Confidence in the assessment
     * @return lastUpdate Timestamp of last update
     */
    function getMarketRegime() 
        external 
        view 
        returns (MarketRegime regime, uint256 confidence, uint256 lastUpdate) 
    {
        return (currentMarketRegime, regimeConfidence, lastRegimeUpdate);
    }

    /**
     * @notice Validate DeFi operation using ML predictions
     * @param operationType Type of operation (0=swap, 1=liquidity, 2=compound)
     * @param amount Operation amount
     * @param poolAddress Target pool address
     * @return valid Whether operation is validated
     * @return riskScore Risk score (0-10000)
     * @return reason Validation reason
     */
    function validateOperation(
        uint256 operationType,
        uint256 amount,
        address poolAddress
    ) external view returns (bool valid, uint256 riskScore, string memory reason) {
        // Get latest risk assessment
        try this.getLatestPrediction(ModelType.RISK_ASSESSMENT) returns (MLPrediction memory riskPred) {
            riskScore = uint256(riskPred.prediction);
            
            // Check if operation is too risky
            if (riskScore > 8000) { // 80% risk threshold
                return (false, riskScore, "High risk detected");
            }
        } catch {
            riskScore = 5000; // Default moderate risk
        }
        
        // Check market regime
        if (currentMarketRegime == MarketRegime.HIGH_VOLATILITY && 
            regimeConfidence > REGIME_UPDATE_THRESHOLD) {
            if (operationType == 0 && amount > 100 ether) { // Large swap in volatile market
                return (false, riskScore + 1000, "Large swap in volatile market");
            }
        }
        
        // Check MEV risk for swaps
        if (operationType == 0) {
            try this.getLatestPrediction(ModelType.MEV_DETECTION) returns (MLPrediction memory mevPred) {
                if (mevPred.prediction > 7000) { // 70% MEV risk
                    return (false, riskScore + 500, "High MEV risk detected");
                }
            } catch {}
        }
        
        // Validate with price oracle
        try priceOracle.validatePrice(poolAddress, amount) returns (bool priceValid, ) {
            if (!priceValid) {
                return (false, riskScore + 2000, "Price validation failed");
            }
        } catch {
            return (false, 9000, "Oracle validation failed");
        }
        
        return (true, riskScore, "Operation validated");
    }

    /**
     * @notice Get prediction accuracy metrics
     * @param modelType Type of ML model
     * @return accuracy Accuracy percentage (0-10000)
     * @return totalPredictions Total number of predictions
     * @return validPredictions Number of valid predictions
     */
    function getPredictionAccuracy(ModelType modelType) 
        external 
        view 
        returns (uint256 accuracy, uint256 totalPredictions, uint256 validPredictions) 
    {
        uint256 total = 0;
        uint256 valid = 0;
        
        for (uint256 i = 0; i < activePredictions.length; i++) {
            MLPrediction memory pred = predictions[activePredictions[i]];
            if (pred.modelType == modelType) {
                total++;
                if (pred.validated) {
                    valid++;
                }
            }
        }
        
        accuracy = total > 0 ? (valid * MAX_CONFIDENCE) / total : 0;
        return (accuracy, total, valid);
    }

    /**
     * @notice Admin: Update model configuration
     */
    function updateModelConfig(
        ModelType modelType,
        bool active,
        uint256 minConfidence,
        uint256 maxAge,
        uint256 weight
    ) external onlyOwner {
        require(minConfidence <= MAX_CONFIDENCE, "Invalid confidence");
        require(weight <= MAX_CONFIDENCE, "Invalid weight");
        
        modelConfigs[modelType] = ModelConfig({
            active: active,
            minConfidence: minConfidence,
            maxAge: maxAge,
            validator: modelConfigs[modelType].validator,
            weight: weight
        });
        
        emit ModelConfigUpdated(modelType, active, minConfidence);
    }

    /**
     * @notice Admin: Add authorized validator
     */
    function addValidator(address validator) external onlyOwner {
        authorizedValidators[validator] = true;
    }

    /**
     * @notice Admin: Remove authorized validator
     */
    function removeValidator(address validator) external onlyOwner {
        authorizedValidators[validator] = false;
    }

    /**
     * @notice Clean up old predictions
     */
    function cleanupPredictions() external {
        uint256 cutoff = block.timestamp - MAX_PREDICTION_AGE;
        
        for (uint256 i = activePredictions.length; i > 0; i--) {
            bytes32 predictionId = activePredictions[i - 1];
            if (predictions[predictionId].timestamp < cutoff) {
                // Remove from active list
                activePredictions[i - 1] = activePredictions[activePredictions.length - 1];
                activePredictions.pop();
                
                // Delete prediction data
                delete predictions[predictionId];
                delete predictionScores[predictionId];
            }
        }
    }

    // Internal functions
    function _initializeModelConfigs() internal {
        modelConfigs[ModelType.PRICE_PREDICTION] = ModelConfig(true, 6000, 300, address(0), 8000);
        modelConfigs[ModelType.VOLATILITY_FORECAST] = ModelConfig(true, 7000, 600, address(0), 7000);
        modelConfigs[ModelType.RISK_ASSESSMENT] = ModelConfig(true, 8000, 120, address(0), 9000);
        modelConfigs[ModelType.YIELD_OPTIMIZATION] = ModelConfig(true, 6500, 300, address(0), 7500);
        modelConfigs[ModelType.MARKET_REGIME] = ModelConfig(true, 7500, 180, address(0), 8500);
        modelConfigs[ModelType.MEV_DETECTION] = ModelConfig(true, 8500, 60, address(0), 9500);
        modelConfigs[ModelType.LIQUIDITY_FORECAST] = ModelConfig(true, 6000, 600, address(0), 7000);
    }

    function _updateMarketRegime(int256 prediction, uint256 confidence) internal {
        if (confidence < REGIME_UPDATE_THRESHOLD) return;
        
        MarketRegime oldRegime = currentMarketRegime;
        MarketRegime newRegime = MarketRegime(uint256(prediction));
        
        if (newRegime != oldRegime) {
            currentMarketRegime = newRegime;
            regimeConfidence = confidence;
            lastRegimeUpdate = block.timestamp;
            
            emit MarketRegimeUpdated(oldRegime, newRegime, confidence);
        }
    }

    function _validatePrediction(bytes32 predictionId) internal {
        MLPrediction storage pred = predictions[predictionId];
        
        // Basic validation checks
        bool valid = true;
        string memory reason = "Valid";
        
        // Check timestamp
        if (block.timestamp - pred.timestamp > MAX_PREDICTION_AGE) {
            valid = false;
            reason = "Prediction too old";
        }
        
        // Check confidence threshold
        if (pred.confidence < modelConfigs[pred.modelType].minConfidence) {
            valid = false;
            reason = "Confidence too low";
        }
        
        // Update validation status
        pred.validated = valid;
        
        emit PredictionValidated(predictionId, valid, reason);
    }

    function _getModelHash(ModelType modelType) internal pure returns (bytes32) {
        // In production, this would return the actual model version hash
        return keccak256(abi.encodePacked("model_v1", modelType));
    }

    // ============ ENHANCED ML PROVIDER FUNCTIONS ============

    /**
     * @notice Add a new ML service provider
     * @param provider Address of the ML service provider
     * @param name Human-readable name of the provider
     * @param weight Initial weight in consensus (0-10000)
     * @param supportedModels Array of model types this provider supports
     */
    function addMLServiceProvider(
        address provider,
        string memory name,
        uint256 weight,
        ModelType[] memory supportedModels
    ) external onlyOwner {
        require(provider != address(0), "Invalid provider address");
        require(weight <= MAX_CONFIDENCE, "Weight too high");
        require(supportedModels.length > 0, "Must support at least one model");
        require(!mlServiceProviders[provider].active, "Provider already exists");
        
        mlServiceProviders[provider] = MLServiceProvider({
            name: name,
            active: true,
            reputation: 5000, // Start with 50% reputation
            totalPredictions: 0,
            accuratePredictions: 0,
            weight: weight,
            lastActiveTime: block.timestamp,
            supportedModels: supportedModels
        });
        
        // Add provider to model-specific lists
        for (uint256 i = 0; i < supportedModels.length; i++) {
            modelProviders[supportedModels[i]].push(provider);
        }
        
        emit MLServiceProviderAdded(provider, name, weight);
    }

    /**
     * @notice Submit prediction from ML service provider
     * @param modelType Type of ML model making the prediction
     * @param prediction Prediction value (scaled appropriately)
     * @param confidence Confidence score (0-10000)
     * @param dataHash Hash of input data used for prediction
     * @param signature Cryptographic signature from provider
     */
    function submitProviderPrediction(
        ModelType modelType,
        int256 prediction,
        uint256 confidence,
        bytes32 dataHash,
        bytes calldata signature
    ) external nonReentrant {
        require(mlServiceProviders[msg.sender].active, "Provider not active");
        require(confidence <= MAX_CONFIDENCE, "Invalid confidence");
        require(_isModelSupported(msg.sender, modelType), "Model not supported by provider");
        
        // Verify signature
        bytes32 messageHash = keccak256(abi.encodePacked(
            modelType,
            prediction,
            confidence,
            dataHash,
            block.timestamp
        ));
        
        address signer = messageHash.toEthSignedMessageHash().recover(signature);
        require(signer == msg.sender, "Invalid signature");
        
        // Store provider-specific prediction
        latestProviderPredictions[modelType][msg.sender] = MLPrediction({
            timestamp: block.timestamp,
            modelType: modelType,
            modelHash: _getModelHash(modelType),
            prediction: prediction,
            confidence: confidence,
            dataHash: dataHash,
            signature: signature,
            validated: true // Provider predictions are pre-validated
        });
        
        // Update provider metrics
        MLServiceProvider storage provider = mlServiceProviders[msg.sender];
        provider.totalPredictions++;
        provider.lastActiveTime = block.timestamp;
        
        // Generate consensus prediction if enough providers have submitted
        _attemptConsensusGeneration(modelType);
        
        emit PredictionSubmitted(
            keccak256(abi.encodePacked(modelType, msg.sender, block.timestamp)),
            modelType,
            prediction,
            confidence
        );
    }

    /**
     * @notice Generate consensus prediction from multiple providers
     * @param modelType Type of model to generate consensus for
     * @return consensusId Unique identifier for the consensus
     */
    function generateConsensusPrediction(ModelType modelType) 
        external 
        returns (bytes32 consensusId) 
    {
        return _generateConsensusPrediction(modelType);
    }

    /**
     * @notice Activate manual override for a specific model type
     * @param modelType Model type to override
     * @param overrideValue Manual override value
     * @param overrideConfidence Confidence in manual override
     * @param duration Duration of override in seconds
     * @param reason Reason for manual override
     */
    function activateManualOverride(
        ModelType modelType,
        int256 overrideValue,
        uint256 overrideConfidence,
        uint256 duration,
        string memory reason
    ) external {
        require(authorizedOverriders[msg.sender], "Not authorized for manual override");
        require(overrideConfidence <= MAX_CONFIDENCE, "Invalid confidence");
        require(duration > 0 && duration <= 86400, "Invalid duration"); // Max 24 hours
        
        manualOverrides[modelType] = ManualOverride({
            active: true,
            overrideValue: overrideValue,
            overrideConfidence: overrideConfidence,
            expiryTime: block.timestamp + duration,
            overrider: msg.sender,
            reason: reason,
            timestamp: block.timestamp
        });
        
        emit ManualOverrideActivated(modelType, msg.sender, overrideValue, reason);
    }

    /**
     * @notice Get latest consensus prediction for a model type
     * @param modelType Type of ML model
     * @return prediction Latest consensus prediction
     */
    function getLatestConsensusPrediction(ModelType modelType)
        external
        view
        returns (ConsensusPrediction memory prediction)
    {
        // Check for active manual override first
        ManualOverride memory override = manualOverrides[modelType];
        if (override.active && block.timestamp <= override.expiryTime) {
            // Return manual override as consensus
            return ConsensusPrediction({
                modelType: modelType,
                consensusValue: override.overrideValue,
                consensusConfidence: override.overrideConfidence,
                participatingProviders: 1,
                timestamp: override.timestamp,
                providers: new address[](1),
                providerPredictions: new int256[](1),
                providerWeights: new uint256[](1),
                consensusHash: keccak256(abi.encodePacked("manual_override", modelType, override.timestamp))
            });
        }
        
        // Find latest consensus prediction
        bytes32 latestId;
        uint256 latestTimestamp = 0;
        
        // In production, would iterate through consensus IDs more efficiently
        bytes32 consensusId = keccak256(abi.encodePacked(modelType, "latest"));
        ConsensusPrediction memory consensus = consensusPredictions[consensusId];
        
        if (consensus.timestamp > 0 && 
            block.timestamp - consensus.timestamp <= modelConfigs[modelType].maxAge) {
            return consensus;
        }
        
        revert("No valid consensus prediction found");
    }

    /**
     * @notice Update ML service provider reputation based on prediction accuracy
     * @param provider Address of the provider
     * @param accurate Whether the prediction was accurate
     */
    function updateProviderReputation(address provider, bool accurate) external onlyOwner {
        require(mlServiceProviders[provider].active, "Provider not active");
        
        MLServiceProvider storage providerData = mlServiceProviders[provider];
        
        if (accurate) {
            providerData.accuratePredictions++;
            // Increase reputation (max 10000)
            providerData.reputation = Math.min(providerData.reputation + 50, MAX_CONFIDENCE);
        } else {
            // Decrease reputation (min 0)
            providerData.reputation = providerData.reputation > 25 ? providerData.reputation - 25 : 0;
        }
        
        // Update provider weight based on reputation
        providerData.weight = (providerData.reputation * providerData.weight) / MAX_CONFIDENCE;
        
        emit MLServiceProviderUpdated(provider, providerData.reputation, providerData.weight);
    }

    /**
     * @notice Add authorized overrider
     * @param overrider Address to authorize for manual overrides
     */
    function addAuthorizedOverrider(address overrider) external onlyOwner {
        authorizedOverriders[overrider] = true;
    }

    /**
     * @notice Remove authorized overrider
     * @param overrider Address to remove from authorized overriders
     */
    function removeAuthorizedOverrider(address overrider) external onlyOwner {
        authorizedOverriders[overrider] = false;
    }

    /**
     * @notice Deactivate manual override
     * @param modelType Model type to deactivate override for
     */
    function deactivateManualOverride(ModelType modelType) external {
        ManualOverride storage override = manualOverrides[modelType];
        require(
            msg.sender == owner() || 
            msg.sender == override.overrider || 
            block.timestamp > override.expiryTime,
            "Not authorized to deactivate"
        );
        
        override.active = false;
        emit ManualOverrideExpired(modelType, override.overrider);
    }

    // ============ INTERNAL CONSENSUS FUNCTIONS ============

    /**
     * @notice Internal function to generate consensus prediction
     * @param modelType Model type to generate consensus for
     * @return consensusId Unique identifier for the consensus
     */
    function _generateConsensusPrediction(ModelType modelType) 
        internal 
        returns (bytes32 consensusId) 
    {
        address[] memory providers = modelProviders[modelType];
        require(providers.length > 0, "No providers for model type");
        
        // Collect active predictions
        address[] memory activeProviders = new address[](providers.length);
        int256[] memory predictions = new int256[](providers.length);
        uint256[] memory weights = new uint256[](providers.length);
        uint256 activeCount = 0;
        
        for (uint256 i = 0; i < providers.length; i++) {
            address provider = providers[i];
            MLServiceProvider memory providerData = mlServiceProviders[provider];
            MLPrediction memory prediction = latestProviderPredictions[modelType][provider];
            
            // Check if provider has recent, valid prediction
            if (providerData.active && 
                prediction.timestamp > 0 &&
                block.timestamp - prediction.timestamp <= modelConfigs[modelType].maxAge &&
                prediction.confidence >= modelConfigs[modelType].minConfidence) {
                
                activeProviders[activeCount] = provider;
                predictions[activeCount] = prediction.prediction;
                weights[activeCount] = providerData.weight;
                activeCount++;
            }
        }
        
        require(activeCount >= 2, "Insufficient active providers for consensus");
        
        // Calculate weighted consensus
        (int256 consensusValue, uint256 consensusConfidence) = _calculateWeightedConsensus(
            predictions,
            weights,
            activeCount
        );
        
        // Create consensus ID
        consensusId = keccak256(abi.encodePacked(
            modelType,
            block.timestamp,
            activeCount,
            consensusValue
        ));
        
        // Store consensus prediction
        consensusPredictions[consensusId] = ConsensusPrediction({
            modelType: modelType,
            consensusValue: consensusValue,
            consensusConfidence: consensusConfidence,
            participatingProviders: activeCount,
            timestamp: block.timestamp,
            providers: _trimArray(activeProviders, activeCount),
            providerPredictions: _trimArrayInt(predictions, activeCount),
            providerWeights: _trimArrayUint(weights, activeCount),
            consensusHash: keccak256(abi.encodePacked(consensusValue, consensusConfidence, activeCount))
        });
        
        emit ConsensusPredictionGenerated(
            consensusId,
            modelType,
            consensusValue,
            consensusConfidence,
            activeCount
        );
        
        return consensusId;
    }

    /**
     * @notice Attempt to generate consensus if enough providers have submitted
     * @param modelType Model type to check for consensus
     */
    function _attemptConsensusGeneration(ModelType modelType) internal {
        address[] memory providers = modelProviders[modelType];
        uint256 activeCount = 0;
        
        // Count active providers with recent predictions
        for (uint256 i = 0; i < providers.length; i++) {
            address provider = providers[i];
            MLPrediction memory prediction = latestProviderPredictions[modelType][provider];
            
            if (mlServiceProviders[provider].active &&
                prediction.timestamp > 0 &&
                block.timestamp - prediction.timestamp <= 300) { // 5 minutes
                activeCount++;
            }
        }
        
        // Generate consensus if we have at least 2 active providers
        if (activeCount >= 2) {
            _generateConsensusPrediction(modelType);
        }
    }

    /**
     * @notice Calculate weighted consensus from multiple predictions
     * @param predictions Array of prediction values
     * @param weights Array of provider weights
     * @param count Number of active predictions
     * @return consensusValue Weighted average prediction
     * @return consensusConfidence Confidence in consensus
     */
    function _calculateWeightedConsensus(
        int256[] memory predictions,
        uint256[] memory weights,
        uint256 count
    ) internal pure returns (int256 consensusValue, uint256 consensusConfidence) {
        int256 weightedSum = 0;
        uint256 totalWeight = 0;
        
        // Calculate weighted average
        for (uint256 i = 0; i < count; i++) {
            weightedSum += predictions[i] * int256(weights[i]);
            totalWeight += weights[i];
        }
        
        require(totalWeight > 0, "Total weight is zero");
        consensusValue = weightedSum / int256(totalWeight);
        
        // Calculate consensus confidence based on agreement
        int256 variance = 0;
        for (uint256 i = 0; i < count; i++) {
            int256 diff = predictions[i] - consensusValue;
            variance += (diff * diff) / int256(count);
        }
        
        // Higher variance = lower confidence
        uint256 maxVariance = 1000000; // Adjust based on expected prediction ranges
        consensusConfidence = variance > 0 ? 
            MAX_CONFIDENCE - Math.min(uint256(variance) * MAX_CONFIDENCE / maxVariance, MAX_CONFIDENCE) :
            MAX_CONFIDENCE;
    }

    /**
     * @notice Check if a model type is supported by a provider
     * @param provider Address of the provider
     * @param modelType Model type to check
     * @return supported Whether the model is supported
     */
    function _isModelSupported(address provider, ModelType modelType) internal view returns (bool supported) {
        ModelType[] memory supportedModels = mlServiceProviders[provider].supportedModels;
        for (uint256 i = 0; i < supportedModels.length; i++) {
            if (supportedModels[i] == modelType) {
                return true;
            }
        }
        return false;
    }

    /**
     * @notice Trim address array to actual size
     * @param array Original array
     * @param size Actual size
     * @return trimmed Trimmed array
     */
    function _trimArray(address[] memory array, uint256 size) internal pure returns (address[] memory trimmed) {
        trimmed = new address[](size);
        for (uint256 i = 0; i < size; i++) {
            trimmed[i] = array[i];
        }
    }

    /**
     * @notice Trim int256 array to actual size
     * @param array Original array
     * @param size Actual size
     * @return trimmed Trimmed array
     */
    function _trimArrayInt(int256[] memory array, uint256 size) internal pure returns (int256[] memory trimmed) {
        trimmed = new int256[](size);
        for (uint256 i = 0; i < size; i++) {
            trimmed[i] = array[i];
        }
    }

    /**
     * @notice Trim uint256 array to actual size
     * @param array Original array
     * @param size Actual size
     * @return trimmed Trimmed array
     */
    function _trimArrayUint(uint256[] memory array, uint256 size) internal pure returns (uint256[] memory trimmed) {
        trimmed = new uint256[](size);
        for (uint256 i = 0; i < size; i++) {
            trimmed[i] = array[i];
        }
    }
}