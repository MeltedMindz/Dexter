// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title MEVProtection
 * @notice Advanced MEV protection using commit-reveal schemes and multi-block validation
 * @dev Implements sophisticated protection against sandwich attacks, front-running, and back-running
 */
contract MEVProtection is Ownable, ReentrancyGuard {
    using ECDSA for bytes32;

    // ========================================
    // COMMIT-REVEAL SCHEME
    // ========================================

    struct Commitment {
        bytes32 commitHash;
        uint256 commitBlock;
        uint256 revealDeadline;
        address committer;
        uint256 value;
        bool revealed;
        bool executed;
    }

    struct RevealData {
        uint256 operation;      // Operation type (compound, swap, etc.)
        uint256 tokenId;        // Position token ID
        uint256 nonce;          // Unique nonce
        uint256 amount;         // Operation amount
        bytes32 salt;           // Random salt
        uint256 maxGasPrice;    // Maximum gas price
        uint256 deadline;       // Transaction deadline
    }

    mapping(bytes32 => Commitment) public commitments;
    mapping(address => uint256) public userNonces;
    mapping(address => uint256) private _lastCommitBlock;
    
    // Commit-reveal timing parameters
    uint256 public constant MIN_COMMIT_REVEAL_DELAY = 2; // Minimum blocks between commit and reveal
    uint256 public constant MAX_COMMIT_REVEAL_DELAY = 50; // Maximum blocks (~10 minutes)
    uint256 public constant REVEAL_WINDOW = 20; // Blocks available for reveal after min delay

    // ========================================
    // MULTI-BLOCK VALIDATION
    // ========================================

    struct BlockValidation {
        uint256 blockNumber;
        bytes32 blockHash;
        uint256 gasPrice;
        uint256 timestamp;
        uint256 transactionCount;
        bool isHighMEVRisk;
    }

    mapping(uint256 => BlockValidation) public blockData;
    mapping(address => uint256[]) private _userRecentBlocks;
    
    uint256 public constant MAX_GAS_PRICE_INCREASE = 2000; // 20% max increase between blocks
    uint256 public constant HIGH_MEV_TX_THRESHOLD = 100; // Transactions per block indicating high MEV activity
    uint256 public constant BLOCK_HISTORY_LIMIT = 10; // Blocks to track per user

    // ========================================
    // MEV DETECTION PARAMETERS
    // ========================================

    struct MEVMetrics {
        uint256 avgGasPrice;
        uint256 gasSpike;
        uint256 blockUtilization;
        uint256 suspiciousPatterns;
        bool frontRunningDetected;
        bool sandwichDetected;
        bool backRunningDetected;
    }

    mapping(uint256 => MEVMetrics) public blockMEVData;
    
    uint256 public constant GAS_SPIKE_THRESHOLD = 5000; // 50% gas price spike
    uint256 public constant BLOCK_UTILIZATION_THRESHOLD = 9000; // 90% block utilization
    uint256 public constant PATTERN_THRESHOLD = 3; // Suspicious pattern count

    // ========================================
    // OPERATION TYPES
    // ========================================

    enum OperationType {
        COMPOUND,
        SWAP,
        ADD_LIQUIDITY,
        REMOVE_LIQUIDITY,
        COLLECT_FEES,
        REBALANCE
    }

    // ========================================
    // COMMIT PHASE
    // ========================================

    modifier requireValidCommitTiming() {
        require(
            block.number > _lastCommitBlock[msg.sender] + 1,
            "MEVProtection: Must wait at least 1 block between commits"
        );
        _;
    }

    function commitOperation(
        bytes32 commitHash,
        uint256 value
    ) external payable requireValidCommitTiming nonReentrant {
        require(commitHash != bytes32(0), "MEVProtection: Invalid commit hash");
        require(msg.value == value, "MEVProtection: Value mismatch");
        
        // Check for MEV risk in current block
        _updateBlockMetrics(block.number);
        MEVMetrics memory metrics = blockMEVData[block.number];
        
        // Require lower gas price during high MEV periods
        if (metrics.frontRunningDetected || metrics.sandwichDetected) {
            require(
                tx.gasprice <= metrics.avgGasPrice * 110 / 100, // Max 10% above average
                "MEVProtection: Gas price too high during MEV activity"
            );
        }

        uint256 revealDeadline = block.number + MIN_COMMIT_REVEAL_DELAY + REVEAL_WINDOW;
        
        commitments[commitHash] = Commitment({
            commitHash: commitHash,
            commitBlock: block.number,
            revealDeadline: revealDeadline,
            committer: msg.sender,
            value: value,
            revealed: false,
            executed: false
        });

        _lastCommitBlock[msg.sender] = block.number;
        
        emit OperationCommitted(msg.sender, commitHash, block.number, revealDeadline);
    }

    // ========================================
    // REVEAL PHASE
    // ========================================

    function revealAndExecute(
        uint256 operation,
        uint256 tokenId,
        uint256 nonce,
        uint256 amount,
        bytes32 salt,
        uint256 maxGasPrice,
        uint256 deadline
    ) external nonReentrant {
        // Reconstruct commit hash
        bytes32 commitHash = keccak256(abi.encodePacked(
            msg.sender,
            operation,
            tokenId,
            nonce,
            amount,
            salt,
            maxGasPrice,
            deadline
        ));

        Commitment storage commitment = commitments[commitHash];
        
        // Validate commitment exists and belongs to sender
        require(commitment.committer == msg.sender, "MEVProtection: Invalid committer");
        require(!commitment.revealed, "MEVProtection: Already revealed");
        require(!commitment.executed, "MEVProtection: Already executed");
        
        // Check timing constraints
        require(
            block.number >= commitment.commitBlock + MIN_COMMIT_REVEAL_DELAY,
            "MEVProtection: Too early to reveal"
        );
        require(
            block.number <= commitment.revealDeadline,
            "MEVProtection: Reveal deadline passed"
        );
        
        // Check deadline hasn't passed
        require(block.timestamp <= deadline, "MEVProtection: Operation deadline passed");
        
        // Validate gas price constraints
        require(tx.gasprice <= maxGasPrice, "MEVProtection: Gas price too high");
        
        // Multi-block validation
        require(
            _validateMultiBlockExecution(commitment.commitBlock, block.number),
            "MEVProtection: Multi-block validation failed"
        );

        // Check nonce
        require(nonce == userNonces[msg.sender], "MEVProtection: Invalid nonce");
        userNonces[msg.sender]++;

        // Mark as revealed
        commitment.revealed = true;

        // Execute the operation based on type
        _executeOperation(
            OperationType(operation),
            tokenId,
            amount,
            commitment.value
        );

        commitment.executed = true;

        emit OperationRevealed(msg.sender, commitHash, OperationType(operation), tokenId, amount);
    }

    // ========================================
    // MULTI-BLOCK VALIDATION
    // ========================================

    function _validateMultiBlockExecution(
        uint256 commitBlock,
        uint256 executeBlock
    ) internal view returns (bool) {
        // Check for suspicious gas price patterns
        for (uint256 i = commitBlock + 1; i <= executeBlock && i <= commitBlock + 10; i++) {
            if (blockData[i].blockNumber == 0) continue; // Skip if no data
            
            MEVMetrics memory metrics = blockMEVData[i];
            
            // Reject if high MEV activity detected
            if (metrics.frontRunningDetected || 
                metrics.sandwichDetected || 
                metrics.suspiciousPatterns >= PATTERN_THRESHOLD) {
                return false;
            }
            
            // Check for unusual gas price spikes
            if (i > commitBlock + 1) {
                MEVMetrics memory prevMetrics = blockMEVData[i - 1];
                if (prevMetrics.avgGasPrice > 0) {
                    uint256 gasIncrease = metrics.avgGasPrice > prevMetrics.avgGasPrice ?
                        ((metrics.avgGasPrice - prevMetrics.avgGasPrice) * 10000) / prevMetrics.avgGasPrice : 0;
                    
                    if (gasIncrease > MAX_GAS_PRICE_INCREASE) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    function _updateBlockMetrics(uint256 blockNumber) internal {
        if (blockData[blockNumber].blockNumber != 0) return; // Already updated

        // Store basic block data
        blockData[blockNumber] = BlockValidation({
            blockNumber: blockNumber,
            blockHash: blockhash(blockNumber - 1), // Previous block hash for randomness
            gasPrice: tx.gasprice,
            timestamp: block.timestamp,
            transactionCount: 0, // Would be set by off-chain oracle
            isHighMEVRisk: false
        });

        // Calculate MEV metrics
        MEVMetrics storage metrics = blockMEVData[blockNumber];
        
        // Calculate average gas price (simplified)
        if (blockNumber > 1) {
            MEVMetrics memory prevMetrics = blockMEVData[blockNumber - 1];
            metrics.avgGasPrice = (prevMetrics.avgGasPrice + tx.gasprice) / 2;
            
            // Detect gas spikes
            if (prevMetrics.avgGasPrice > 0) {
                uint256 increase = tx.gasprice > prevMetrics.avgGasPrice ?
                    ((tx.gasprice - prevMetrics.avgGasPrice) * 10000) / prevMetrics.avgGasPrice : 0;
                
                if (increase > GAS_SPIKE_THRESHOLD) {
                    metrics.gasSpike = increase;
                    metrics.frontRunningDetected = true;
                }
            }
        } else {
            metrics.avgGasPrice = tx.gasprice;
        }

        // Update user's recent blocks
        _updateUserBlockHistory(msg.sender, blockNumber);
    }

    function _updateUserBlockHistory(address user, uint256 blockNumber) internal {
        uint256[] storage userBlocks = _userRecentBlocks[user];
        userBlocks.push(blockNumber);
        
        // Keep only recent blocks
        if (userBlocks.length > BLOCK_HISTORY_LIMIT) {
            // Remove oldest block
            for (uint256 i = 0; i < userBlocks.length - 1; i++) {
                userBlocks[i] = userBlocks[i + 1];
            }
            userBlocks.pop();
        }
    }

    // ========================================
    // OPERATION EXECUTION
    // ========================================

    function _executeOperation(
        OperationType operation,
        uint256 tokenId,
        uint256 amount,
        uint256 value
    ) internal {
        // This would integrate with the actual DexterCompoundor contract
        // For now, we emit an event to track the execution
        
        if (operation == OperationType.COMPOUND) {
            _executeCompound(tokenId);
        } else if (operation == OperationType.SWAP) {
            _executeSwap(tokenId, amount);
        } else if (operation == OperationType.ADD_LIQUIDITY) {
            _executeAddLiquidity(tokenId, amount);
        } else if (operation == OperationType.REMOVE_LIQUIDITY) {
            _executeRemoveLiquidity(tokenId, amount);
        } else if (operation == OperationType.COLLECT_FEES) {
            _executeCollectFees(tokenId);
        } else if (operation == OperationType.REBALANCE) {
            _executeRebalance(tokenId, amount);
        }

        emit OperationExecuted(msg.sender, operation, tokenId, amount, block.number);
    }

    function _executeCompound(uint256 tokenId) internal {
        // Integration point with DexterCompoundor
        // This would call the actual compound function
        emit CompoundExecuted(msg.sender, tokenId, block.number);
    }

    function _executeSwap(uint256 tokenId, uint256 amount) internal {
        // Integration point with swap functionality
        emit SwapExecuted(msg.sender, tokenId, amount, block.number);
    }

    function _executeAddLiquidity(uint256 tokenId, uint256 amount) internal {
        // Integration point with liquidity addition
        emit LiquidityAdded(msg.sender, tokenId, amount, block.number);
    }

    function _executeRemoveLiquidity(uint256 tokenId, uint256 amount) internal {
        // Integration point with liquidity removal
        emit LiquidityRemoved(msg.sender, tokenId, amount, block.number);
    }

    function _executeCollectFees(uint256 tokenId) internal {
        // Integration point with fee collection
        emit FeesCollected(msg.sender, tokenId, block.number);
    }

    function _executeRebalance(uint256 tokenId, uint256 amount) internal {
        // Integration point with position rebalancing
        emit PositionRebalanced(msg.sender, tokenId, amount, block.number);
    }

    // ========================================
    // MEV MONITORING
    // ========================================

    function updateMEVMetrics(
        uint256 blockNumber,
        uint256 txCount,
        bool frontRunning,
        bool sandwich,
        bool backRunning,
        uint256 patterns
    ) external onlyOwner {
        MEVMetrics storage metrics = blockMEVData[blockNumber];
        
        blockData[blockNumber].transactionCount = txCount;
        blockData[blockNumber].isHighMEVRisk = frontRunning || sandwich || backRunning;
        
        metrics.frontRunningDetected = frontRunning;
        metrics.sandwichDetected = sandwich;
        metrics.backRunningDetected = backRunning;
        metrics.suspiciousPatterns = patterns;
        
        if (txCount > HIGH_MEV_TX_THRESHOLD) {
            metrics.blockUtilization = (txCount * 10000) / HIGH_MEV_TX_THRESHOLD;
        }

        emit MEVMetricsUpdated(blockNumber, frontRunning, sandwich, backRunning, patterns);
    }

    // ========================================
    // EMERGENCY FUNCTIONS
    // ========================================

    function emergencyWithdrawCommitment(bytes32 commitHash) external {
        Commitment storage commitment = commitments[commitHash];
        
        require(commitment.committer == msg.sender, "MEVProtection: Not your commitment");
        require(!commitment.executed, "MEVProtection: Already executed");
        require(
            block.number > commitment.revealDeadline,
            "MEVProtection: Reveal window still open"
        );

        uint256 value = commitment.value;
        commitment.executed = true; // Prevent re-entry

        if (value > 0) {
            payable(msg.sender).transfer(value);
        }

        emit CommitmentWithdrawn(msg.sender, commitHash, value);
    }

    // ========================================
    // VIEW FUNCTIONS
    // ========================================

    function generateCommitHash(
        address user,
        uint256 operation,
        uint256 tokenId,
        uint256 nonce,
        uint256 amount,
        bytes32 salt,
        uint256 maxGasPrice,
        uint256 deadline
    ) external pure returns (bytes32) {
        return keccak256(abi.encodePacked(
            user,
            operation,
            tokenId,
            nonce,
            amount,
            salt,
            maxGasPrice,
            deadline
        ));
    }

    function getCommitmentStatus(bytes32 commitHash) external view returns (
        bool exists,
        bool canReveal,
        bool expired,
        uint256 remainingBlocks
    ) {
        Commitment memory commitment = commitments[commitHash];
        
        exists = commitment.committer != address(0);
        if (!exists) return (false, false, false, 0);

        uint256 minRevealBlock = commitment.commitBlock + MIN_COMMIT_REVEAL_DELAY;
        canReveal = block.number >= minRevealBlock && 
                   block.number <= commitment.revealDeadline &&
                   !commitment.revealed;
        
        expired = block.number > commitment.revealDeadline;
        
        if (block.number < minRevealBlock) {
            remainingBlocks = minRevealBlock - block.number;
        } else if (block.number <= commitment.revealDeadline) {
            remainingBlocks = commitment.revealDeadline - block.number;
        } else {
            remainingBlocks = 0;
        }
    }

    function getMEVRiskLevel(uint256 blockNumber) external view returns (
        string memory riskLevel,
        bool shouldWait,
        uint256 recommendedDelay
    ) {
        MEVMetrics memory metrics = blockMEVData[blockNumber];
        
        uint256 riskScore = 0;
        if (metrics.frontRunningDetected) riskScore += 3;
        if (metrics.sandwichDetected) riskScore += 4;
        if (metrics.backRunningDetected) riskScore += 2;
        if (metrics.gasSpike > GAS_SPIKE_THRESHOLD) riskScore += 2;
        if (metrics.suspiciousPatterns >= PATTERN_THRESHOLD) riskScore += 3;

        if (riskScore >= 8) {
            return ("VERY_HIGH", true, 10);
        } else if (riskScore >= 5) {
            return ("HIGH", true, 5);
        } else if (riskScore >= 3) {
            return ("MEDIUM", true, 2);
        } else if (riskScore >= 1) {
            return ("LOW", false, 0);
        } else {
            return ("VERY_LOW", false, 0);
        }
    }

    function getUserRecentBlocks(address user) external view returns (uint256[] memory) {
        return _userRecentBlocks[user];
    }

    // ========================================
    // EVENTS
    // ========================================

    event OperationCommitted(
        address indexed user,
        bytes32 indexed commitHash,
        uint256 commitBlock,
        uint256 revealDeadline
    );

    event OperationRevealed(
        address indexed user,
        bytes32 indexed commitHash,
        OperationType operation,
        uint256 tokenId,
        uint256 amount
    );

    event OperationExecuted(
        address indexed user,
        OperationType operation,
        uint256 tokenId,
        uint256 amount,
        uint256 blockNumber
    );

    event CompoundExecuted(address indexed user, uint256 tokenId, uint256 blockNumber);
    event SwapExecuted(address indexed user, uint256 tokenId, uint256 amount, uint256 blockNumber);
    event LiquidityAdded(address indexed user, uint256 tokenId, uint256 amount, uint256 blockNumber);
    event LiquidityRemoved(address indexed user, uint256 tokenId, uint256 amount, uint256 blockNumber);
    event FeesCollected(address indexed user, uint256 tokenId, uint256 blockNumber);
    event PositionRebalanced(address indexed user, uint256 tokenId, uint256 amount, uint256 blockNumber);

    event MEVMetricsUpdated(
        uint256 indexed blockNumber,
        bool frontRunning,
        bool sandwich,
        bool backRunning,
        uint256 patterns
    );

    event CommitmentWithdrawn(address indexed user, bytes32 indexed commitHash, uint256 value);
}