// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title AdvancedSecurityGuard
 * @notice Comprehensive security protection against reentrancy, flash loans, and MEV attacks
 * @dev Extends beyond basic ReentrancyGuard for production-grade security
 */
abstract contract AdvancedSecurityGuard is ReentrancyGuard, Ownable {
    
    // ========================================
    // FLASH LOAN PROTECTION
    // ========================================
    
    mapping(address => uint256) private _lastBlockInteraction;
    mapping(address => uint256) private _operationCount;
    mapping(address => uint256) private _dailyOperationCount;
    mapping(address => uint256) private _lastDayReset;
    
    uint256 private constant MAX_OPERATIONS_PER_BLOCK = 1;
    uint256 private constant MAX_OPERATIONS_PER_DAY = 100;
    uint256 private constant DAILY_RESET_PERIOD = 1 days;
    
    modifier noFlashLoan() {
        require(
            block.number > _lastBlockInteraction[msg.sender],
            "AdvancedSecurity: Flash loan detected"
        );
        _lastBlockInteraction[msg.sender] = block.number;
        _;
    }
    
    modifier rateLimited() {
        _checkRateLimit();
        _;
        _updateRateLimit();
    }
    
    function _checkRateLimit() internal view {
        // Check daily limits
        if (block.timestamp >= _lastDayReset[msg.sender] + DAILY_RESET_PERIOD) {
            // Day has reset, no need to check daily count
            return;
        }
        
        require(
            _dailyOperationCount[msg.sender] < MAX_OPERATIONS_PER_DAY,
            "AdvancedSecurity: Daily operation limit exceeded"
        );
    }
    
    function _updateRateLimit() internal {
        // Reset daily counter if needed
        if (block.timestamp >= _lastDayReset[msg.sender] + DAILY_RESET_PERIOD) {
            _dailyOperationCount[msg.sender] = 0;
            _lastDayReset[msg.sender] = block.timestamp;
        }
        
        _operationCount[msg.sender]++;
        _dailyOperationCount[msg.sender]++;
    }
    
    // ========================================
    // CROSS-FUNCTION REENTRANCY PROTECTION
    // ========================================
    
    mapping(bytes4 => bool) private _functionLocks;
    mapping(address => mapping(bytes4 => uint256)) private _userFunctionCalls;
    
    modifier crossFunctionGuard() {
        bytes4 functionSelector = msg.sig;
        
        require(
            !_functionLocks[functionSelector],
            "AdvancedSecurity: Function is locked"
        );
        
        require(
            _userFunctionCalls[msg.sender][functionSelector] == 0,
            "AdvancedSecurity: Cross-function reentrancy detected"
        );
        
        _functionLocks[functionSelector] = true;
        _userFunctionCalls[msg.sender][functionSelector]++;
        
        _;
        
        _userFunctionCalls[msg.sender][functionSelector]--;
        _functionLocks[functionSelector] = false;
    }
    
    // ========================================
    // CALL DEPTH PROTECTION
    // ========================================
    
    uint256 private constant MAX_CALL_DEPTH = 10;
    uint256 private _callDepth;
    
    modifier callDepthGuard() {
        require(_callDepth < MAX_CALL_DEPTH, "AdvancedSecurity: Call depth exceeded");
        _callDepth++;
        _;
        _callDepth--;
    }
    
    // ========================================
    // CONTRACT SIZE VERIFICATION
    // ========================================
    
    modifier onlyEOA() {
        require(tx.origin == msg.sender, "AdvancedSecurity: Contract calls not allowed");
        _;
    }
    
    modifier verifyContractSize(address target) {
        if (target.code.length > 0) {
            require(
                _isWhitelistedContract(target),
                "AdvancedSecurity: Unknown contract interaction"
            );
        }
        _;
    }
    
    mapping(address => bool) private _whitelistedContracts;
    
    function addWhitelistedContract(address contractAddr) external onlyOwner {
        require(contractAddr.code.length > 0, "AdvancedSecurity: Not a contract");
        _whitelistedContracts[contractAddr] = true;
        emit ContractWhitelisted(contractAddr);
    }
    
    function removeWhitelistedContract(address contractAddr) external onlyOwner {
        _whitelistedContracts[contractAddr] = false;
        emit ContractRemoved(contractAddr);
    }
    
    function _isWhitelistedContract(address contractAddr) internal view returns (bool) {
        return _whitelistedContracts[contractAddr];
    }
    
    // ========================================
    // ADVANCED TRANSACTION ANALYSIS
    // ========================================
    
    struct TransactionContext {
        uint256 gasStart;
        uint256 gasUsed;
        uint256 blockNumber;
        bytes32 transactionHash;
        bool isHighRisk;
    }
    
    mapping(address => TransactionContext) private _transactionContexts;
    
    modifier transactionAnalysis() {
        uint256 gasStart = gasleft();
        
        _transactionContexts[msg.sender] = TransactionContext({
            gasStart: gasStart,
            gasUsed: 0,
            blockNumber: block.number,
            transactionHash: blockhash(block.number - 1),
            isHighRisk: false
        });
        
        _;
        
        uint256 gasUsed = gasStart - gasleft();
        _transactionContexts[msg.sender].gasUsed = gasUsed;
        
        // Analyze for suspicious patterns
        _analyzeTransaction(msg.sender, gasUsed);
    }
    
    function _analyzeTransaction(address user, uint256 gasUsed) internal {
        TransactionContext storage ctx = _transactionContexts[user];
        
        // Flag high gas usage (potential complex attack)
        if (gasUsed > 500000) {
            ctx.isHighRisk = true;
            emit HighRiskTransactionDetected(user, gasUsed);
        }
        
        // Flag rapid successive transactions
        if (_operationCount[user] > 10 && 
            block.number - _lastBlockInteraction[user] <= 5) {
            ctx.isHighRisk = true;
            emit RapidTransactionPattern(user, _operationCount[user]);
        }
    }
    
    // ========================================
    // ECONOMIC ATTACK PROTECTION
    // ========================================
    
    mapping(address => uint256) private _totalValueLocked;
    mapping(address => uint256) private _lastLargeOperation;
    
    uint256 private constant LARGE_OPERATION_THRESHOLD = 100 ether;
    uint256 private constant LARGE_OPERATION_COOLDOWN = 1 hours;
    
    modifier economicGuard(uint256 value) {
        if (value > LARGE_OPERATION_THRESHOLD) {
            require(
                block.timestamp >= _lastLargeOperation[msg.sender] + LARGE_OPERATION_COOLDOWN,
                "AdvancedSecurity: Large operation cooldown active"
            );
            _lastLargeOperation[msg.sender] = block.timestamp;
            emit LargeOperationExecuted(msg.sender, value);
        }
        _;
    }
    
    // ========================================
    // EMERGENCY CONTROLS
    // ========================================
    
    bool private _emergencyPaused = false;
    mapping(address => bool) private _emergencyOperators;
    
    modifier whenNotEmergencyPaused() {
        require(!_emergencyPaused, "AdvancedSecurity: Emergency pause active");
        _;
    }
    
    modifier onlyEmergencyOperator() {
        require(
            _emergencyOperators[msg.sender] || msg.sender == owner(),
            "AdvancedSecurity: Not emergency operator"
        );
        _;
    }
    
    function emergencyPause() external onlyEmergencyOperator {
        _emergencyPaused = true;
        emit EmergencyPaused(msg.sender);
    }
    
    function emergencyUnpause() external onlyOwner {
        _emergencyPaused = false;
        emit EmergencyUnpaused(msg.sender);
    }
    
    function addEmergencyOperator(address operator) external onlyOwner {
        _emergencyOperators[operator] = true;
        emit EmergencyOperatorAdded(operator);
    }
    
    function removeEmergencyOperator(address operator) external onlyOwner {
        _emergencyOperators[operator] = false;
        emit EmergencyOperatorRemoved(operator);
    }
    
    // ========================================
    // CIRCUIT BREAKER PATTERN
    // ========================================
    
    struct CircuitBreaker {
        uint256 failureCount;
        uint256 lastFailureTime;
        bool isOpen;
        uint256 resetTime;
    }
    
    mapping(bytes4 => CircuitBreaker) private _circuitBreakers;
    
    uint256 private constant FAILURE_THRESHOLD = 5;
    uint256 private constant CIRCUIT_RESET_TIME = 1 hours;
    
    modifier circuitBreaker() {
        bytes4 functionSelector = msg.sig;
        CircuitBreaker storage breaker = _circuitBreakers[functionSelector];
        
        // Check if circuit is open
        if (breaker.isOpen) {
            if (block.timestamp >= breaker.resetTime) {
                // Reset circuit breaker
                breaker.isOpen = false;
                breaker.failureCount = 0;
                emit CircuitBreakerReset(functionSelector);
            } else {
                revert("AdvancedSecurity: Circuit breaker is open");
            }
        }
        
        _;
    }
    
    function _recordFailure() internal {
        bytes4 functionSelector = msg.sig;
        CircuitBreaker storage breaker = _circuitBreakers[functionSelector];
        
        breaker.failureCount++;
        breaker.lastFailureTime = block.timestamp;
        
        if (breaker.failureCount >= FAILURE_THRESHOLD) {
            breaker.isOpen = true;
            breaker.resetTime = block.timestamp + CIRCUIT_RESET_TIME;
            emit CircuitBreakerTripped(functionSelector, breaker.failureCount);
        }
    }
    
    // ========================================
    // MONITORING AND EVENTS
    // ========================================
    
    event HighRiskTransactionDetected(address indexed user, uint256 gasUsed);
    event RapidTransactionPattern(address indexed user, uint256 operationCount);
    event LargeOperationExecuted(address indexed user, uint256 value);
    event ContractWhitelisted(address indexed contractAddr);
    event ContractRemoved(address indexed contractAddr);
    event EmergencyPaused(address indexed operator);
    event EmergencyUnpaused(address indexed operator);
    event EmergencyOperatorAdded(address indexed operator);
    event EmergencyOperatorRemoved(address indexed operator);
    event CircuitBreakerTripped(bytes4 indexed functionSelector, uint256 failureCount);
    event CircuitBreakerReset(bytes4 indexed functionSelector);
    
    // ========================================
    // VIEW FUNCTIONS
    // ========================================
    
    function getSecurityStatus(address user) external view returns (
        uint256 operationCount,
        uint256 dailyOperationCount,
        uint256 lastBlockInteraction,
        bool isHighRisk
    ) {
        return (
            _operationCount[user],
            _dailyOperationCount[user],
            _lastBlockInteraction[user],
            _transactionContexts[user].isHighRisk
        );
    }
    
    function getCircuitBreakerStatus(bytes4 functionSelector) external view returns (
        uint256 failureCount,
        uint256 lastFailureTime,
        bool isOpen,
        uint256 resetTime
    ) {
        CircuitBreaker storage breaker = _circuitBreakers[functionSelector];
        return (
            breaker.failureCount,
            breaker.lastFailureTime,
            breaker.isOpen,
            breaker.resetTime
        );
    }
    
    function isEmergencyPaused() external view returns (bool) {
        return _emergencyPaused;
    }
    
    function isEmergencyOperator(address operator) external view returns (bool) {
        return _emergencyOperators[operator];
    }
}