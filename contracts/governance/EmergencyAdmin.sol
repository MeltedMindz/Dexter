// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/Multicall.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/// @title EmergencyAdmin
/// @notice Emergency administration functions with time-locked controls
/// @dev Based on Revert Finance's emergency management patterns
contract EmergencyAdmin is AccessControl, Pausable, Multicall {
    using SafeERC20 for IERC20;

    bytes32 public constant EMERGENCY_ADMIN_ROLE = keccak256("EMERGENCY_ADMIN_ROLE");
    bytes32 public constant TIMELOCK_ADMIN_ROLE = keccak256("TIMELOCK_ADMIN_ROLE");
    bytes32 public constant GUARDIAN_ROLE = keccak256("GUARDIAN_ROLE");
    
    uint256 public constant EMERGENCY_DELAY = 24 hours;
    uint256 public constant MAX_EMERGENCY_DELAY = 7 days;
    
    struct TimeLockItem {
        bytes32 id;
        address target;
        bytes data;
        uint256 executeTime;
        bool executed;
        string description;
    }
    
    mapping(bytes32 => TimeLockItem) public timeLockItems;
    mapping(address => bool) public emergencyPaused;
    
    uint256 public emergencyDelay = EMERGENCY_DELAY;
    bool public globalEmergencyMode = false;
    
    event EmergencyActionScheduled(
        bytes32 indexed id,
        address indexed target,
        bytes data,
        uint256 executeTime,
        string description
    );
    
    event EmergencyActionExecuted(
        bytes32 indexed id,
        address indexed target,
        bool success,
        bytes returnData
    );
    
    event EmergencyActionCancelled(bytes32 indexed id);
    event ContractEmergencyPaused(address indexed target);
    event ContractEmergencyUnpaused(address indexed target);
    event GlobalEmergencyModeToggled(bool enabled);
    event EmergencyDelayUpdated(uint256 oldDelay, uint256 newDelay);
    
    error UnauthorizedAccess();
    error ActionNotReady();
    error ActionAlreadyExecuted();
    error ActionNotFound();
    error InvalidDelay();
    error ExecutionFailed();
    
    constructor(address admin, address[] memory emergencyAdmins, address[] memory guardians) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(TIMELOCK_ADMIN_ROLE, admin);
        
        for (uint256 i = 0; i < emergencyAdmins.length; i++) {
            _grantRole(EMERGENCY_ADMIN_ROLE, emergencyAdmins[i]);
        }
        
        for (uint256 i = 0; i < guardians.length; i++) {
            _grantRole(GUARDIAN_ROLE, guardians[i]);
        }
    }
    
    /// @notice Schedule an emergency action with timelock
    /// @param target Target contract address
    /// @param data Encoded function call data
    /// @param description Human-readable description
    /// @return actionId Unique identifier for the scheduled action
    function scheduleEmergencyAction(
        address target,
        bytes calldata data,
        string calldata description
    ) external onlyRole(EMERGENCY_ADMIN_ROLE) returns (bytes32 actionId) {
        actionId = keccak256(abi.encodePacked(target, data, block.timestamp, description));
        
        require(timeLockItems[actionId].executeTime == 0, "Action already scheduled");
        
        uint256 executeTime = block.timestamp + emergencyDelay;
        
        timeLockItems[actionId] = TimeLockItem({
            id: actionId,
            target: target,
            data: data,
            executeTime: executeTime,
            executed: false,
            description: description
        });
        
        emit EmergencyActionScheduled(actionId, target, data, executeTime, description);
    }
    
    /// @notice Execute a scheduled emergency action
    /// @param actionId The action identifier
    /// @return success Whether the execution succeeded
    /// @return returnData Return data from the function call
    function executeEmergencyAction(bytes32 actionId)
        external
        onlyRole(EMERGENCY_ADMIN_ROLE)
        returns (bool success, bytes memory returnData)
    {
        TimeLockItem storage item = timeLockItems[actionId];
        
        if (item.executeTime == 0) {
            revert ActionNotFound();
        }
        
        if (item.executed) {
            revert ActionAlreadyExecuted();
        }
        
        if (block.timestamp < item.executeTime) {
            revert ActionNotReady();
        }
        
        item.executed = true;
        
        (success, returnData) = item.target.call(item.data);
        
        emit EmergencyActionExecuted(actionId, item.target, success, returnData);
        
        if (!success) {
            revert ExecutionFailed();
        }
    }
    
    /// @notice Cancel a scheduled emergency action
    /// @param actionId The action identifier
    function cancelEmergencyAction(bytes32 actionId) external onlyRole(TIMELOCK_ADMIN_ROLE) {
        TimeLockItem storage item = timeLockItems[actionId];
        
        if (item.executeTime == 0) {
            revert ActionNotFound();
        }
        
        if (item.executed) {
            revert ActionAlreadyExecuted();
        }
        
        delete timeLockItems[actionId];
        
        emit EmergencyActionCancelled(actionId);
    }
    
    /// @notice Immediately pause a contract (guardian power)
    /// @param target Contract to pause
    function emergencyPause(address target) external onlyRole(GUARDIAN_ROLE) {
        emergencyPaused[target] = true;
        
        // Try to call pause function if it exists
        (bool success, ) = target.call(abi.encodeWithSignature("pause()"));
        
        emit ContractEmergencyPaused(target);
    }
    
    /// @notice Unpause a contract
    /// @param target Contract to unpause
    function emergencyUnpause(address target) external onlyRole(EMERGENCY_ADMIN_ROLE) {
        emergencyPaused[target] = false;
        
        // Try to call unpause function if it exists
        (bool success, ) = target.call(abi.encodeWithSignature("unpause()"));
        
        emit ContractEmergencyUnpaused(target);
    }
    
    /// @notice Toggle global emergency mode
    /// @param enabled Whether to enable global emergency mode
    function setGlobalEmergencyMode(bool enabled) external onlyRole(GUARDIAN_ROLE) {
        globalEmergencyMode = enabled;
        
        if (enabled) {
            _pause();
        } else {
            _unpause();
        }
        
        emit GlobalEmergencyModeToggled(enabled);
    }
    
    /// @notice Update emergency delay
    /// @param newDelay New delay in seconds
    function setEmergencyDelay(uint256 newDelay) external onlyRole(TIMELOCK_ADMIN_ROLE) {
        if (newDelay > MAX_EMERGENCY_DELAY) {
            revert InvalidDelay();
        }
        
        uint256 oldDelay = emergencyDelay;
        emergencyDelay = newDelay;
        
        emit EmergencyDelayUpdated(oldDelay, newDelay);
    }
    
    /// @notice Batch execute emergency actions
    /// @param actionIds Array of action identifiers
    function batchExecuteEmergencyActions(bytes32[] calldata actionIds)
        external
        onlyRole(EMERGENCY_ADMIN_ROLE)
    {
        for (uint256 i = 0; i < actionIds.length; i++) {
            (bool success, ) = this.executeEmergencyAction(actionIds[i]);
            // Continue execution even if one fails
        }
    }
    
    /// @notice Emergency token recovery
    /// @param token Token address (address(0) for ETH)
    /// @param amount Amount to recover
    /// @param to Recipient address
    function emergencyRecoverToken(
        address token,
        uint256 amount,
        address to
    ) external onlyRole(EMERGENCY_ADMIN_ROLE) {
        if (token == address(0)) {
            // Recover ETH
            (bool success, ) = payable(to).call{value: amount}("");
            require(success, "ETH transfer failed");
        } else {
            // Recover ERC20
            IERC20(token).safeTransfer(to, amount);
        }
    }
    
    /// @notice Check if a contract is under emergency pause
    /// @param target Contract address
    /// @return isPaused Whether the contract is paused
    function isEmergencyPaused(address target) external view returns (bool isPaused) {
        return emergencyPaused[target] || globalEmergencyMode;
    }
    
    /// @notice Get timelock item details
    /// @param actionId Action identifier
    /// @return item TimeLockItem struct
    function getTimeLockItem(bytes32 actionId) external view returns (TimeLockItem memory item) {
        return timeLockItems[actionId];
    }
    
    /// @notice Check if action can be executed
    /// @param actionId Action identifier
    /// @return canExecute Whether action can be executed
    /// @return timeRemaining Time remaining until execution (0 if ready)
    function canExecuteAction(bytes32 actionId) 
        external 
        view 
        returns (bool canExecute, uint256 timeRemaining) 
    {
        TimeLockItem storage item = timeLockItems[actionId];
        
        if (item.executeTime == 0 || item.executed) {
            return (false, 0);
        }
        
        if (block.timestamp >= item.executeTime) {
            return (true, 0);
        } else {
            return (false, item.executeTime - block.timestamp);
        }
    }
    
    /// @notice Emergency function to upgrade contract implementation
    /// @param proxy Proxy contract address
    /// @param newImplementation New implementation address
    function emergencyUpgrade(address proxy, address newImplementation) 
        external 
        onlyRole(EMERGENCY_ADMIN_ROLE) 
    {
        // This would integrate with proxy upgrade mechanisms
        // Implementation depends on specific proxy pattern used
        (bool success, ) = proxy.call(
            abi.encodeWithSignature("upgradeTo(address)", newImplementation)
        );
        require(success, "Upgrade failed");
    }
    
    /// @notice Enable receiving ETH for emergency recovery
    receive() external payable {}
    
    /// @notice Fallback function for emergency calls
    fallback() external payable {}
    
    /// @notice Modifier to check emergency status
    modifier notInEmergency() {
        require(!globalEmergencyMode, "Global emergency mode active");
        _;
    }
    
    /// @notice Modifier to check if caller has emergency admin role
    modifier onlyEmergencyAdmin() {
        if (!hasRole(EMERGENCY_ADMIN_ROLE, msg.sender)) {
            revert UnauthorizedAccess();
        }
        _;
    }
    
    /// @notice Grant emergency admin role with time lock
    /// @param account Account to grant role to
    function grantEmergencyAdminRole(address account) external {
        bytes memory data = abi.encodeWithSignature("grantRole(bytes32,address)", EMERGENCY_ADMIN_ROLE, account);
        scheduleEmergencyAction(address(this), data, "Grant Emergency Admin Role");
    }
    
    /// @notice Revoke emergency admin role with time lock
    /// @param account Account to revoke role from
    function revokeEmergencyAdminRole(address account) external {
        bytes memory data = abi.encodeWithSignature("revokeRole(bytes32,address)", EMERGENCY_ADMIN_ROLE, account);
        scheduleEmergencyAction(address(this), data, "Revoke Emergency Admin Role");
    }
}