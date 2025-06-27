// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title GasOptimizationLib
 * @notice Library containing gas optimization patterns and utilities
 * @dev Provides reusable gas optimization techniques for Dexter Protocol contracts
 */
library GasOptimizationLib {
    
    // ============ CONSTANTS ============
    
    uint256 private constant WORD_SIZE = 32;
    uint256 private constant WORD_BITS = 256;
    
    // ============ PACKED STORAGE UTILITIES ============
    
    /**
     * @notice Pack two uint128 values into a single uint256
     * @param a First value (lower 128 bits)
     * @param b Second value (upper 128 bits)
     * @return packed Packed uint256
     */
    function packUint128(uint128 a, uint128 b) internal pure returns (uint256 packed) {
        assembly {
            packed := or(a, shl(128, b))
        }
    }
    
    /**
     * @notice Unpack two uint128 values from a single uint256
     * @param packed Packed uint256
     * @return a First value (lower 128 bits)
     * @return b Second value (upper 128 bits)
     */
    function unpackUint128(uint256 packed) internal pure returns (uint128 a, uint128 b) {
        assembly {
            a := and(packed, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
            b := shr(128, packed)
        }
    }
    
    /**
     * @notice Pack four uint64 values into a single uint256
     * @param a First value (bits 0-63)
     * @param b Second value (bits 64-127)
     * @param c Third value (bits 128-191)
     * @param d Fourth value (bits 192-255)
     * @return packed Packed uint256
     */
    function packUint64x4(uint64 a, uint64 b, uint64 c, uint64 d) internal pure returns (uint256 packed) {
        assembly {
            packed := or(a, shl(64, b))
            packed := or(packed, shl(128, c))
            packed := or(packed, shl(192, d))
        }
    }
    
    /**
     * @notice Unpack four uint64 values from a single uint256
     * @param packed Packed uint256
     * @return a First value (bits 0-63)
     * @return b Second value (bits 64-127)
     * @return c Third value (bits 128-191)
     * @return d Fourth value (bits 192-255)
     */
    function unpackUint64x4(uint256 packed) internal pure returns (uint64 a, uint64 b, uint64 c, uint64 d) {
        assembly {
            a := and(packed, 0xFFFFFFFFFFFFFFFF)
            b := and(shr(64, packed), 0xFFFFFFFFFFFFFFFF)
            c := and(shr(128, packed), 0xFFFFFFFFFFFFFFFF)
            d := shr(192, packed)
        }
    }
    
    /**
     * @notice Pack eight uint32 values into a single uint256
     * @param values Array of 8 uint32 values
     * @return packed Packed uint256
     */
    function packUint32x8(uint32[8] memory values) internal pure returns (uint256 packed) {
        assembly {
            let ptr := add(values, 0x20)
            for { let i := 0 } lt(i, 8) { i := add(i, 1) } {
                let value := mload(add(ptr, mul(i, 0x20)))
                packed := or(packed, shl(mul(i, 32), value))
            }
        }
    }
    
    /**
     * @notice Unpack eight uint32 values from a single uint256
     * @param packed Packed uint256
     * @return values Array of 8 uint32 values
     */
    function unpackUint32x8(uint256 packed) internal pure returns (uint32[8] memory values) {
        assembly {
            let ptr := add(values, 0x20)
            for { let i := 0 } lt(i, 8) { i := add(i, 1) } {
                let value := and(shr(mul(i, 32), packed), 0xFFFFFFFF)
                mstore(add(ptr, mul(i, 0x20)), value)
            }
        }
    }

    // ============ BITWISE OPERATIONS ============
    
    /**
     * @notice Set a bit at a specific position
     * @param bitmap The bitmap to modify
     * @param position Bit position (0-255)
     * @param value Boolean value to set
     * @return newBitmap Modified bitmap
     */
    function setBit(uint256 bitmap, uint8 position, bool value) internal pure returns (uint256 newBitmap) {
        assembly {
            let mask := shl(position, 1)
            newBitmap := and(bitmap, not(mask))
            if value {
                newBitmap := or(newBitmap, mask)
            }
        }
    }
    
    /**
     * @notice Get a bit at a specific position
     * @param bitmap The bitmap to read from
     * @param position Bit position (0-255)
     * @return value Boolean value at position
     */
    function getBit(uint256 bitmap, uint8 position) internal pure returns (bool value) {
        assembly {
            value := and(shr(position, bitmap), 1)
        }
    }
    
    /**
     * @notice Set multiple bits efficiently
     * @param bitmap The bitmap to modify
     * @param positions Array of bit positions
     * @param values Array of boolean values
     * @return newBitmap Modified bitmap
     */
    function setBits(uint256 bitmap, uint8[] memory positions, bool[] memory values) 
        internal 
        pure 
        returns (uint256 newBitmap) 
    {
        require(positions.length == values.length, "Array length mismatch");
        
        newBitmap = bitmap;
        assembly {
            let posPtr := add(positions, 0x20)
            let valPtr := add(values, 0x20)
            let len := mload(positions)
            
            for { let i := 0 } lt(i, len) { i := add(i, 1) } {
                let pos := byte(31, mload(add(posPtr, mul(i, 0x20))))
                let val := byte(31, mload(add(valPtr, mul(i, 0x20))))
                let mask := shl(pos, 1)
                
                newBitmap := and(newBitmap, not(mask))
                if val {
                    newBitmap := or(newBitmap, mask)
                }
            }
        }
    }

    // ============ ARRAY OPTIMIZATIONS ============
    
    /**
     * @notice Efficiently sum an array of uint256
     * @param array Array to sum
     * @return sum Total sum
     */
    function sumArray(uint256[] memory array) internal pure returns (uint256 sum) {
        assembly {
            let len := mload(array)
            let ptr := add(array, 0x20)
            
            for { let i := 0 } lt(i, len) { i := add(i, 1) } {
                sum := add(sum, mload(add(ptr, mul(i, 0x20))))
            }
        }
    }
    
    /**
     * @notice Find maximum value in array efficiently
     * @param array Array to search
     * @return max Maximum value
     * @return index Index of maximum value
     */
    function findMax(uint256[] memory array) internal pure returns (uint256 max, uint256 index) {
        require(array.length > 0, "Empty array");
        
        assembly {
            let len := mload(array)
            let ptr := add(array, 0x20)
            max := mload(ptr)
            index := 0
            
            for { let i := 1 } lt(i, len) { i := add(i, 1) } {
                let value := mload(add(ptr, mul(i, 0x20)))
                if gt(value, max) {
                    max := value
                    index := i
                }
            }
        }
    }
    
    /**
     * @notice Remove element from array by swapping with last element
     * @param array Array to modify
     * @param index Index to remove
     * @return newArray Modified array with last element removed
     */
    function removeBySwap(uint256[] memory array, uint256 index) 
        internal 
        pure 
        returns (uint256[] memory newArray) 
    {
        require(index < array.length, "Index out of bounds");
        
        if (array.length == 1) {
            return new uint256[](0);
        }
        
        newArray = new uint256[](array.length - 1);
        
        assembly {
            let oldPtr := add(array, 0x20)
            let newPtr := add(newArray, 0x20)
            let len := mload(array)
            let lastIndex := sub(len, 1)
            
            // Copy all elements except the one to remove
            for { let i := 0 } lt(i, len) { i := add(i, 1) } {
                if eq(i, index) {
                    // Replace with last element
                    if lt(i, lastIndex) {
                        let lastValue := mload(add(oldPtr, mul(lastIndex, 0x20)))
                        mstore(add(newPtr, mul(i, 0x20)), lastValue)
                    }
                    continue
                }
                
                let copyIndex := i
                if gt(i, index) {
                    copyIndex := sub(i, 1)
                }
                
                if lt(copyIndex, sub(len, 1)) {
                    let value := mload(add(oldPtr, mul(i, 0x20)))
                    mstore(add(newPtr, mul(copyIndex, 0x20)), value)
                }
            }
        }
    }

    // ============ MATHEMATICAL OPTIMIZATIONS ============
    
    /**
     * @notice Efficient percentage calculation with rounding
     * @param value Base value
     * @param percentage Percentage (in basis points, 10000 = 100%)
     * @return result Calculated percentage
     */
    function calculatePercentage(uint256 value, uint256 percentage) internal pure returns (uint256 result) {
        assembly {
            result := div(mul(value, percentage), 10000)
        }
    }
    
    /**
     * @notice Efficient percentage calculation with rounding up
     * @param value Base value
     * @param percentage Percentage (in basis points, 10000 = 100%)
     * @return result Calculated percentage (rounded up)
     */
    function calculatePercentageCeil(uint256 value, uint256 percentage) internal pure returns (uint256 result) {
        assembly {
            let numerator := mul(value, percentage)
            result := div(add(numerator, 9999), 10000)
        }
    }
    
    /**
     * @notice Fast integer square root using Newton's method
     * @param x Input value
     * @return result Square root
     */
    function sqrt(uint256 x) internal pure returns (uint256 result) {
        if (x == 0) return 0;
        
        assembly {
            // Initial guess
            result := x
            let xAux := x
            
            // Newton's method: x_{n+1} = (x_n + x/x_n) / 2
            if gte(xAux, 0x100000000000000000000000000000000) {
                xAux := shr(128, xAux)
                result := shl(64, result)
            }
            if gte(xAux, 0x10000000000000000) {
                xAux := shr(64, xAux)
                result := shl(32, result)
            }
            if gte(xAux, 0x100000000) {
                xAux := shr(32, xAux)
                result := shl(16, result)
            }
            if gte(xAux, 0x10000) {
                xAux := shr(16, xAux)
                result := shl(8, result)
            }
            if gte(xAux, 0x100) {
                xAux := shr(8, xAux)
                result := shl(4, result)
            }
            if gte(xAux, 0x10) {
                xAux := shr(4, xAux)
                result := shl(2, result)
            }
            if gte(xAux, 0x4) {
                result := shl(1, result)
            }
            
            // Refine with Newton's method
            result := shr(1, add(result, div(x, result)))
            result := shr(1, add(result, div(x, result)))
            result := shr(1, add(result, div(x, result)))
            result := shr(1, add(result, div(x, result)))
            result := shr(1, add(result, div(x, result)))
            result := shr(1, add(result, div(x, result)))
            result := shr(1, add(result, div(x, result)))
            
            // Ensure result <= sqrt(x)
            let resultSquared := mul(result, result)
            if gt(resultSquared, x) {
                result := sub(result, 1)
            }
        }
    }

    // ============ MEMORY OPTIMIZATIONS ============
    
    /**
     * @notice Efficiently copy bytes from one location to another
     * @param dest Destination pointer
     * @param src Source pointer
     * @param len Number of bytes to copy
     */
    function memcopy(uint256 dest, uint256 src, uint256 len) internal pure {
        assembly {
            // Copy word by word for efficiency
            for { } gte(len, 32) { len := sub(len, 32) } {
                mstore(dest, mload(src))
                dest := add(dest, 32)
                src := add(src, 32)
            }
            
            // Copy remaining bytes
            if gt(len, 0) {
                let mask := sub(shl(mul(len, 8), 1), 1)
                let srcData := and(mload(src), mask)
                let destData := and(mload(dest), not(mask))
                mstore(dest, or(destData, srcData))
            }
        }
    }
    
    /**
     * @notice Efficiently clear memory region
     * @param ptr Memory pointer
     * @param len Number of bytes to clear
     */
    function memclear(uint256 ptr, uint256 len) internal pure {
        assembly {
            // Clear word by word
            for { } gte(len, 32) { len := sub(len, 32) } {
                mstore(ptr, 0)
                ptr := add(ptr, 32)
            }
            
            // Clear remaining bytes
            if gt(len, 0) {
                let mask := not(sub(shl(mul(len, 8), 1), 1))
                let data := and(mload(ptr), mask)
                mstore(ptr, data)
            }
        }
    }

    // ============ ERROR HANDLING ============
    
    /**
     * @notice Revert with efficient error message
     * @param message Error message (max 32 bytes)
     */
    function efficientRevert(bytes32 message) internal pure {
        assembly {
            mstore(0x00, 0x08c379a000000000000000000000000000000000000000000000000000000000)
            mstore(0x04, 0x0000000000000000000000000000000000000000000000000000000000000020)
            mstore(0x24, message)
            revert(0x00, 0x44)
        }
    }
    
    /**
     * @notice Check condition and revert efficiently if false
     * @param condition Condition to check
     * @param message Error message
     */
    function efficientRequire(bool condition, bytes32 message) internal pure {
        if (!condition) {
            efficientRevert(message);
        }
    }

    // ============ GAS MEASUREMENT ============
    
    /**
     * @notice Measure gas consumed by a function call
     * @param target Contract to call
     * @param data Call data
     * @return success Whether call succeeded
     * @return returnData Return data from call
     * @return gasUsed Gas consumed
     */
    function measureGas(address target, bytes memory data) 
        internal 
        returns (bool success, bytes memory returnData, uint256 gasUsed) 
    {
        uint256 gasStart = gasleft();
        (success, returnData) = target.call(data);
        gasUsed = gasStart - gasleft();
    }

    // ============ STORAGE OPTIMIZATION HELPERS ============
    
    /**
     * @notice Pack address and boolean into single storage slot
     * @param addr Address (160 bits)
     * @param flag Boolean flag (1 bit)
     * @return packed Packed uint256
     */
    function packAddressBool(address addr, bool flag) internal pure returns (uint256 packed) {
        assembly {
            packed := or(addr, shl(160, flag))
        }
    }
    
    /**
     * @notice Unpack address and boolean from single storage slot
     * @param packed Packed uint256
     * @return addr Address (160 bits)
     * @return flag Boolean flag (1 bit)
     */
    function unpackAddressBool(uint256 packed) internal pure returns (address addr, bool flag) {
        assembly {
            addr := and(packed, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
            flag := shr(160, packed)
        }
    }
    
    /**
     * @notice Pack timestamp (uint32) and counter (uint224) into single slot
     * @param timestamp Timestamp value
     * @param counter Counter value
     * @return packed Packed uint256
     */
    function packTimestampCounter(uint32 timestamp, uint224 counter) internal pure returns (uint256 packed) {
        assembly {
            packed := or(timestamp, shl(32, counter))
        }
    }
    
    /**
     * @notice Unpack timestamp and counter from single storage slot
     * @param packed Packed uint256
     * @return timestamp Timestamp value
     * @return counter Counter value
     */
    function unpackTimestampCounter(uint256 packed) internal pure returns (uint32 timestamp, uint224 counter) {
        assembly {
            timestamp := and(packed, 0xFFFFFFFF)
            counter := shr(32, packed)
        }
    }
}