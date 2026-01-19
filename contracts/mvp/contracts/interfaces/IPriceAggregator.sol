// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IPriceAggregator
 * @notice Interface for multi-oracle price aggregation
 * @dev Used by DexterMVP for USD conversion of fees
 */
interface IPriceAggregator {
    /**
     * @notice Get validated price with multi-oracle consensus
     * @param token0 First token address
     * @param token1 Second token address
     * @return price Validated price in 18 decimals
     * @return confidence Confidence level (0-100)
     * @return isValid Whether price passed all validations
     */
    function getValidatedPrice(
        address token0,
        address token1
    ) external view returns (uint256 price, uint256 confidence, bool isValid);

    /**
     * @notice Check if price feed is healthy
     * @param token0 First token address
     * @param token1 Second token address
     * @return isHealthy Whether the price feed is functioning properly
     * @return lastUpdate Timestamp of last successful price update
     */
    function checkPriceFeedHealth(
        address token0,
        address token1
    ) external view returns (bool isHealthy, uint256 lastUpdate);
}
