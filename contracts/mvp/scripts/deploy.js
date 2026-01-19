/**
 * Deployment Script for DexterMVP Contracts
 *
 * Usage:
 *   npx hardhat run scripts/deploy.js --network base-sepolia
 *   npx hardhat run scripts/deploy.js --network base-mainnet
 *
 * Required Environment Variables:
 *   DEPLOYER_PRIVATE_KEY - Private key of deployer wallet
 *   BASESCAN_API_KEY - For contract verification (optional)
 *
 * Base Network Addresses:
 *   Uniswap V3 Position Manager: 0x03a520b32C04BF3bEEf7BEb72E919cf822Ed34f1
 *   Uniswap V3 Swap Router: 0x2626664c2603336E57B271c5C0b26F421741e481
 *   WETH (Base): 0x4200000000000000000000000000000000000006
 */

const { ethers, network, run } = require("hardhat");

// Base Network Contract Addresses
const ADDRESSES = {
  "base-mainnet": {
    positionManager: "0x03a520b32C04BF3bEEf7BEb72E919cf822Ed34f1",
    swapRouter: "0x2626664c2603336E57B271c5C0b26F421741e481",
    weth: "0x4200000000000000000000000000000000000006"
  },
  "base-sepolia": {
    // Sepolia testnet addresses - using same addresses as mainnet for Uniswap
    positionManager: "0x03a520b32C04BF3bEEf7BEb72E919cf822Ed34f1",
    swapRouter: "0x2626664c2603336E57B271c5C0b26F421741e481",
    weth: "0x4200000000000000000000000000000000000006"
  },
  "hardhat": {
    // For local testing, these will be overridden or mocked
    positionManager: ethers.ZeroAddress,
    swapRouter: ethers.ZeroAddress,
    weth: "0x4200000000000000000000000000000000000006"
  }
};

async function main() {
  console.log("=".repeat(60));
  console.log("DexterMVP Deployment Script");
  console.log("=".repeat(60));

  // Get network info
  const networkName = network.name;
  console.log(`\nNetwork: ${networkName}`);
  console.log(`Chain ID: ${network.config.chainId || "local"}`);

  // Get deployer
  const [deployer] = await ethers.getSigners();
  console.log(`Deployer: ${deployer.address}`);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`Balance: ${ethers.formatEther(balance)} ETH`);

  if (balance === 0n) {
    console.error("\nERROR: Deployer has no ETH. Please fund the wallet first.");
    process.exit(1);
  }

  // Get addresses for this network
  const addresses = ADDRESSES[networkName] || ADDRESSES["hardhat"];
  console.log("\nUsing addresses:");
  console.log(`  Position Manager: ${addresses.positionManager}`);
  console.log(`  Swap Router: ${addresses.swapRouter}`);
  console.log(`  WETH: ${addresses.weth}`);

  // Deploy DexterMVP
  console.log("\n" + "-".repeat(60));
  console.log("Deploying DexterMVP...");

  const DexterMVP = await ethers.getContractFactory("DexterMVP");
  const dexterMVP = await DexterMVP.deploy(
    addresses.positionManager,
    addresses.swapRouter,
    addresses.weth
  );

  await dexterMVP.waitForDeployment();
  const dexterMVPAddress = await dexterMVP.getAddress();

  console.log(`DexterMVP deployed to: ${dexterMVPAddress}`);

  // Deploy BinRebalancer
  console.log("\n" + "-".repeat(60));
  console.log("Deploying BinRebalancer...");

  const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
  const binRebalancer = await BinRebalancer.deploy(
    addresses.positionManager,
    addresses.swapRouter,
    addresses.weth
  );

  await binRebalancer.waitForDeployment();
  const binRebalancerAddress = await binRebalancer.getAddress();

  console.log(`BinRebalancer deployed to: ${binRebalancerAddress}`);

  // Deploy UltraFrequentCompounder
  console.log("\n" + "-".repeat(60));
  console.log("Deploying UltraFrequentCompounder...");

  const UltraFrequentCompounder = await ethers.getContractFactory("UltraFrequentCompounder");
  const ultraFrequentCompounder = await UltraFrequentCompounder.deploy(
    addresses.positionManager,
    addresses.swapRouter,
    addresses.weth
  );

  await ultraFrequentCompounder.waitForDeployment();
  const ultraFrequentCompounderAddress = await ultraFrequentCompounder.getAddress();

  console.log(`UltraFrequentCompounder deployed to: ${ultraFrequentCompounderAddress}`);

  // Summary
  console.log("\n" + "=".repeat(60));
  console.log("DEPLOYMENT SUMMARY");
  console.log("=".repeat(60));
  console.log(`\nNetwork: ${networkName}`);
  console.log(`\nDeployed Contracts:`);
  console.log(`  DexterMVP:              ${dexterMVPAddress}`);
  console.log(`  BinRebalancer:          ${binRebalancerAddress}`);
  console.log(`  UltraFrequentCompounder: ${ultraFrequentCompounderAddress}`);

  // Save deployment info
  const deploymentInfo = {
    network: networkName,
    chainId: network.config.chainId,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    contracts: {
      DexterMVP: dexterMVPAddress,
      BinRebalancer: binRebalancerAddress,
      UltraFrequentCompounder: ultraFrequentCompounderAddress
    },
    dependencies: addresses
  };

  const fs = require("fs");
  const deploymentPath = `./deployments/${networkName}.json`;

  // Create deployments directory if it doesn't exist
  if (!fs.existsSync("./deployments")) {
    fs.mkdirSync("./deployments");
  }

  fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`\nDeployment info saved to: ${deploymentPath}`);

  // Verify contracts on explorer (if not local)
  if (networkName !== "hardhat" && networkName !== "localhost") {
    console.log("\n" + "-".repeat(60));
    console.log("Verifying contracts on BaseScan...");
    console.log("(This may take a few minutes)");

    try {
      // Wait for block confirmations
      console.log("\nWaiting for block confirmations...");
      await new Promise(resolve => setTimeout(resolve, 30000)); // 30 seconds

      // Verify DexterMVP
      console.log("\nVerifying DexterMVP...");
      await run("verify:verify", {
        address: dexterMVPAddress,
        constructorArguments: [
          addresses.positionManager,
          addresses.swapRouter,
          addresses.weth
        ]
      });

      // Verify BinRebalancer
      console.log("\nVerifying BinRebalancer...");
      await run("verify:verify", {
        address: binRebalancerAddress,
        constructorArguments: [
          addresses.positionManager,
          addresses.swapRouter,
          addresses.weth
        ]
      });

      // Verify UltraFrequentCompounder
      console.log("\nVerifying UltraFrequentCompounder...");
      await run("verify:verify", {
        address: ultraFrequentCompounderAddress,
        constructorArguments: [
          addresses.positionManager,
          addresses.swapRouter,
          addresses.weth
        ]
      });

      console.log("\nAll contracts verified successfully!");

    } catch (error) {
      console.error("\nVerification failed:", error.message);
      console.log("You can verify manually later using:");
      console.log(`  npx hardhat verify --network ${networkName} ${dexterMVPAddress} ${addresses.positionManager} ${addresses.swapRouter} ${addresses.weth}`);
    }
  }

  console.log("\n" + "=".repeat(60));
  console.log("DEPLOYMENT COMPLETE");
  console.log("=".repeat(60));

  // Post-deployment instructions
  console.log("\nNext Steps:");
  console.log("1. Authorize keeper addresses:");
  console.log(`   await dexterMVP.setKeeperAuthorization(keeperAddress, true)`);
  console.log("2. Set price aggregator (if available):");
  console.log(`   await dexterMVP.setPriceAggregator(aggregatorAddress)`);
  console.log("3. Configure TWAP settings (optional):");
  console.log(`   await dexterMVP.setTWAPConfig(period, maxTickDiff, enabled)`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
