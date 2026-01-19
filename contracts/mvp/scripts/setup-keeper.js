/**
 * Keeper Setup Script for DexterMVP
 *
 * Usage:
 *   npx hardhat run scripts/setup-keeper.js --network base-sepolia
 *
 * Required Environment Variables:
 *   DEPLOYER_PRIVATE_KEY - Private key of contract owner
 *   KEEPER_ADDRESS - Address to authorize as keeper
 *   DEXTERMVP_ADDRESS - Deployed DexterMVP contract address
 */

const { ethers, network } = require("hardhat");
const fs = require("fs");

async function main() {
  console.log("=".repeat(60));
  console.log("DexterMVP Keeper Setup");
  console.log("=".repeat(60));

  const networkName = network.name;
  console.log(`\nNetwork: ${networkName}`);

  // Get deployer (owner)
  const [owner] = await ethers.getSigners();
  console.log(`Owner: ${owner.address}`);

  // Load deployment info
  const deploymentPath = `./deployments/${networkName}.json`;
  if (!fs.existsSync(deploymentPath)) {
    console.error(`\nERROR: No deployment found at ${deploymentPath}`);
    console.error("Please run deploy.js first.");
    process.exit(1);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, "utf8"));
  const dexterMVPAddress = deployment.contracts.DexterMVP;

  console.log(`DexterMVP Address: ${dexterMVPAddress}`);

  // Get keeper address from environment or use a default for testing
  const keeperAddress = process.env.KEEPER_ADDRESS || owner.address;
  console.log(`Keeper Address: ${keeperAddress}`);

  // Connect to contract
  const DexterMVP = await ethers.getContractFactory("DexterMVP");
  const dexterMVP = DexterMVP.attach(dexterMVPAddress);

  // Check if already authorized
  const isAuthorized = await dexterMVP.authorizedKeepers(keeperAddress);
  if (isAuthorized) {
    console.log(`\nKeeper ${keeperAddress} is already authorized.`);
  } else {
    console.log(`\nAuthorizing keeper ${keeperAddress}...`);
    const tx = await dexterMVP.setKeeperAuthorization(keeperAddress, true);
    await tx.wait();
    console.log(`Keeper authorized. TX: ${tx.hash}`);
  }

  // Configure TWAP settings (recommended defaults)
  console.log("\nConfiguring TWAP protection...");
  const twapPeriod = await dexterMVP.twapPeriod();
  const maxTickDiff = await dexterMVP.maxTickDifference();
  const twapEnabled = await dexterMVP.twapProtectionEnabled();

  console.log(`  TWAP Period: ${twapPeriod} seconds`);
  console.log(`  Max Tick Difference: ${maxTickDiff}`);
  console.log(`  TWAP Protection Enabled: ${twapEnabled}`);

  // Summary
  console.log("\n" + "=".repeat(60));
  console.log("SETUP COMPLETE");
  console.log("=".repeat(60));

  console.log("\nContract Configuration:");
  console.log(`  Owner: ${await dexterMVP.owner()}`);
  console.log(`  Paused: ${await dexterMVP.paused()}`);
  console.log(`  Keeper (${keeperAddress}): ${await dexterMVP.authorizedKeepers(keeperAddress)}`);

  console.log("\nKeeper can now call:");
  console.log("  - executeCompound(tokenId)");
  console.log("  - executeRebalance(tokenId)");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
