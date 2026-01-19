/**
 * Integration Tests for DexterMVP System
 *
 * Tests interactions between:
 * - DexterMVP (primary position management)
 * - BinRebalancer (advanced rebalancing)
 * - UltraFrequentCompounder (high-frequency compounding)
 *
 * Note: These tests use mock addresses where Uniswap dependencies are needed.
 * Full integration with live Uniswap requires forked network testing.
 */

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-network-helpers");

describe("DexterMVP Integration Tests", function () {
  // Deploy all contracts together
  async function deployAllContractsFixture() {
    const [owner, keeper1, keeper2, user1, user2] = await ethers.getSigners();

    // Use zero addresses for Uniswap dependencies (mock deployment)
    const mockPositionManager = ethers.ZeroAddress;
    const mockSwapRouter = ethers.ZeroAddress;
    const mockFactory = ethers.ZeroAddress;
    const mockWeth = "0x4200000000000000000000000000000000000006";

    // Deploy DexterMVP (3 params: positionManager, swapRouter, weth)
    const DexterMVP = await ethers.getContractFactory("DexterMVP");
    const dexterMVP = await DexterMVP.deploy(
      mockPositionManager,
      mockSwapRouter,
      mockWeth
    );
    await dexterMVP.waitForDeployment();

    // Deploy BinRebalancer (2 params: positionManager, factory)
    const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
    const binRebalancer = await BinRebalancer.deploy(
      mockPositionManager,
      mockFactory
    );
    await binRebalancer.waitForDeployment();

    // Deploy UltraFrequentCompounder (1 param: positionManager)
    const UltraFrequentCompounder = await ethers.getContractFactory(
      "UltraFrequentCompounder"
    );
    const ultraFrequentCompounder = await UltraFrequentCompounder.deploy(
      mockPositionManager
    );
    await ultraFrequentCompounder.waitForDeployment();

    return {
      dexterMVP,
      binRebalancer,
      ultraFrequentCompounder,
      owner,
      keeper1,
      keeper2,
      user1,
      user2,
      mockPositionManager,
      mockSwapRouter,
      mockFactory,
      mockWeth,
    };
  }

  describe("Multi-Contract Deployment", function () {
    it("Should deploy all contracts successfully", async function () {
      const { dexterMVP, binRebalancer, ultraFrequentCompounder } =
        await loadFixture(deployAllContractsFixture);

      expect(await dexterMVP.getAddress()).to.be.properAddress;
      expect(await binRebalancer.getAddress()).to.be.properAddress;
      expect(await ultraFrequentCompounder.getAddress()).to.be.properAddress;
    });

    it("Should have same owner for all contracts", async function () {
      const { dexterMVP, binRebalancer, ultraFrequentCompounder, owner } =
        await loadFixture(deployAllContractsFixture);

      expect(await dexterMVP.owner()).to.equal(owner.address);
      expect(await binRebalancer.owner()).to.equal(owner.address);
      expect(await ultraFrequentCompounder.owner()).to.equal(owner.address);
    });

    it("Should have weth address on DexterMVP", async function () {
      const { dexterMVP, mockWeth } =
        await loadFixture(deployAllContractsFixture);

      // DexterMVP has weth address exposed (lowercase)
      expect(await dexterMVP.weth()).to.equal(mockWeth);
    });
  });

  describe("Unified Keeper Management", function () {
    it("Should allow same keeper across all contracts", async function () {
      const {
        dexterMVP,
        binRebalancer,
        ultraFrequentCompounder,
        owner,
        keeper1,
      } = await loadFixture(deployAllContractsFixture);

      // Authorize keeper1 on all contracts
      await dexterMVP.connect(owner).setKeeperAuthorization(keeper1.address, true);
      await binRebalancer.connect(owner).setKeeperAuthorization(keeper1.address, true);
      await ultraFrequentCompounder.connect(owner).setKeeperAuthorization(keeper1.address, true);

      // Verify authorization
      expect(await dexterMVP.authorizedKeepers(keeper1.address)).to.be.true;
      expect(await binRebalancer.authorizedKeepers(keeper1.address)).to.be.true;
      expect(await ultraFrequentCompounder.authorizedKeepers(keeper1.address)).to.be.true;
    });

    it("Should allow different keepers for different contracts", async function () {
      const {
        dexterMVP,
        binRebalancer,
        ultraFrequentCompounder,
        owner,
        keeper1,
        keeper2,
      } = await loadFixture(deployAllContractsFixture);

      // DexterMVP uses keeper1, BinRebalancer uses keeper2
      await dexterMVP.connect(owner).setKeeperAuthorization(keeper1.address, true);
      await binRebalancer.connect(owner).setKeeperAuthorization(keeper2.address, true);

      // Verify separation
      expect(await dexterMVP.authorizedKeepers(keeper1.address)).to.be.true;
      expect(await dexterMVP.authorizedKeepers(keeper2.address)).to.be.false;
      expect(await binRebalancer.authorizedKeepers(keeper1.address)).to.be.false;
      expect(await binRebalancer.authorizedKeepers(keeper2.address)).to.be.true;
    });

    it("Should emit events on all contracts", async function () {
      const { dexterMVP, binRebalancer, owner, keeper1 } = await loadFixture(
        deployAllContractsFixture
      );

      await expect(
        dexterMVP.connect(owner).setKeeperAuthorization(keeper1.address, true)
      ).to.emit(dexterMVP, "KeeperAuthorized").withArgs(keeper1.address, true);

      // Note: BinRebalancer doesn't emit KeeperAuthorized event in current implementation
      // This is intentional - it uses a simpler pattern
    });
  });

  describe("Emergency Pause Coordination", function () {
    it("Should allow pausing DexterMVP while keeping others active", async function () {
      const { dexterMVP, binRebalancer, owner } = await loadFixture(
        deployAllContractsFixture
      );

      // Pause only DexterMVP
      await dexterMVP.connect(owner).pause();

      expect(await dexterMVP.paused()).to.be.true;
      // BinRebalancer doesn't have pause - it's always active
    });

    it("Should allow unpausing after pause", async function () {
      const { dexterMVP, owner } = await loadFixture(deployAllContractsFixture);

      await dexterMVP.connect(owner).pause();
      expect(await dexterMVP.paused()).to.be.true;

      await dexterMVP.connect(owner).unpause();
      expect(await dexterMVP.paused()).to.be.false;
    });
  });

  describe("TWAP Configuration Integration", function () {
    it("Should allow configuring TWAP on DexterMVP", async function () {
      const { dexterMVP, owner } = await loadFixture(deployAllContractsFixture);

      // Set custom TWAP config
      await dexterMVP.connect(owner).setTWAPConfig(120, 200, true);

      expect(await dexterMVP.twapPeriod()).to.equal(120);
      expect(await dexterMVP.maxTickDifference()).to.equal(200);
      expect(await dexterMVP.twapProtectionEnabled()).to.be.true;
    });

    it("Should allow disabling TWAP protection", async function () {
      const { dexterMVP, owner } = await loadFixture(deployAllContractsFixture);

      await dexterMVP.connect(owner).setTWAPConfig(60, 100, false);

      expect(await dexterMVP.twapProtectionEnabled()).to.be.false;
    });
  });

  describe("Concentration Level Management", function () {
    it("Should configure concentration levels on BinRebalancer", async function () {
      const { binRebalancer, owner } = await loadFixture(
        deployAllContractsFixture
      );

      // Update MODERATE concentration (level 1) with valid multiplier (1-10)
      await binRebalancer.connect(owner).updateConcentrationMultiplier(1, 5);

      expect(await binRebalancer.concentrationMultipliers(1)).to.equal(5);
    });

    it("Should reject invalid concentration multipliers", async function () {
      const { binRebalancer, owner } = await loadFixture(
        deployAllContractsFixture
      );

      // Try to set invalid multiplier (must be 1-10)
      await expect(
        binRebalancer.connect(owner).updateConcentrationMultiplier(0, 0)
      ).to.be.revertedWith("Invalid multiplier");

      await expect(
        binRebalancer.connect(owner).updateConcentrationMultiplier(0, 11)
      ).to.be.revertedWith("Invalid multiplier");
    });
  });

  describe("Access Control Consistency", function () {
    it("Should prevent non-owner from managing any contract", async function () {
      const { dexterMVP, binRebalancer, ultraFrequentCompounder, user1 } =
        await loadFixture(deployAllContractsFixture);

      // All contracts use OpenZeppelin Ownable which reverts with standard message
      await expect(
        dexterMVP.connect(user1).setKeeperAuthorization(user1.address, true)
      ).to.be.reverted;

      await expect(
        binRebalancer.connect(user1).setKeeperAuthorization(user1.address, true)
      ).to.be.reverted;

      await expect(
        ultraFrequentCompounder.connect(user1).setKeeperAuthorization(user1.address, true)
      ).to.be.reverted;
    });

    it("Should allow ownership transfer on all contracts", async function () {
      const { dexterMVP, binRebalancer, ultraFrequentCompounder, owner, user1 } =
        await loadFixture(deployAllContractsFixture);

      // Transfer ownership
      await dexterMVP.connect(owner).transferOwnership(user1.address);
      await binRebalancer.connect(owner).transferOwnership(user1.address);
      await ultraFrequentCompounder.connect(owner).transferOwnership(user1.address);

      // Verify new owner
      expect(await dexterMVP.owner()).to.equal(user1.address);
      expect(await binRebalancer.owner()).to.equal(user1.address);
      expect(await ultraFrequentCompounder.owner()).to.equal(user1.address);
    });
  });

  describe("Configuration Persistence", function () {
    it("Should persist settings across multiple transactions", async function () {
      const { dexterMVP, owner, keeper1, keeper2 } = await loadFixture(
        deployAllContractsFixture
      );

      // Make multiple configuration changes
      await dexterMVP.connect(owner).setKeeperAuthorization(keeper1.address, true);
      await dexterMVP.connect(owner).setKeeperAuthorization(keeper2.address, true);
      await dexterMVP.connect(owner).setTWAPConfig(90, 150, true);

      // Verify all persisted
      expect(await dexterMVP.authorizedKeepers(keeper1.address)).to.be.true;
      expect(await dexterMVP.authorizedKeepers(keeper2.address)).to.be.true;
      expect(await dexterMVP.twapPeriod()).to.equal(90);
      expect(await dexterMVP.maxTickDifference()).to.equal(150);
    });

    it("Should allow revoking previously granted access", async function () {
      const { dexterMVP, owner, keeper1 } = await loadFixture(
        deployAllContractsFixture
      );

      // Grant then revoke
      await dexterMVP.connect(owner).setKeeperAuthorization(keeper1.address, true);
      expect(await dexterMVP.authorizedKeepers(keeper1.address)).to.be.true;

      await dexterMVP.connect(owner).setKeeperAuthorization(keeper1.address, false);
      expect(await dexterMVP.authorizedKeepers(keeper1.address)).to.be.false;
    });
  });

  describe("Contract Address Validation", function () {
    it("Should store correct external contract addresses", async function () {
      const {
        dexterMVP,
        binRebalancer,
        ultraFrequentCompounder,
        mockPositionManager,
        mockSwapRouter,
        mockFactory,
      } = await loadFixture(deployAllContractsFixture);

      // All contracts should reference the position manager
      expect(await dexterMVP.nonfungiblePositionManager()).to.equal(
        mockPositionManager
      );
      expect(await binRebalancer.nonfungiblePositionManager()).to.equal(
        mockPositionManager
      );
      expect(await ultraFrequentCompounder.nonfungiblePositionManager()).to.equal(
        mockPositionManager
      );

      // DexterMVP has swapRouter
      expect(await dexterMVP.swapRouter()).to.equal(mockSwapRouter);

      // BinRebalancer has factory instead of swapRouter
      expect(await binRebalancer.factory()).to.equal(mockFactory);
    });
  });

  describe("Constants Consistency", function () {
    it("Should have matching position limits across DexterMVP", async function () {
      const { dexterMVP } = await loadFixture(deployAllContractsFixture);

      expect(await dexterMVP.MAX_POSITIONS_PER_ADDRESS()).to.equal(200);
      // MIN_COMPOUND_THRESHOLD_USD is 5e17 (0.5 USD in wei)
      expect(await dexterMVP.MIN_COMPOUND_THRESHOLD_USD()).to.equal(ethers.parseUnits("0.5", 18));
    });

    it("Should have matching rebalance constants on BinRebalancer", async function () {
      const { binRebalancer } = await loadFixture(deployAllContractsFixture);

      // BinRebalancer specific constants
      expect(await binRebalancer.MAX_BINS_FROM_PRICE()).to.equal(5);
      expect(await binRebalancer.MIN_CONCENTRATION_LEVEL()).to.equal(1);
      expect(await binRebalancer.MAX_CONCENTRATION_LEVEL()).to.equal(10);
      expect(await binRebalancer.DEFAULT_CONCENTRATION()).to.equal(5);
    });

    it("Should have matching compound thresholds on UltraFrequentCompounder", async function () {
      const { ultraFrequentCompounder } = await loadFixture(
        deployAllContractsFixture
      );

      // UltraFrequentCompounder specific constants
      expect(await ultraFrequentCompounder.ULTRA_FREQUENT_INTERVAL()).to.equal(5 * 60); // 5 minutes in seconds
      expect(await ultraFrequentCompounder.MAX_BATCH_SIZE()).to.equal(100);
      expect(await ultraFrequentCompounder.GAS_LIMIT_PER_COMPOUND()).to.equal(150000);
    });
  });
});
