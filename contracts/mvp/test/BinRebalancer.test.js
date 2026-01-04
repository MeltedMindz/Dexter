const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("BinRebalancer", function () {
  let binRebalancer;
  let owner;
  let keeper;
  let user;
  let mockPositionManager;
  let mockFactory;

  beforeEach(async function () {
    [owner, keeper, user] = await ethers.getSigners();
  });

  describe("Deployment", function () {
    it("Should deploy with valid parameters", async function () {
      const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
      const contract = await BinRebalancer.deploy(
        ethers.ZeroAddress, // positionManager
        ethers.ZeroAddress  // factory
      );
      
      expect(await contract.owner()).to.equal(owner.address);
    });

    it("Should initialize concentration levels", async function () {
      const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
      const contract = await BinRebalancer.deploy(
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );
      
      // Check that concentration multipliers are set
      // ULTRA_TIGHT should have multiplier 1
      expect(await contract.concentrationMultipliers(0)).to.equal(1);
    });
  });

  describe("Access Control", function () {
    it("Should allow owner to set keeper authorization", async function () {
      const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
      const contract = await BinRebalancer.deploy(
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );
      
      await contract.connect(owner).setKeeperAuthorization(keeper.address, true);
      expect(await contract.authorizedKeepers(keeper.address)).to.be.true;
    });

    it("Should prevent non-owner from setting keepers", async function () {
      const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
      const contract = await BinRebalancer.deploy(
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );
      
      await expect(
        contract.connect(user).setKeeperAuthorization(keeper.address, true)
      ).to.be.revertedWith("Ownable: caller is not the owner");
    });
  });

  describe("Constants", function () {
    it("Should have correct constant values", async function () {
      const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
      const contract = await BinRebalancer.deploy(
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );
      
      expect(await contract.MAX_BINS_FROM_PRICE()).to.equal(5);
      expect(await contract.MIN_CONCENTRATION_LEVEL()).to.equal(1);
      expect(await contract.MAX_CONCENTRATION_LEVEL()).to.equal(10);
      expect(await contract.DEFAULT_CONCENTRATION()).to.equal(5);
    });
  });

  describe("Concentration Levels", function () {
    it("Should allow owner to update concentration multipliers", async function () {
      const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
      const contract = await BinRebalancer.deploy(
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );
      
      await expect(
        contract.connect(owner).updateConcentrationMultiplier(0, 2) // ULTRA_TIGHT
      ).to.emit(contract, "ConcentrationLevelUpdated")
        .withArgs(0, 2);
      
      expect(await contract.concentrationMultipliers(0)).to.equal(2);
    });

    it("Should reject invalid multiplier values", async function () {
      const BinRebalancer = await ethers.getContractFactory("BinRebalancer");
      const contract = await BinRebalancer.deploy(
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );
      
      await expect(
        contract.connect(owner).updateConcentrationMultiplier(0, 0)
      ).to.be.revertedWith("Invalid multiplier");
      
      await expect(
        contract.connect(owner).updateConcentrationMultiplier(0, 11)
      ).to.be.revertedWith("Invalid multiplier");
    });
  });
});

