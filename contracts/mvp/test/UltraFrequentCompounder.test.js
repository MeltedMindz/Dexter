const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("UltraFrequentCompounder", function () {
  let compounder;
  let owner;
  let keeper;
  let user;

  beforeEach(async function () {
    [owner, keeper, user] = await ethers.getSigners();
  });

  describe("Deployment", function () {
    it("Should deploy with valid parameters", async function () {
      const UltraFrequentCompounder = await ethers.getContractFactory("UltraFrequentCompounder");
      const contract = await UltraFrequentCompounder.deploy(
        ethers.ZeroAddress // positionManager
      );
      
      expect(await contract.owner()).to.equal(owner.address);
    });
  });

  describe("Access Control", function () {
    it("Should allow owner to authorize keepers", async function () {
      const UltraFrequentCompounder = await ethers.getContractFactory("UltraFrequentCompounder");
      const contract = await UltraFrequentCompounder.deploy(
        ethers.ZeroAddress
      );
      
      await contract.connect(owner).setKeeperAuthorization(keeper.address, true);
      expect(await contract.authorizedKeepers(keeper.address)).to.be.true;
    });

    it("Should prevent non-owner from authorizing keepers", async function () {
      const UltraFrequentCompounder = await ethers.getContractFactory("UltraFrequentCompounder");
      const contract = await UltraFrequentCompounder.deploy(
        ethers.ZeroAddress
      );
      
      await expect(
        contract.connect(user).setKeeperAuthorization(keeper.address, true)
      ).to.be.revertedWith("Ownable: caller is not the owner");
    });
  });

  describe("shouldCompound", function () {
    it("Should be callable as public function", async function () {
      const UltraFrequentCompounder = await ethers.getContractFactory("UltraFrequentCompounder");
      const contract = await UltraFrequentCompounder.deploy(
        ethers.ZeroAddress
      );
      
      // Function should exist and be callable (may revert with invalid tokenId, but that's expected)
      await expect(contract.shouldCompound(1)).to.not.be.reverted;
    });
  });
});

