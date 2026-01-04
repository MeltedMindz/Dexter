import React, { useState, useEffect } from 'react';
import { NextPage } from 'next';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { useAccount, useConnect, useDisconnect } from 'wagmi';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Wallet, ArrowLeft, CheckCircle, AlertTriangle, Loader } from 'lucide-react';
import PositionCreator from '@/components/PositionCreator';
import { toast } from 'sonner';

// Mock data for development - in production this would come from APIs
const mockTokens = [
  {
    address: '0xA0b86a33E6441D9C77df40C7f12A8c8DEE6e2d5F',
    symbol: 'ETH',
    name: 'Ethereum',
    decimals: 18,
    logoURI: 'https://raw.githubusercontent.com/trustwallet/assets/master/blockchains/ethereum/assets/0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2/logo.png',
    balance: '2.45',
    price: 2350.67,
    priceChange24h: 3.42
  },
  {
    address: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
    symbol: 'USDC',
    name: 'USD Coin',
    decimals: 6,
    logoURI: 'https://raw.githubusercontent.com/trustwallet/assets/master/blockchains/ethereum/assets/0xA0b86a33E6441D9C77df40C7f12A8c8DEE6e2d5F/logo.png',
    balance: '5420.18',
    price: 1.00,
    priceChange24h: 0.01
  },
  {
    address: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
    symbol: 'DAI',
    name: 'Dai Stablecoin',
    decimals: 18,
    logoURI: 'https://raw.githubusercontent.com/trustwallet/assets/master/blockchains/ethereum/assets/0x6B175474E89094C44Da98b954EedeAC495271d0F/logo.png',
    balance: '1230.50',
    price: 0.999,
    priceChange24h: -0.05
  }
];

const mockPools = [
  {
    address: '0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640',
    token0: mockTokens[1], // USDC
    token1: mockTokens[0], // ETH
    fee: 500,
    tvl: 125000000,
    volume24h: 15000000,
    apr: 25.67,
    feesGenerated24h: 45000,
    positions: 1247,
    isActive: true
  },
  {
    address: '0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8',
    token0: mockTokens[1], // USDC
    token1: mockTokens[0], // ETH
    fee: 3000,
    tvl: 89000000,
    volume24h: 12000000,
    apr: 18.34,
    feesGenerated24h: 32000,
    positions: 892,
    isActive: true
  },
  {
    address: '0x5777d92f208679DB4b9778590Fa3CAB3aC9e2168',
    token0: mockTokens[2], // DAI
    token1: mockTokens[1], // USDC
    fee: 100,
    tvl: 42000000,
    volume24h: 3200000,
    apr: 8.92,
    feesGenerated24h: 8900,
    positions: 543,
    isActive: true
  }
];

const CreatePositionPage: NextPage = () => {
  const router = useRouter();
  const { address, isConnected } = useAccount();
  const { connect, connectors } = useConnect();
  const { disconnect } = useDisconnect();
  
  const [isLoading, setIsLoading] = useState(false);
  const [creationStep, setCreationStep] = useState<'setup' | 'creating' | 'success' | 'error'>('setup');
  const [createdPosition, setCreatedPosition] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Handle position creation
  const handleCreatePosition = async (positionData: any) => {
    if (!isConnected || !address) {
      toast.error('Please connect your wallet first');
      return;
    }

    setIsLoading(true);
    setCreationStep('creating');
    setError(null);

    try {
      // Simulate position creation process
      console.log('Creating position with data:', positionData);
      
      // In production, this would:
      // 1. Approve tokens if needed
      // 2. Call DexterMVP.depositPosition()
      // 3. Wait for transaction confirmation
      // 4. Update UI with new position

      // Simulate async operation
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Mock successful creation
      const newPosition = {
        id: Date.now().toString(),
        tokenId: Math.floor(Math.random() * 10000),
        ...positionData,
        createdAt: new Date().toISOString(),
        status: 'active'
      };

      setCreatedPosition(newPosition);
      setCreationStep('success');
      
      toast.success('Position created successfully!');
      
      // Redirect to position details after a delay
      setTimeout(() => {
        router.push(`/positions/${newPosition.tokenId}`);
      }, 2000);

    } catch (err: any) {
      console.error('Failed to create position:', err);
      setError(err.message || 'Failed to create position');
      setCreationStep('error');
      toast.error('Failed to create position');
    } finally {
      setIsLoading(false);
    }
  };

  // Wallet connection handler
  const handleConnectWallet = () => {
    const injectedConnector = connectors.find(connector => connector.name === 'MetaMask');
    if (injectedConnector) {
      connect({ connector: injectedConnector });
    }
  };

  // Render different states
  if (!isConnected) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center p-6">
        <Head>
          <title>Create Position - Dexter MVP</title>
          <meta name="description" content="Create a new automated liquidity position with ultra-frequent compounding" />
        </Head>
        
        <Card className="w-full max-w-md bg-gray-800 border-gray-700">
          <CardHeader className="text-center">
            <CardTitle className="text-white flex items-center justify-center space-x-2">
              <Wallet className="w-6 h-6" />
              <span>Connect Wallet</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-gray-400 text-center">
              Connect your wallet to create automated liquidity positions with ultra-frequent compounding.
            </p>
            
            <Alert className="bg-blue-900/20 border-blue-400">
              <CheckCircle className="h-4 w-4" />
              <AlertDescription className="text-blue-300">
                <strong>Dexter MVP Features:</strong>
                <ul className="mt-2 space-y-1 text-sm">
                  <li>• 5-minute auto-compounding</li>
                  <li>• Bin-based rebalancing</li>
                  <li>• Base chain optimized</li>
                  <li>• Ultra-low gas costs</li>
                </ul>
              </AlertDescription>
            </Alert>
            
            <Button 
              onClick={handleConnectWallet}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
            >
              Connect Wallet
            </Button>
            
            <div className="text-center">
              <Button 
                variant="ghost" 
                onClick={() => router.back()}
                className="text-gray-400 hover:text-white"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Go Back
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (creationStep === 'creating') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center p-6">
        <Head>
          <title>Creating Position - Dexter MVP</title>
        </Head>
        
        <Card className="w-full max-w-md bg-gray-800 border-gray-700">
          <CardContent className="p-8 text-center space-y-4">
            <div className="flex justify-center">
              <Loader className="w-12 h-12 text-purple-400 animate-spin" />
            </div>
            <h2 className="text-xl font-bold text-white">Creating Position</h2>
            <p className="text-gray-400">
              Please confirm the transaction in your wallet and wait for it to be confirmed on Base chain.
            </p>
            <div className="space-y-2 text-sm text-gray-500">
              <div>• Approving tokens...</div>
              <div>• Creating liquidity position...</div>
              <div>• Setting up automation...</div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (creationStep === 'success') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center p-6">
        <Head>
          <title>Position Created - Dexter MVP</title>
        </Head>
        
        <Card className="w-full max-w-md bg-gray-800 border-gray-700">
          <CardContent className="p-8 text-center space-y-4">
            <div className="flex justify-center">
              <CheckCircle className="w-12 h-12 text-green-400" />
            </div>
            <h2 className="text-xl font-bold text-white">Position Created!</h2>
            <p className="text-gray-400">
              Your automated liquidity position has been created successfully.
            </p>
            
            {createdPosition && (
              <div className="bg-gray-700 rounded-lg p-4 space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Position ID:</span>
                  <span className="text-white font-mono">#{createdPosition.tokenId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Pool:</span>
                  <span className="text-white">
                    {createdPosition.pool.token0.symbol}/{createdPosition.pool.token1.symbol}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Strategy:</span>
                  <Badge variant="outline" className="text-purple-400 border-purple-400">
                    {createdPosition.strategy}
                  </Badge>
                </div>
              </div>
            )}
            
            <p className="text-sm text-gray-500">
              Redirecting to position details...
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (creationStep === 'error') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center p-6">
        <Head>
          <title>Error - Dexter MVP</title>
        </Head>
        
        <Card className="w-full max-w-md bg-gray-800 border-gray-700">
          <CardContent className="p-8 text-center space-y-4">
            <div className="flex justify-center">
              <AlertTriangle className="w-12 h-12 text-red-400" />
            </div>
            <h2 className="text-xl font-bold text-white">Creation Failed</h2>
            <p className="text-gray-400">
              There was an error creating your position.
            </p>
            
            {error && (
              <Alert className="bg-red-900/20 border-red-400 text-left">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription className="text-red-300">
                  {error}
                </AlertDescription>
              </Alert>
            )}
            
            <div className="space-y-2">
              <Button 
                onClick={() => setCreationStep('setup')}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
              >
                Try Again
              </Button>
              <Button 
                variant="ghost"
                onClick={() => router.push('/positions')}
                className="w-full text-gray-400 hover:text-white"
              >
                Go to Positions
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Main position creation interface
  return (
    <div>
      <Head>
        <title>Create Position - Dexter MVP</title>
        <meta name="description" content="Create a new automated liquidity position with ultra-frequent compounding and bin-based rebalancing" />
      </Head>
      
      <PositionCreator
        onCreatePosition={handleCreatePosition}
        availablePools={mockPools}
      />
    </div>
  );
};

export default CreatePositionPage;