import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Plus, 
  Minus, 
  Settings, 
  TrendingUp, 
  Zap, 
  Target, 
  BarChart3,
  DollarSign,
  Timer,
  AlertTriangle,
  CheckCircle,
  Activity,
  RefreshCw
} from 'lucide-react';
import { useAccount, useBalance } from 'wagmi';
import { ethers } from 'ethers';

interface Token {
  address: string;
  symbol: string;
  decimals: number;
  logoURI: string;
  balance?: string;
  price?: number;
}

interface VolatilityStrategy {
  id: 'spot' | 'curve' | 'bidAsk';
  name: string;
  description: string;
  icon: React.ReactNode;
  riskLevel: 'low' | 'medium' | 'high';
}

interface PriceRange {
  min: number;
  max: number;
  current: number;
  minPercent: number;
  maxPercent: number;
  binCount: number;
}

interface AutomationSettings {
  autoCompoundEnabled: boolean;
  autoRebalanceEnabled: boolean;
  compoundThresholdUSD: number;
  concentrationLevel: number;
  maxBinsFromPrice: number;
  volatilityStrategy: 'spot' | 'curve' | 'bidAsk';
}

interface PositionCreatorProps {
  onCreatePosition: (positionData: any) => void;
  availablePools: Array<{
    token0: Token;
    token1: Token;
    fee: number;
    tvl: number;
    apr: number;
  }>;
}

const PositionCreator: React.FC<PositionCreatorProps> = ({ onCreatePosition, availablePools }) => {
  const { address } = useAccount();
  
  // State management
  const [selectedPool, setSelectedPool] = useState<number>(0);
  const [token0Amount, setToken0Amount] = useState<string>('0.00');
  const [token1Amount, setToken1Amount] = useState<string>('0.00');
  const [autoFill, setAutoFill] = useState<boolean>(true);
  const [selectedStrategy, setSelectedStrategy] = useState<'spot' | 'curve' | 'bidAsk'>('spot');
  const [priceRange, setPriceRange] = useState<PriceRange>({
    min: 0,
    max: 0,
    current: 0,
    minPercent: -50,
    maxPercent: 50,
    binCount: 69
  });
  const [automationSettings, setAutomationSettings] = useState<AutomationSettings>({
    autoCompoundEnabled: true,
    autoRebalanceEnabled: true,
    compoundThresholdUSD: 2.0,
    concentrationLevel: 7,
    maxBinsFromPrice: 3,
    volatilityStrategy: 'spot'
  });
  const [isCreating, setIsCreating] = useState<boolean>(false);
  const [estimatedGas, setEstimatedGas] = useState<string>('~$0.05');

  // Volatility strategies
  const volatilityStrategies: VolatilityStrategy[] = [
    {
      id: 'spot',
      name: 'Spot',
      description: 'Uniform distribution that is versatile and risk adjusted, suitable for any type of market conditions. Similar to setting a price range on a CLMM.',
      icon: <BarChart3 className="w-5 h-5" />,
      riskLevel: 'medium'
    },
    {
      id: 'curve',
      name: 'Curve',
      description: 'Concentrated around current price with smooth distribution curve for stable pairs and trending markets.',
      icon: <TrendingUp className="w-5 h-5" />,
      riskLevel: 'low'
    },
    {
      id: 'bidAsk',
      name: 'Bid Ask',
      description: 'Aggressive strategy focusing on bid-ask spread capture with tight ranges around current price.',
      icon: <Target className="w-5 h-5" />,
      riskLevel: 'high'
    }
  ];

  // Current pool data
  const currentPool = useMemo(() => {
    return availablePools[selectedPool] || null;
  }, [availablePools, selectedPool]);

  // Calculate position metrics
  const positionMetrics = useMemo(() => {
    if (!currentPool || !token0Amount || !token1Amount) {
      return {
        totalValue: 0,
        expectedFees: 0,
        concentrationBoost: 1,
        impermanentLossRisk: 'Medium'
      };
    }

    const amount0 = parseFloat(token0Amount);
    const amount1 = parseFloat(token1Amount);
    const totalValue = (amount0 * (currentPool.token0.price || 0)) + (amount1 * (currentPool.token1.price || 0));
    
    // Calculate concentration boost based on range
    const rangeSize = Math.abs(priceRange.maxPercent - priceRange.minPercent);
    const concentrationBoost = Math.max(1, 100 / rangeSize);
    
    // Estimate fees based on concentration and pool APR
    const expectedFees = (totalValue * currentPool.apr * concentrationBoost) / 100;
    
    // IL risk based on range size
    const impermanentLossRisk = rangeSize > 100 ? 'Low' : rangeSize > 50 ? 'Medium' : 'High';

    return {
      totalValue,
      expectedFees,
      concentrationBoost,
      impermanentLossRisk
    };
  }, [currentPool, token0Amount, token1Amount, priceRange]);

  // Update price range when strategy changes
  useEffect(() => {
    if (!currentPool) return;

    const currentPrice = 0.4240; // Mock current price - would come from pool data
    
    let newRange: Partial<PriceRange>;
    
    switch (selectedStrategy) {
      case 'spot':
        newRange = {
          minPercent: -50,
          maxPercent: 50,
          binCount: 69
        };
        break;
      case 'curve':
        newRange = {
          minPercent: -25,
          maxPercent: 25,
          binCount: 35
        };
        break;
      case 'bidAsk':
        newRange = {
          minPercent: -5,
          maxPercent: 5,
          binCount: 15
        };
        break;
    }

    setPriceRange(prev => ({
      ...prev,
      ...newRange,
      current: currentPrice,
      min: currentPrice * (1 + (newRange.minPercent || 0) / 100),
      max: currentPrice * (1 + (newRange.maxPercent || 0) / 100)
    }));
  }, [selectedStrategy, currentPool]);

  // Auto-fill amounts when one changes
  useEffect(() => {
    if (!autoFill || !currentPool) return;

    // Simple 50/50 value split for demo
    if (token0Amount && !token1Amount) {
      const value0 = parseFloat(token0Amount) * (currentPool.token0.price || 1);
      const amount1 = value0 / (currentPool.token1.price || 1);
      setToken1Amount(amount1.toFixed(6));
    }
  }, [token0Amount, autoFill, currentPool]);

  // Generate price range visualization
  const generatePriceRangeVisualization = () => {
    const bins = [];
    const totalBins = priceRange.binCount;
    const currentPriceIndex = Math.floor(totalBins / 2);
    
    for (let i = 0; i < totalBins; i++) {
      const isInRange = true; // All bins in range for this demo
      const isCurrentPrice = Math.abs(i - currentPriceIndex) < 2;
      
      bins.push(
        <div
          key={i}
          className={`h-8 w-2 mx-px ${
            isCurrentPrice 
              ? 'bg-purple-500' 
              : isInRange 
                ? 'bg-cyan-400' 
                : 'bg-gray-600'
          }`}
        />
      );
    }
    
    return bins;
  };

  // Calculate automation preview
  const automationPreview = useMemo(() => {
    const totalValue = positionMetrics.totalValue;
    const dailyFees = positionMetrics.expectedFees / 365;
    const compoundsPerDay = automationSettings.autoCompoundEnabled ? 24 / (5/60) : 0; // Every 5 minutes
    const rebalancesPerWeek = automationSettings.autoRebalanceEnabled ? 7 * 24 / (30/60) : 0; // Every 30 minutes
    
    return {
      compoundsPerDay,
      rebalancesPerWeek,
      dailyFees,
      weeklyBoost: totalValue * 0.001 * compoundsPerDay * 7 // Compound effect estimate
    };
  }, [positionMetrics, automationSettings]);

  const handleCreatePosition = async () => {
    if (!currentPool || isCreating) return;
    
    setIsCreating(true);
    
    try {
      const positionData = {
        pool: currentPool,
        amounts: {
          token0: token0Amount,
          token1: token1Amount
        },
        priceRange: {
          tickLower: Math.floor(priceRange.min * 100), // Convert to ticks
          tickUpper: Math.floor(priceRange.max * 100)
        },
        automation: {
          ...automationSettings,
          volatilityStrategy: selectedStrategy
        },
        strategy: selectedStrategy,
        expectedMetrics: positionMetrics
      };
      
      await onCreatePosition(positionData);
    } catch (error) {
      console.error('Failed to create position:', error);
    } finally {
      setIsCreating(false);
    }
  };

  if (!currentPool) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        <div className="text-center">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2" />
          <p>No pools available. Please connect your wallet.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-gradient-to-br from-gray-900 to-black min-h-screen">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Create Position</h1>
          <p className="text-gray-400">
            Ultra-frequent automation • 5-minute compounds • Bin-based rebalancing
          </p>
        </div>
        <Badge variant="outline" className="text-green-400 border-green-400">
          <Activity className="w-4 h-4 mr-2" />
          Base Optimized
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Position Creation */}
        <div className="lg:col-span-2 space-y-6">
          {/* Pool Selection */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <div className="flex items-center space-x-3">
                  <img src={currentPool.token0.logoURI} alt={currentPool.token0.symbol} className="w-8 h-8 rounded-full" />
                  <img src={currentPool.token1.logoURI} alt={currentPool.token1.symbol} className="w-8 h-8 rounded-full -ml-2" />
                  <div>
                    <div className="text-lg">{currentPool.token0.symbol}/{currentPool.token1.symbol}</div>
                    <div className="text-sm text-gray-400">{currentPool.fee / 10000}% Fee Tier</div>
                  </div>
                </div>
              </CardTitle>
              <div className="flex space-x-6 text-sm">
                <div>
                  <span className="text-gray-400">24h Fee/TVL: </span>
                  <span className="text-green-400 font-bold">{currentPool.apr.toFixed(2)}%</span>
                </div>
                <div>
                  <span className="text-gray-400">Your Liquidity: </span>
                  <span className="text-white font-bold">${positionMetrics.totalValue.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-gray-400">Unclaimed Fee: </span>
                  <span className="text-white">$0</span>
                </div>
              </div>
            </CardHeader>
          </Card>

          {/* Deposit Amounts */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white">Deposit Amount</CardTitle>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-400">Auto-fill</span>
                  <Switch checked={autoFill} onCheckedChange={setAutoFill} />
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Token 0 Input */}
              <div className="space-y-2">
                <div className="flex items-center justify-between bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center space-x-3">
                    <img src={currentPool.token0.logoURI} alt={currentPool.token0.symbol} className="w-8 h-8 rounded-full" />
                    <div>
                      <div className="text-white font-semibold">{currentPool.token0.symbol}</div>
                      <div className="text-xs text-gray-400">
                        {currentPool.token0.balance} {currentPool.token0.symbol}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Input
                      type="number"
                      value={token0Amount}
                      onChange={(e) => setToken0Amount(e.target.value)}
                      className="w-32 bg-transparent border-none text-right text-white text-xl"
                      placeholder="0.00"
                    />
                    <div className="flex space-x-1">
                      <Button size="sm" variant="outline" className="text-xs">100%</Button>
                      <Button size="sm" variant="outline" className="text-xs">50%</Button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Token 1 Input */}
              <div className="space-y-2">
                <div className="flex items-center justify-between bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center space-x-3">
                    <img src={currentPool.token1.logoURI} alt={currentPool.token1.symbol} className="w-8 h-8 rounded-full" />
                    <div>
                      <div className="text-white font-semibold">{currentPool.token1.symbol}</div>
                      <div className="text-xs text-gray-400">
                        {currentPool.token1.balance} {currentPool.token1.symbol}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Input
                      type="number"
                      value={token1Amount}
                      onChange={(e) => setToken1Amount(e.target.value)}
                      className="w-32 bg-transparent border-none text-right text-white text-xl"
                      placeholder="0.00"
                    />
                    <div className="flex space-x-1">
                      <Button size="sm" variant="outline" className="text-xs">100%</Button>
                      <Button size="sm" variant="outline" className="text-xs">50%</Button>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Volatility Strategy Selection */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Select Volatility Strategy</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4 mb-4">
                {volatilityStrategies.map((strategy) => (
                  <div
                    key={strategy.id}
                    onClick={() => setSelectedStrategy(strategy.id)}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedStrategy === strategy.id
                        ? 'border-purple-500 bg-purple-500/20'
                        : 'border-gray-600 bg-gray-700/50 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center justify-center mb-2">
                      <div className={`p-2 rounded ${
                        selectedStrategy === strategy.id ? 'text-purple-400' : 'text-gray-400'
                      }`}>
                        {strategy.icon}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className={`font-semibold ${
                        selectedStrategy === strategy.id ? 'text-white' : 'text-gray-300'
                      }`}>
                        {strategy.name}
                      </div>
                      <Badge 
                        variant="outline" 
                        className={`mt-1 text-xs ${
                          strategy.riskLevel === 'low' ? 'border-green-500 text-green-400' :
                          strategy.riskLevel === 'medium' ? 'border-yellow-500 text-yellow-400' :
                          'border-red-500 text-red-400'
                        }`}
                      >
                        {strategy.riskLevel} risk
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
              
              <p className="text-sm text-gray-400">
                {volatilityStrategies.find(s => s.id === selectedStrategy)?.description}
              </p>
            </CardContent>
          </Card>

          {/* Price Range Configuration */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Set Price Range</CardTitle>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-purple-500 rounded"></div>
                  <span className="text-sm text-gray-400">{currentPool.token0.symbol}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-cyan-400 rounded"></div>
                  <span className="text-sm text-gray-400">{currentPool.token1.symbol}</span>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Current Price Display */}
              <div className="text-center">
                <div className="text-sm text-gray-400">Current Price</div>
                <div className="text-2xl font-bold text-white">
                  {priceRange.current.toFixed(6)} {currentPool.token1.symbol}/{currentPool.token0.symbol}
                </div>
              </div>

              {/* Price Range Visualization */}
              <div className="space-y-4">
                <div className="flex justify-center items-end space-x-px h-20 overflow-hidden rounded">
                  {generatePriceRangeVisualization()}
                </div>
                
                {/* Price scale */}
                <div className="flex justify-between text-xs text-gray-500">
                  <span>{(priceRange.current * 0.6).toFixed(3)}</span>
                  <span>{(priceRange.current * 0.8).toFixed(3)}</span>
                  <span>{(priceRange.current * 1.0).toFixed(3)}</span>
                  <span>{(priceRange.current * 1.2).toFixed(3)}</span>
                  <span>{(priceRange.current * 1.4).toFixed(3)}</span>
                </div>
              </div>

              {/* Range Controls */}
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <label className="text-sm text-gray-400">Min Price</label>
                  <div className="flex items-center space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => setPriceRange(prev => ({ ...prev, minPercent: prev.minPercent - 5 }))}
                    >
                      <Minus className="w-3 h-3" />
                    </Button>
                    <Input
                      type="number"
                      value={priceRange.min.toFixed(8)}
                      onChange={(e) => {
                        const newMin = parseFloat(e.target.value);
                        const newMinPercent = ((newMin / priceRange.current) - 1) * 100;
                        setPriceRange(prev => ({ ...prev, min: newMin, minPercent: newMinPercent }));
                      }}
                      className="text-center bg-gray-700"
                    />
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => setPriceRange(prev => ({ ...prev, minPercent: prev.minPercent + 5 }))}
                    >
                      <Plus className="w-3 h-3" />
                    </Button>
                  </div>
                  <div className="text-center text-xs text-gray-500">
                    {priceRange.minPercent.toFixed(2)}%
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm text-gray-400">Max Price</label>
                  <div className="flex items-center space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => setPriceRange(prev => ({ ...prev, maxPercent: prev.maxPercent - 5 }))}
                    >
                      <Minus className="w-3 h-3" />
                    </Button>
                    <Input
                      type="number"
                      value={priceRange.max.toFixed(8)}
                      onChange={(e) => {
                        const newMax = parseFloat(e.target.value);
                        const newMaxPercent = ((newMax / priceRange.current) - 1) * 100;
                        setPriceRange(prev => ({ ...prev, max: newMax, maxPercent: newMaxPercent }));
                      }}
                      className="text-center bg-gray-700"
                    />
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => setPriceRange(prev => ({ ...prev, maxPercent: prev.maxPercent + 5 }))}
                    >
                      <Plus className="w-3 h-3" />
                    </Button>
                  </div>
                  <div className="text-center text-xs text-gray-500">
                    {priceRange.maxPercent.toFixed(2)}%
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm text-gray-400">Bin counts</label>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">{priceRange.binCount}</div>
                    <div className="text-xs text-gray-500">
                      {(Math.abs(priceRange.maxPercent - priceRange.minPercent)).toFixed(1)}% range
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar - Automation & Analytics */}
        <div className="space-y-6">
          {/* MVP Automation Settings */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Zap className="w-5 h-5 mr-2 text-yellow-400" />
                MVP Automation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Auto-Compound */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Timer className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-white">5-Min Compounds</span>
                </div>
                <Switch
                  checked={automationSettings.autoCompoundEnabled}
                  onCheckedChange={(checked) => 
                    setAutomationSettings(prev => ({ ...prev, autoCompoundEnabled: checked }))
                  }
                />
              </div>
              
              {/* Auto-Rebalance */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Target className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-white">Bin Rebalancing</span>
                </div>
                <Switch
                  checked={automationSettings.autoRebalanceEnabled}
                  onCheckedChange={(checked) => 
                    setAutomationSettings(prev => ({ ...prev, autoRebalanceEnabled: checked }))
                  }
                />
              </div>

              {/* Compound Threshold */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-white">Compound Threshold</span>
                  <span className="text-xs text-gray-400">
                    ${automationSettings.compoundThresholdUSD.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[automationSettings.compoundThresholdUSD]}
                  onValueChange={([value]) =>
                    setAutomationSettings(prev => ({ ...prev, compoundThresholdUSD: value }))
                  }
                  max={10}
                  min={0.5}
                  step={0.5}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>$0.50</span>
                  <span>$10.00</span>
                </div>
              </div>

              {/* Concentration Level */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-white">Concentration Level</span>
                  <Badge className="text-xs bg-purple-500">
                    {automationSettings.concentrationLevel >= 9 ? 'Ultra-Tight' :
                     automationSettings.concentrationLevel >= 7 ? 'Very Tight' :
                     automationSettings.concentrationLevel >= 5 ? 'Tight' :
                     automationSettings.concentrationLevel >= 3 ? 'Moderate' : 'Wide'}
                  </Badge>
                </div>
                <Slider
                  value={[automationSettings.concentrationLevel]}
                  onValueChange={([value]) =>
                    setAutomationSettings(prev => ({ ...prev, concentrationLevel: value }))
                  }
                  max={10}
                  min={1}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>Wide</span>
                  <span>Ultra-Tight</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Position Metrics */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Position Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Total Value</span>
                  <span className="text-sm font-bold text-white">
                    ${positionMetrics.totalValue.toFixed(2)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Expected Daily Fees</span>
                  <span className="text-sm font-bold text-green-400">
                    ${(positionMetrics.expectedFees / 365).toFixed(4)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Concentration Boost</span>
                  <span className="text-sm font-bold text-purple-400">
                    {positionMetrics.concentrationBoost.toFixed(1)}x
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">IL Risk</span>
                  <span className={`text-sm font-bold ${
                    positionMetrics.impermanentLossRisk === 'Low' ? 'text-green-400' :
                    positionMetrics.impermanentLossRisk === 'Medium' ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {positionMetrics.impermanentLossRisk}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Automation Preview */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <RefreshCw className="w-5 h-5 mr-2 text-cyan-400" />
                Automation Preview
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Compounds/Day</span>
                  <span className="text-sm font-bold text-green-400">
                    {automationPreview.compoundsPerDay.toFixed(0)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Rebalances/Week</span>
                  <span className="text-sm font-bold text-blue-400">
                    {automationPreview.rebalancesPerWeek.toFixed(0)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Weekly Boost</span>
                  <span className="text-sm font-bold text-purple-400">
                    +${automationPreview.weeklyBoost.toFixed(2)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Gas Cost/Day</span>
                  <span className="text-sm font-bold text-yellow-400">
                    {estimatedGas}
                  </span>
                </div>
              </div>

              <Alert className="bg-blue-900/20 border-blue-400">
                <CheckCircle className="h-4 w-4" />
                <AlertDescription className="text-blue-300 text-xs">
                  Base chain's low gas costs enable ultra-frequent automation that's profitable even with small positions.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          {/* Create Position Button */}
          <Button 
            onClick={handleCreatePosition}
            disabled={!token0Amount || !token1Amount || isCreating}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-3 text-lg font-bold"
          >
            {isCreating ? (
              <>
                <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                Creating Position...
              </>
            ) : (
              <>
                <Plus className="w-5 h-5 mr-2" />
                Add Liquidity
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default PositionCreator;