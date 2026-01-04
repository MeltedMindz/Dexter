import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { 
  Timer, 
  Zap, 
  TrendingUp, 
  BarChart3, 
  Settings, 
  Activity,
  DollarSign,
  Target
} from 'lucide-react';

interface Position {
  tokenId: string;
  pool: string;
  automation: {
    autoCompoundEnabled: boolean;
    autoRebalanceEnabled: boolean;
    compoundThresholdUSD: number;
    concentrationLevel: number;
    lastCompoundTime: number;
    lastRebalanceTime: number;
  };
  metrics: {
    compoundCount: number;
    rebalanceCount: number;
    totalFeesCompounded: number;
  };
  binData: {
    binsFromPrice: number;
    inRange: boolean;
    concentrationLevel: string;
  };
}

interface MVPDashboardProps {
  positions: Position[];
  onUpdateAutomation: (tokenId: string, settings: any) => void;
}

const MVPDashboard: React.FC<MVPDashboardProps> = ({ positions, onUpdateAutomation }) => {
  const [selectedPosition, setSelectedPosition] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState({
    totalCompounds: 0,
    totalRebalances: 0,
    avgCompoundFreq: '5-15 min',
    totalFeesCompounded: 0
  });

  // Real-time countdown timers
  const [countdowns, setCountdowns] = useState<Record<string, number>>({});

  useEffect(() => {
    // Update countdowns every second
    const interval = setInterval(() => {
      const newCountdowns: Record<string, number> = {};
      
      positions.forEach(position => {
        const timeSinceCompound = Date.now() - position.automation.lastCompoundTime * 1000;
        const nextCompoundIn = Math.max(0, (5 * 60 * 1000) - timeSinceCompound); // 5 minutes
        newCountdowns[position.tokenId] = Math.floor(nextCompoundIn / 1000);
      });
      
      setCountdowns(newCountdowns);
    }, 1000);

    return () => clearInterval(interval);
  }, [positions]);

  const formatCountdown = (seconds: number) => {
    if (seconds <= 0) return "Ready now";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConcentrationLabel = (level: number) => {
    if (level >= 9) return "Ultra-Tight";
    if (level >= 7) return "Very Tight";
    if (level >= 5) return "Tight";
    if (level >= 3) return "Moderate";
    return "Wide";
  };

  const getConcentrationColor = (level: number) => {
    if (level >= 9) return "bg-red-500";
    if (level >= 7) return "bg-orange-500";
    if (level >= 5) return "bg-yellow-500";
    if (level >= 3) return "bg-blue-500";
    return "bg-green-500";
  };

  return (
    <div className="space-y-6 p-6 bg-gradient-to-br from-gray-900 to-black min-h-screen">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">
            MVP Ultra-Frequent Automation
          </h1>
          <p className="text-gray-400">
            5-minute compounds • Bin-based rebalancing • Base chain optimized
          </p>
        </div>
        <Badge variant="outline" className="text-green-400 border-green-400">
          <Activity className="w-4 h-4 mr-2" />
          LIVE
        </Badge>
      </div>

      {/* Live Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-yellow-400" />
              <div>
                <p className="text-sm text-gray-400">Total Compounds</p>
                <p className="text-2xl font-bold text-white">{liveMetrics.totalCompounds}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Target className="w-5 h-5 text-blue-400" />
              <div>
                <p className="text-sm text-gray-400">Total Rebalances</p>
                <p className="text-2xl font-bold text-white">{liveMetrics.totalRebalances}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Timer className="w-5 h-5 text-green-400" />
              <div>
                <p className="text-sm text-gray-400">Avg Frequency</p>
                <p className="text-2xl font-bold text-white">{liveMetrics.avgCompoundFreq}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="w-5 h-5 text-purple-400" />
              <div>
                <p className="text-sm text-gray-400">Fees Compounded</p>
                <p className="text-2xl font-bold text-white">
                  ${liveMetrics.totalFeesCompounded.toFixed(2)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Positions Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {positions.map((position) => (
          <Card key={position.tokenId} className="bg-gray-800 border-gray-700">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white">
                  Position #{position.tokenId}
                </CardTitle>
                <Badge variant="outline" className="text-cyan-400 border-cyan-400">
                  {position.pool}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Live Status */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Next Compound</span>
                    <span className="text-sm font-mono text-green-400">
                      {formatCountdown(countdowns[position.tokenId] || 0)}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Bin Position</span>
                    <Badge 
                      variant={position.binData.inRange ? "default" : "destructive"}
                      className="text-xs"
                    >
                      {position.binData.inRange 
                        ? "In Range" 
                        : `${position.binData.binsFromPrice} bins away`
                      }
                    </Badge>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Compounds</span>
                    <span className="text-sm font-bold text-white">
                      {position.metrics.compoundCount}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Rebalances</span>
                    <span className="text-sm font-bold text-white">
                      {position.metrics.rebalanceCount}
                    </span>
                  </div>
                </div>
              </div>

              {/* Automation Controls */}
              <div className="space-y-3 pt-4 border-t border-gray-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Zap className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm text-white">5-Min Compounds</span>
                  </div>
                  <Switch
                    checked={position.automation.autoCompoundEnabled}
                    onCheckedChange={(checked) => 
                      onUpdateAutomation(position.tokenId, {
                        ...position.automation,
                        autoCompoundEnabled: checked
                      })
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Target className="w-4 h-4 text-blue-400" />
                    <span className="text-sm text-white">Bin Rebalancing</span>
                  </div>
                  <Switch
                    checked={position.automation.autoRebalanceEnabled}
                    onCheckedChange={(checked) => 
                      onUpdateAutomation(position.tokenId, {
                        ...position.automation,
                        autoRebalanceEnabled: checked
                      })
                    }
                  />
                </div>

                {/* Concentration Level Slider */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">Concentration Level</span>
                    <Badge 
                      className={`text-xs ${getConcentrationColor(position.automation.concentrationLevel)}`}
                    >
                      {getConcentrationLabel(position.automation.concentrationLevel)}
                    </Badge>
                  </div>
                  
                  <div className="px-2">
                    <Slider
                      value={[position.automation.concentrationLevel]}
                      onValueChange={([value]) =>
                        onUpdateAutomation(position.tokenId, {
                          ...position.automation,
                          concentrationLevel: value
                        })
                      }
                      max={10}
                      min={1}
                      step={1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Wide</span>
                      <span>Ultra-Tight</span>
                    </div>
                  </div>
                </div>

                {/* Compound Threshold */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">Compound Threshold</span>
                    <span className="text-xs text-gray-400">
                      ${position.automation.compoundThresholdUSD.toFixed(2)}
                    </span>
                  </div>
                  
                  <div className="px-2">
                    <Slider
                      value={[position.automation.compoundThresholdUSD]}
                      onValueChange={([value]) =>
                        onUpdateAutomation(position.tokenId, {
                          ...position.automation,
                          compoundThresholdUSD: value
                        })
                      }
                      max={10}
                      min={0.5}
                      step={0.5}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>$0.50</span>
                      <span>$10.00</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="pt-4 border-t border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Total Fees Compounded</span>
                  <span className="text-sm font-bold text-green-400">
                    ${position.metrics.totalFeesCompounded.toFixed(2)}
                  </span>
                </div>
                
                <Progress 
                  value={(position.metrics.totalFeesCompounded / 100) * 100} 
                  className="h-2"
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Add New Position Button */}
      <Card className="bg-gray-800 border-gray-700 border-dashed">
        <CardContent className="p-8 text-center">
          <Button 
            variant="outline" 
            className="border-gray-600 text-gray-400 hover:text-white hover:border-white"
          >
            <Settings className="w-4 h-4 mr-2" />
            Add Position to MVP Automation
          </Button>
          <p className="text-xs text-gray-500 mt-2">
            Ultra-frequent compounding with bin-based rebalancing
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default MVPDashboard;