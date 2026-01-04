import React, { useState, useMemo } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Search, ChevronDown, TrendingUp, DollarSign, Zap, Filter } from 'lucide-react';

interface Pool {
  address: string;
  token0: {
    address: string;
    symbol: string;
    name: string;
    logoURI: string;
  };
  token1: {
    address: string;
    symbol: string;
    name: string;
    logoURI: string;
  };
  fee: number;
  tvl: number;
  volume24h: number;
  apr: number;
  feesGenerated24h: number;
  positions: number;
  isActive: boolean;
}

interface PoolSelectorProps {
  selectedPool?: Pool;
  onPoolSelect: (pool: Pool) => void;
  availablePools: Pool[];
  className?: string;
}

const PoolSelector: React.FC<PoolSelectorProps> = ({
  selectedPool,
  onPoolSelect,
  availablePools,
  className = ""
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'tvl' | 'volume' | 'apr' | 'fees'>('tvl');
  const [filterByFee, setFilterByFee] = useState<number | null>(null);

  // Fee tiers available on Uniswap V3
  const feeTiers = [
    { value: 100, label: '0.01%', description: 'Stablecoin pairs' },
    { value: 500, label: '0.05%', description: 'Most pairs' },
    { value: 3000, label: '0.3%', description: 'Standard pairs' },
    { value: 10000, label: '1%', description: 'Exotic pairs' }
  ];

  // Filter and sort pools
  const filteredPools = useMemo(() => {
    let filtered = availablePools;

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(pool => 
        pool.token0.symbol.toLowerCase().includes(query) ||
        pool.token1.symbol.toLowerCase().includes(query) ||
        pool.token0.name.toLowerCase().includes(query) ||
        pool.token1.name.toLowerCase().includes(query) ||
        `${pool.token0.symbol}/${pool.token1.symbol}`.toLowerCase().includes(query)
      );
    }

    // Filter by fee tier
    if (filterByFee !== null) {
      filtered = filtered.filter(pool => pool.fee === filterByFee);
    }

    // Sort pools
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'tvl':
          return b.tvl - a.tvl;
        case 'volume':
          return b.volume24h - a.volume24h;
        case 'apr':
          return b.apr - a.apr;
        case 'fees':
          return b.feesGenerated24h - a.feesGenerated24h;
        default:
          return b.tvl - a.tvl;
      }
    });

    return filtered;
  }, [availablePools, searchQuery, sortBy, filterByFee]);

  const handlePoolSelect = (pool: Pool) => {
    onPoolSelect(pool);
    setIsOpen(false);
    setSearchQuery('');
  };

  const formatCurrency = (value: number) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
    return `$${value.toFixed(0)}`;
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  const getFeeLabel = (fee: number) => {
    const tier = feeTiers.find(t => t.value === fee);
    return tier?.label || `${(fee / 10000).toFixed(2)}%`;
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className={`justify-between h-16 ${className}`}>
          {selectedPool ? (
            <div className="flex items-center space-x-3">
              <div className="flex items-center -space-x-2">
                <img 
                  src={selectedPool.token0.logoURI} 
                  alt={selectedPool.token0.symbol}
                  className="w-8 h-8 rounded-full border-2 border-gray-600"
                />
                <img 
                  src={selectedPool.token1.logoURI} 
                  alt={selectedPool.token1.symbol}
                  className="w-8 h-8 rounded-full border-2 border-gray-600"
                />
              </div>
              <div>
                <div className="font-semibold text-white">
                  {selectedPool.token0.symbol}/{selectedPool.token1.symbol}
                </div>
                <div className="text-sm text-gray-400">
                  {getFeeLabel(selectedPool.fee)} • TVL: {formatCurrency(selectedPool.tvl)}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-gray-400">Select a pool</div>
          )}
          <ChevronDown className="w-4 h-4 opacity-50" />
        </Button>
      </DialogTrigger>
      
      <DialogContent className="sm:max-w-4xl bg-gray-800 border-gray-700 max-h-[80vh]">
        <DialogHeader>
          <DialogTitle className="text-white">Select a Pool</DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          {/* Search and Filters */}
          <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search pools by token name or symbol"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-gray-700 border-gray-600 text-white"
              />
            </div>
            
            <div className="flex space-x-2">
              {/* Sort Dropdown */}
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="bg-gray-700 border-gray-600 text-white rounded-md px-3 py-2 text-sm"
              >
                <option value="tvl">Sort by TVL</option>
                <option value="volume">Sort by Volume</option>
                <option value="apr">Sort by APR</option>
                <option value="fees">Sort by Fees</option>
              </select>
            </div>
          </div>

          {/* Fee Tier Filter */}
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">Fee tier:</span>
            <div className="flex space-x-2">
              <Button
                variant={filterByFee === null ? "default" : "outline"}
                size="sm"
                onClick={() => setFilterByFee(null)}
                className="text-xs"
              >
                All
              </Button>
              {feeTiers.map(tier => (
                <Button
                  key={tier.value}
                  variant={filterByFee === tier.value ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFilterByFee(filterByFee === tier.value ? null : tier.value)}
                  className="text-xs"
                >
                  {tier.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Pool List */}
          <ScrollArea className="h-96">
            <div className="space-y-2">
              {/* Header */}
              <div className="grid grid-cols-6 gap-4 p-3 text-xs text-gray-400 font-medium border-b border-gray-700">
                <div className="col-span-2">Pool</div>
                <div>TVL</div>
                <div>Volume (24h)</div>
                <div>APR</div>
                <div>Fees (24h)</div>
              </div>

              {/* Pool Items */}
              {filteredPools.map((pool) => (
                <button
                  key={pool.address}
                  onClick={() => handlePoolSelect(pool)}
                  className="w-full grid grid-cols-6 gap-4 p-3 rounded-lg hover:bg-gray-700 transition-colors group"
                >
                  {/* Pool Info */}
                  <div className="col-span-2 flex items-center space-x-3">
                    <div className="flex items-center -space-x-2">
                      <img 
                        src={pool.token0.logoURI} 
                        alt={pool.token0.symbol}
                        className="w-8 h-8 rounded-full border-2 border-gray-600"
                      />
                      <img 
                        src={pool.token1.logoURI} 
                        alt={pool.token1.symbol}
                        className="w-8 h-8 rounded-full border-2 border-gray-600"
                      />
                    </div>
                    <div className="text-left">
                      <div className="font-semibold text-white">
                        {pool.token0.symbol}/{pool.token1.symbol}
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline" className="text-xs border-purple-400 text-purple-400">
                          {getFeeLabel(pool.fee)}
                        </Badge>
                        {pool.isActive && (
                          <Badge variant="outline" className="text-xs border-green-400 text-green-400">
                            <Zap className="w-3 h-3 mr-1" />
                            Active
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* TVL */}
                  <div className="text-left">
                    <div className="font-medium text-white">
                      {formatCurrency(pool.tvl)}
                    </div>
                    <div className="text-xs text-gray-400">
                      {pool.positions} positions
                    </div>
                  </div>

                  {/* Volume */}
                  <div className="text-left">
                    <div className="font-medium text-white">
                      {formatCurrency(pool.volume24h)}
                    </div>
                    <div className="text-xs text-gray-400">
                      Volume
                    </div>
                  </div>

                  {/* APR */}
                  <div className="text-left">
                    <div className={`font-medium ${
                      pool.apr > 50 ? 'text-green-400' : 
                      pool.apr > 20 ? 'text-yellow-400' : 'text-white'
                    }`}>
                      {formatPercentage(pool.apr)}
                    </div>
                    <div className="text-xs text-gray-400">
                      APR
                    </div>
                  </div>

                  {/* Fees */}
                  <div className="text-left">
                    <div className="font-medium text-white">
                      {formatCurrency(pool.feesGenerated24h)}
                    </div>
                    <div className="text-xs text-gray-400">
                      Fees
                    </div>
                  </div>
                </button>
              ))}
              
              {filteredPools.length === 0 && (
                <div className="text-center py-8 text-gray-400">
                  <div className="text-lg mb-2">No pools found</div>
                  <div className="text-sm">
                    {searchQuery 
                      ? `No results for "${searchQuery}"`
                      : filterByFee !== null
                        ? `No pools with ${getFeeLabel(filterByFee)} fee tier`
                        : 'No pools available'
                    }
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Pool Statistics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-gray-700">
            <div className="text-center">
              <div className="text-lg font-bold text-white">
                {availablePools.length}
              </div>
              <div className="text-xs text-gray-400">Total Pools</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-white">
                {formatCurrency(availablePools.reduce((sum, pool) => sum + pool.tvl, 0))}
              </div>
              <div className="text-xs text-gray-400">Total TVL</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-white">
                {formatCurrency(availablePools.reduce((sum, pool) => sum + pool.volume24h, 0))}
              </div>
              <div className="text-xs text-gray-400">24h Volume</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-white">
                {formatPercentage(availablePools.reduce((sum, pool) => sum + pool.apr, 0) / availablePools.length)}
              </div>
              <div className="text-xs text-gray-400">Avg APR</div>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-700">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <TrendingUp className="w-3 h-3" />
                <span>Live Data</span>
              </div>
              <div className="flex items-center space-x-1">
                <DollarSign className="w-3 h-3" />
                <span>Base Network</span>
              </div>
            </div>
            <div>
              Showing {filteredPools.length} of {availablePools.length} pools
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default PoolSelector;