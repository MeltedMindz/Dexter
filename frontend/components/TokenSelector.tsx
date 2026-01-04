import React, { useState, useMemo } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Search, ChevronDown, Star, TrendingUp, Activity } from 'lucide-react';

interface Token {
  address: string;
  symbol: string;
  name: string;
  decimals: number;
  logoURI: string;
  balance?: string;
  price?: number;
  priceChange24h?: number;
  volume24h?: number;
  isFavorite?: boolean;
}

interface TokenSelectorProps {
  selectedToken?: Token;
  onTokenSelect: (token: Token) => void;
  availableTokens: Token[];
  className?: string;
}

const TokenSelector: React.FC<TokenSelectorProps> = ({
  selectedToken,
  onTokenSelect,
  availableTokens,
  className = ""
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'all' | 'favorites' | 'popular'>('all');

  // Filter tokens based on search and tab
  const filteredTokens = useMemo(() => {
    let filtered = availableTokens;

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(token => 
        token.symbol.toLowerCase().includes(query) ||
        token.name.toLowerCase().includes(query) ||
        token.address.toLowerCase().includes(query)
      );
    }

    // Filter by tab
    switch (activeTab) {
      case 'favorites':
        filtered = filtered.filter(token => token.isFavorite);
        break;
      case 'popular':
        filtered = filtered.sort((a, b) => (b.volume24h || 0) - (a.volume24h || 0)).slice(0, 10);
        break;
      default:
        // Sort all tokens by volume
        filtered = filtered.sort((a, b) => (b.volume24h || 0) - (a.volume24h || 0));
    }

    return filtered;
  }, [availableTokens, searchQuery, activeTab]);

  // Popular token addresses for Base chain
  const popularTokens = [
    'ETH', 'USDC', 'USDT', 'DAI', 'WBTC', 'LINK', 'UNI', 'COMP', 'AAVE', 'MKR'
  ];

  const handleTokenSelect = (token: Token) => {
    onTokenSelect(token);
    setIsOpen(false);
    setSearchQuery('');
  };

  const formatPrice = (price?: number) => {
    if (!price) return '$0.00';
    return price < 0.01 ? `$${price.toFixed(6)}` : `$${price.toFixed(2)}`;
  };

  const formatPriceChange = (change?: number) => {
    if (!change) return '0.00%';
    const formatted = Math.abs(change).toFixed(2);
    return change >= 0 ? `+${formatted}%` : `-${formatted}%`;
  };

  const formatVolume = (volume?: number) => {
    if (!volume) return '$0';
    if (volume >= 1e9) return `$${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `$${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `$${(volume / 1e3).toFixed(1)}K`;
    return `$${volume.toFixed(0)}`;
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className={`justify-between ${className}`}>
          {selectedToken ? (
            <div className="flex items-center space-x-2">
              <img 
                src={selectedToken.logoURI} 
                alt={selectedToken.symbol}
                className="w-6 h-6 rounded-full"
              />
              <span className="font-semibold">{selectedToken.symbol}</span>
            </div>
          ) : (
            <span className="text-gray-400">Select a token</span>
          )}
          <ChevronDown className="w-4 h-4 opacity-50" />
        </Button>
      </DialogTrigger>
      
      <DialogContent className="sm:max-w-md bg-gray-800 border-gray-700">
        <DialogHeader>
          <DialogTitle className="text-white">Select a Token</DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          {/* Search Input */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              placeholder="Search name or paste address"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-gray-700 border-gray-600 text-white"
            />
          </div>

          {/* Tab Navigation */}
          <div className="flex space-x-1 bg-gray-700 rounded-lg p-1">
            {[
              { id: 'all', label: 'All' },
              { id: 'favorites', label: 'Favorites' },
              { id: 'popular', label: 'Popular' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-gray-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Popular Tokens (when no search) */}
          {!searchQuery && activeTab === 'all' && (
            <div className="space-y-2">
              <div className="text-sm text-gray-400 font-medium">Popular tokens</div>
              <div className="flex flex-wrap gap-2">
                {popularTokens.map(symbol => {
                  const token = availableTokens.find(t => t.symbol === symbol);
                  if (!token) return null;
                  
                  return (
                    <button
                      key={symbol}
                      onClick={() => handleTokenSelect(token)}
                      className="flex items-center space-x-2 bg-gray-700 hover:bg-gray-600 rounded-lg px-3 py-2 transition-colors"
                    >
                      <img 
                        src={token.logoURI} 
                        alt={token.symbol}
                        className="w-5 h-5 rounded-full"
                      />
                      <span className="text-white text-sm font-medium">{token.symbol}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Token List */}
          <div className="space-y-1">
            <div className="text-sm text-gray-400 font-medium">
              {activeTab === 'all' && 'All tokens'}
              {activeTab === 'favorites' && 'Favorite tokens'}
              {activeTab === 'popular' && 'Popular tokens'}
            </div>
            
            <ScrollArea className="h-80">
              <div className="space-y-1">
                {filteredTokens.map((token) => (
                  <button
                    key={token.address}
                    onClick={() => handleTokenSelect(token)}
                    className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-gray-700 transition-colors group"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="relative">
                        <img 
                          src={token.logoURI} 
                          alt={token.symbol}
                          className="w-8 h-8 rounded-full"
                        />
                        {token.isFavorite && (
                          <Star className="absolute -top-1 -right-1 w-3 h-3 text-yellow-400 fill-current" />
                        )}
                      </div>
                      
                      <div className="text-left">
                        <div className="flex items-center space-x-2">
                          <span className="font-semibold text-white">{token.symbol}</span>
                          {popularTokens.includes(token.symbol) && (
                            <Badge variant="outline" className="text-xs border-purple-400 text-purple-400">
                              Popular
                            </Badge>
                          )}
                        </div>
                        <div className="text-sm text-gray-400">{token.name}</div>
                        {token.balance && (
                          <div className="text-xs text-gray-500">
                            Balance: {parseFloat(token.balance).toFixed(4)}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-white font-medium">
                        {formatPrice(token.price)}
                      </div>
                      {token.priceChange24h !== undefined && (
                        <div className={`text-sm ${
                          token.priceChange24h >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {formatPriceChange(token.priceChange24h)}
                        </div>
                      )}
                      {token.volume24h && (
                        <div className="text-xs text-gray-500">
                          Vol: {formatVolume(token.volume24h)}
                        </div>
                      )}
                    </div>
                  </button>
                ))}
                
                {filteredTokens.length === 0 && (
                  <div className="text-center py-8 text-gray-400">
                    <div className="text-lg mb-2">No tokens found</div>
                    <div className="text-sm">
                      {searchQuery 
                        ? `No results for "${searchQuery}"`
                        : 'No tokens available'
                      }
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Footer Info */}
          <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-700">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <Activity className="w-3 h-3" />
                <span>Base Network</span>
              </div>
              <div className="flex items-center space-x-1">
                <TrendingUp className="w-3 h-3" />
                <span>Live Prices</span>
              </div>
            </div>
            <div>
              {filteredTokens.length} tokens
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TokenSelector;