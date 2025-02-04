import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { AlertCircle, TrendingUp, DollarSign, Activity } from 'lucide-react';

// API integration with authentication
const API_URL = 'http://localhost:8000/api/v1';

const useApi = () => {
  const [apiKey, setApiKey] = useState(localStorage.getItem('apiKey'));

  const fetchWithAuth = async (endpoint, options = {}) => {
    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
        'X-API-Key': apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  };

  return { fetchWithAuth, apiKey, setApiKey };
};

const PoolList = ({ onPoolSelect }) => {
  const [pools, setPools] = useState([]);
  const [loading, setLoading] = useState(true);
  const { fetchWithAuth } = useApi();

  useEffect(() => {
    const loadPools = async () => {
      try {
        const data = await fetchWithAuth('/pools');
        setPools(data);
      } catch (error) {
        console.error('Failed to load pools:', error);
      } finally {
        setLoading(false);
      }
    };

    loadPools();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading pools...</div>;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {pools.map(pool => (
        <div 
          key={pool.pool_id}
          className="p-4 bg-white rounded-lg shadow cursor-pointer hover:shadow-lg transition-shadow"
          onClick={() => onPoolSelect(pool)}
        >
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-lg font-semibold">{pool.token_a}/{pool.token_b}</h3>
            <span className={`px-2 py-1 rounded text-sm ${
              pool.status === 'healthy' ? 'bg-green-100 text-green-800' :
              'bg-yellow-100 text-yellow-800'
            }`}>
              {pool.status}
            </span>
          </div>
          
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center gap-2">
              <DollarSign size={16} />
              <span>TVL: ${Number(pool.tvl_usd).toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp size={16} />
              <span>APY: {pool.apy}%</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

const StrategyView = ({ pool }) => {
  const [strategy, setStrategy] = useState(null);
  const [loading, setLoading] = useState(true);
  const { fetchWithAuth } = useApi();

  useEffect(() => {
    const loadStrategy = async () => {
      try {
        const data = await fetchWithAuth(`/pools/${pool.pool_id}/strategy?risk_level=0.5`);
        setStrategy(data);
      } catch (error) {
        console.error('Failed to load strategy:', error);
      } finally {
        setLoading(false);
      }
    };

    loadStrategy();
  }, [pool]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading strategy...</div>;
  }

  if (!strategy) {
    return <div className="flex items-center justify-center h-64">No strategy available</div>;
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold mb-4">Recommended Strategy</h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="text-sm font-medium text-gray-500">Price Range</h4>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-lg">{strategy.optimal_range[0]} - {strategy.optimal_range[1]}</span>
          </div>
        </div>
        
        <div>
          <h4 className="text-sm font-medium text-gray-500">Confidence Score</h4>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-lg">{(strategy.confidence_score * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      <div className="mt-4">
        <h4 className="text-sm font-medium text-gray-500">Suggested Fee</h4>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-lg">{(strategy.suggested_fee * 100).toFixed(2)}%</span>
        </div>
      </div>
    </div>
  );
};

const LoginForm = ({ onLogin }) => {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Verify API key
      const response = await fetch(`${API_URL}/verify`, {
        headers: { 'X-API-Key': apiKey }
      });
      
      if (!response.ok) {
        throw new Error('Invalid API key');
      }
      
      localStorage.setItem('apiKey', apiKey);
      onLogin(apiKey);
    } catch (error) {
      setError(error.message);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="p-8 bg-white rounded-lg shadow-md w-96">
        <h2 className="text-2xl font-bold mb-6 text-center">DexBrain Login</h2>
        
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700">API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
          
          {error && (
            <div className="mb-4 p-2 bg-red-100 text-red-700 rounded-md flex items-center gap-2">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
          
          <button
            type="submit"
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Login
          </button>
        </form>
      </div>
    </div>
  );
};

const Dashboard = () => {
  const [selectedPool, setSelectedPool] = useState(null);
  const [apiKey, setApiKey] = useState(localStorage.getItem('apiKey'));

  if (!apiKey) {
    return <LoginForm onLogin={setApiKey} />;
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold">DexBrain</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <h2 className="text-2xl font-bold mb-4">Active Pools</h2>
            <PoolList onPoolSelect={setSelectedPool} />
          </div>
          
          <div>
            <h2 className="text-2xl font-bold mb-4">Strategy</h2>
            {selectedPool ? (
              <StrategyView pool={selectedPool} />
            ) : (
              <div className="bg-white rounded-lg shadow p-4 flex items-center justify-center h-64">
                Select a pool to view strategy
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;