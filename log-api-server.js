#!/usr/bin/env node

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3004;
const LOG_DIR = '/var/log/dexter';
const DEXTER_LOG = 'dexter.log';
const LIQUIDITY_LOG = 'liquidity.log';

// Cache for recent logs
let recentLogs = [];
const MAX_LOGS = 50;

// Read recent log entries from files
function readRecentLogs() {
  const logs = [];
  
  try {
    // Read dexter.log (human readable format)
    const dexterLogPath = path.join(LOG_DIR, DEXTER_LOG);
    if (fs.existsSync(dexterLogPath)) {
      const content = fs.readFileSync(dexterLogPath, 'utf8');
      const lines = content.split('\n').filter(line => line.trim()).slice(-25);
      
      lines.forEach(line => {
        if (line.trim()) {
          logs.push({
            type: line.includes('ERROR') || line.includes('WARN') ? 'error' : 'log',
            data: line,
            timestamp: extractTimestamp(line) || new Date().toISOString(),
            source: 'dexter'
          });
        }
      });
    }
    
    // Read liquidity.log (JSON format) 
    const liquidityLogPath = path.join(LOG_DIR, LIQUIDITY_LOG);
    if (fs.existsSync(liquidityLogPath)) {
      const content = fs.readFileSync(liquidityLogPath, 'utf8');
      const lines = content.split('\n').filter(line => line.trim()).slice(-25);
      
      lines.forEach(line => {
        if (line.trim()) {
          try {
            const parsed = JSON.parse(line);
            logs.push({
              type: 'log',
              data: `[${parsed.agent}] ${parsed.action} | Pool: ${parsed.pool_name} | Amount: $${parsed.amount_usd?.toFixed(0) || '0'} | APR: ${parsed.apr_current?.toFixed(1) || '0.0'}%`,
              timestamp: parsed.timestamp || new Date().toISOString(),
              source: 'liquidity',
              raw: parsed
            });
          } catch (e) {
            // Skip invalid JSON lines
          }
        }
      });
    }
    
  } catch (error) {
    console.error('Error reading logs:', error);
    // Add error log entry
    logs.push({
      type: 'error',
      data: `[LOG-API] Error reading log files: ${error.message}`,
      timestamp: new Date().toISOString(),
      source: 'system'
    });
  }
  
  // If no logs found, add some demo entries
  if (logs.length === 0) {
    logs.push(
      {
        type: 'log',
        data: '[ConservativeAgent] Monitoring ETH/USDC pool | Current APR: 15.2% | Volatility: 22.1%',
        timestamp: new Date().toISOString(),
        source: 'demo'
      },
      {
        type: 'log',
        data: '[AggressiveAgent] Position opened | Pool: WBTC/ETH | Amount: $32,500 | Confidence: 84.3%',
        timestamp: new Date(Date.now() - 45000).toISOString(),
        source: 'demo'
      },
      {
        type: 'log',
        data: '[HyperAggressiveAgent] Arbitrage detected | Spread: 0.45% | Est. Profit: $1,250',
        timestamp: new Date(Date.now() - 90000).toISOString(),
        source: 'demo'
      }
    );
  }
  
  // Sort by timestamp (newest first) and limit
  return logs
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .slice(0, MAX_LOGS);
}

// Extract timestamp from log line
function extractTimestamp(line) {
  const match = line.match(/\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*Z?)\]/);
  return match ? match[1] : null;
}

// Update logs cache every 5 seconds
function updateLogsCache() {
  try {
    recentLogs = readRecentLogs();
    console.log(`Updated logs cache: ${recentLogs.length} entries`);
  } catch (error) {
    console.error('Error updating logs cache:', error);
  }
}

// Create HTTP server
const server = http.createServer((req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }
  
  // Route: Get recent logs
  if (req.url === '/api/recent-logs' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      logs: recentLogs,
      count: recentLogs.length,
      timestamp: new Date().toISOString(),
      status: 'success'
    }));
    return;
  }
  
  // Route: Health check
  if (req.url === '/health' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'healthy',
      logCount: recentLogs.length,
      logDir: LOG_DIR,
      timestamp: new Date().toISOString()
    }));
    return;
  }
  
  // Route: Force refresh logs
  if (req.url === '/api/refresh-logs' && req.method === 'POST') {
    updateLogsCache();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      message: 'Logs refreshed',
      count: recentLogs.length,
      timestamp: new Date().toISOString()
    }));
    return;
  }
  
  // 404 for other routes
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

// Start server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Dexter Log API Server running on port ${PORT}`);
  console.log(`Recent logs endpoint: http://localhost:${PORT}/api/recent-logs`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Monitoring directory: ${LOG_DIR}`);
  
  // Initial cache update
  updateLogsCache();
  
  // Set up periodic cache updates
  setInterval(updateLogsCache, 5000); // Update every 5 seconds
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, closing server...');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, closing server...');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled rejection at:', promise, 'reason:', reason);
  process.exit(1);
});