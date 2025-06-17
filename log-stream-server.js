#!/usr/bin/env node

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 3003;
const LOG_DIR = '/opt/dexter-ai/';
const LOG_PATTERNS = ['*.log'];

// Store active client connections
const clients = new Set();

// Helper to parse log lines into structured format compatible with BrainWindow
function parseLogLine(line, filename) {
  const timestamp = new Date().toISOString();
  
  // Determine log type for BrainWindow compatibility
  let type = 'log';
  if (line.match(/\b(ERROR|ERR|CRITICAL|CRIT|FATAL)\b/i)) {
    type = 'error';
  }
  
  // Format the log line with source info
  const formattedLine = `[${filename}] ${line}`;
  
  return {
    type,
    data: formattedLine,
    timestamp,
    // Additional metadata for debugging
    meta: {
      source: filename,
      raw: line
    }
  };
}

// Send SSE message to all connected clients
function broadcast(data) {
  const message = `data: ${JSON.stringify(data)}\n\n`;
  
  clients.forEach(client => {
    try {
      client.write(message);
    } catch (err) {
      // Remove client if write fails
      clients.delete(client);
    }
  });
}

// Monitor log files using tail
function monitorLogs() {
  // Find all log files
  const logFiles = [];
  
  try {
    const files = fs.readdirSync(LOG_DIR);
    files.forEach(file => {
      if (file.endsWith('.log')) {
        const fullPath = path.join(LOG_DIR, file);
        if (fs.statSync(fullPath).isFile()) {
          logFiles.push(fullPath);
        }
      }
    });
  } catch (err) {
    console.error('Error reading log directory:', err);
    // Fallback: broadcast error message to clients
    broadcast({
      type: 'error',
      data: `[LOG-SERVER] ERROR: Failed to read log directory: ${err.message}`,
      timestamp: new Date().toISOString()
    });
    return;
  }
  
  console.log('Monitoring log files:', logFiles);
  
  if (logFiles.length === 0) {
    console.log('No log files found, will retry in 30 seconds...');
    setTimeout(monitorLogs, 30000);
    return;
  }
  
  // Start tail process for each log file
  logFiles.forEach(logFile => {
    const filename = path.basename(logFile);
    const tail = spawn('tail', ['-f', '-n', '10', logFile]);
    
    tail.stdout.on('data', (data) => {
      const lines = data.toString().split('\n').filter(line => line.trim());
      lines.forEach(line => {
        const logEntry = parseLogLine(line, filename);
        broadcast(logEntry);
      });
    });
    
    tail.stderr.on('data', (data) => {
      console.error(`tail error for ${filename}:`, data.toString());
    });
    
    tail.on('error', (err) => {
      console.error(`Failed to start tail for ${filename}:`, err);
    });
    
    tail.on('close', (code) => {
      console.log(`tail process for ${filename} closed with code ${code}`);
      // Restart monitoring after a delay
      setTimeout(() => monitorLogs(), 5000);
    });
  });
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
  
  // Handle SSE endpoint
  if (req.url === '/logs' && req.method === 'GET') {
    // Set SSE headers
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no' // Disable Nginx buffering
    });
    
    // Send initial connection message
    res.write('event: connected\n');
    res.write(`data: {"type": "log", "data": "[LOG-SERVER] Connected to Dexter AI log stream", "timestamp": "${new Date().toISOString()}"}\n\n`);
    
    // Add client to active connections
    clients.add(res);
    console.log(`Client connected. Total clients: ${clients.size}`);
    
    // Send keepalive every 30 seconds
    const keepaliveInterval = setInterval(() => {
      try {
        res.write(':keepalive\n\n');
      } catch (err) {
        clients.delete(res);
        clearInterval(keepaliveInterval);
      }
    }, 30000);
    
    // Handle client disconnect
    req.on('close', () => {
      clients.delete(res);
      clearInterval(keepaliveInterval);
      console.log(`Client disconnected. Total clients: ${clients.size}`);
    });
    
    req.on('error', () => {
      clients.delete(res);
      clearInterval(keepaliveInterval);
    });
    
    return;
  }
  
  // Handle health check
  if (req.url === '/health' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ 
      status: 'healthy', 
      clients: clients.size,
      timestamp: new Date().toISOString(),
      logDir: LOG_DIR
    }));
    return;
  }
  
  // 404 for other routes
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

// Start server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Dexter AI Log Stream Server running on port ${PORT}`);
  console.log(`SSE endpoint: http://localhost:${PORT}/logs`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Monitoring directory: ${LOG_DIR}`);
  
  // Start monitoring logs after server starts
  setTimeout(monitorLogs, 1000);
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