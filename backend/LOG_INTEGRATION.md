# DexBrain Log Integration

## Overview
The DexBrain logs are now fully integrated with the existing log streaming system that feeds the BrainWindow component on the frontend.

## Integration Details

### Log Files
All logs are stored in `/opt/dexter-ai/`:
- `dexbrain.log` - DexBrain API server activities
- `dexter.log` - Main Dexter agent activities  
- `liquidity.log` - Liquidity management operations
- `defi-analysis.log` - DeFi market analysis
- `sse.log` - SSE streaming logs

### Log Stream Server
- **Endpoint**: `http://5.78.71.231:3003/logs`
- **Protocol**: Server-Sent Events (SSE)
- **Auto-discovery**: Automatically monitors all `.log` files in `/opt/dexter-ai/`
- **Real-time**: Uses `tail -f` to stream new log entries as they're written

### Log Format
Each log entry is sent as JSON with the following structure:
```json
{
  "type": "log" | "error",
  "data": "[filename.log] Log message content",
  "timestamp": "2025-06-20T19:58:21.940Z",
  "meta": {
    "source": "dexbrain.log",
    "raw": "Original log line"
  }
}
```

### Frontend Integration
The BrainWindow component connects to the SSE endpoint and receives a unified stream of all logs. DexBrain logs appear with the prefix `[dexbrain.log]`.

## Testing the Integration

1. **Check log files**:
   ```bash
   ssh root@5.78.71.231 "ls -la /opt/dexter-ai/*.log"
   ```

2. **Test SSE stream**:
   ```bash
   curl -N http://5.78.71.231:3003/logs
   ```

3. **Generate test activity**:
   ```bash
   # Register a test agent
   curl -X POST http://5.78.71.231:8080/api/register \
     -H 'Content-Type: application/json' \
     -d '{"agent_id": "test-agent"}'
   ```

## No Additional Configuration Needed
The existing log stream server automatically picks up the dexbrain.log file without any modifications needed. The BrainWindow frontend will display DexBrain logs alongside all other system logs.