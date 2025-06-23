# Dexter AI Log Stream Server - Deployment Guide

This guide walks through deploying the Node.js SSE log streaming service to your VPS.

## Quick Deploy

Run this single command to deploy everything:

```bash
./deploy-log-stream.sh 5.78.71.231
```

## Manual Deployment Steps

### 1. Prerequisites

Ensure your VPS has:
- Node.js installed (`node --version`)
- systemd (for service management)
- SSH access as root

### 2. Deploy Files

```bash
# Create directory
ssh root@5.78.71.231 "mkdir -p /opt/dexter-ai"

# Copy server file
scp log-stream-server.js root@5.78.71.231:/opt/dexter-ai/
ssh root@5.78.71.231 "chmod +x /opt/dexter-ai/log-stream-server.js"

# Copy systemd service
scp dexter-log-stream.service root@5.78.71.231:/etc/systemd/system/
```

### 3. Start Service

```bash
ssh root@5.78.71.231 "
  systemctl daemon-reload
  systemctl enable dexter-log-stream
  systemctl start dexter-log-stream
  systemctl status dexter-log-stream
"
```

### 4. Test Deployment

```bash
# Run the test script
./test-log-stream.sh 5.78.71.231

# Or test manually
curl -s http://5.78.71.231:3003/health | jq .
```

## Verification Steps

### 1. Check Service Status

```bash
ssh root@5.78.71.231 "systemctl status dexter-log-stream"
```

Expected output:
```
● dexter-log-stream.service - Dexter AI Log Stream Server
   Loaded: loaded (/etc/systemd/system/dexter-log-stream.service; enabled)
   Active: active (running) since [timestamp]
```

### 2. Check Logs

```bash
ssh root@5.78.71.231 "journalctl -u dexter-log-stream -f"
```

Expected output:
```
Dexter AI Log Stream Server running on port 3003
SSE endpoint: http://localhost:3003/logs
Health check: http://localhost:3003/health
Monitoring directory: /opt/dexter-ai/
```

### 3. Test Health Endpoint

```bash
curl http://5.78.71.231:3003/health
```

Expected response:
```json
{
  "status": "healthy",
  "clients": 0,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "logDir": "/opt/dexter-ai/"
}
```

### 4. Test SSE Stream

```bash
curl -N -H "Accept: text/event-stream" http://5.78.71.231:3003/logs
```

Expected output:
```
event: connected
data: {"type": "log", "data": "[LOG-SERVER] Connected to Dexter AI log stream", "timestamp": "2024-01-15T10:30:00.000Z"}
```

## Frontend Integration Test

### 1. Check API Route

The frontend API route at `/frontend/app/api/logs/route.ts` should connect to port 3003:

```typescript
const response = await fetch('http://5.78.71.231:3003/logs', {
```

### 2. Test BrainWindow Component

1. Start your Next.js development server:
   ```bash
   cd frontend && npm run dev
   ```

2. Open http://localhost:3000 in your browser

3. Look for the "WINDOW INTO THE BRAIN" section

4. The component should show:
   - "NETWORK: ONLINE" status
   - Log entries appearing in real-time
   - Proper color coding (green for INFO, red for ERROR, etc.)

## Creating Test Log Files

To test the log streaming, create some test log files:

```bash
ssh root@5.78.71.231 "
  echo '$(date) INFO: Dexter AI system initialized' >> /opt/dexter-ai/dexter.log
  echo '$(date) INFO: Conservative agent started' >> /opt/dexter-ai/liquidity.log
  echo '$(date) DEBUG: Pool analysis complete' >> /opt/dexter-ai/dexbrain.log
  echo '$(date) WARN: High gas prices detected' >> /opt/dexter-ai/dexter.log
  echo '$(date) ERROR: RPC connection timeout' >> /opt/dexter-ai/liquidity.log
"
```

These should appear in the frontend BrainWindow in real-time.

## Troubleshooting

### Service Won't Start

1. Check Node.js is installed:
   ```bash
   ssh root@5.78.71.231 "node --version"
   ```

2. Check service logs:
   ```bash
   ssh root@5.78.71.231 "journalctl -u dexter-log-stream -n 50"
   ```

3. Check file permissions:
   ```bash
   ssh root@5.78.71.231 "ls -la /opt/dexter-ai/log-stream-server.js"
   ```

### Port 3003 Not Accessible

1. Check if service is listening:
   ```bash
   ssh root@5.78.71.231 "netstat -tlnp | grep 3003"
   ```

2. Check firewall rules:
   ```bash
   ssh root@5.78.71.231 "ufw status"
   ```

3. If needed, open port:
   ```bash
   ssh root@5.78.71.231 "ufw allow 3003"
   ```

### No Log Files Found

1. Check directory exists:
   ```bash
   ssh root@5.78.71.231 "ls -la /opt/dexter-ai/"
   ```

2. Create test files:
   ```bash
   ssh root@5.78.71.231 "echo 'Test log entry' >> /opt/dexter-ai/test.log"
   ```

### Frontend Not Connecting

1. Check browser console for errors
2. Verify API route is calling correct port (3003)
3. Test direct connection:
   ```bash
   curl -N -H "Accept: text/event-stream" http://5.78.71.231:3003/logs
   ```

### Memory Issues

The service is limited to 512MB. If needed, increase limit in systemd service:

```bash
# Edit service file
ssh root@5.78.71.231 "systemctl edit dexter-log-stream"

# Add:
[Service]
MemoryMax=1G
```

## Service Management Commands

```bash
# Start service
systemctl start dexter-log-stream

# Stop service
systemctl stop dexter-log-stream

# Restart service
systemctl restart dexter-log-stream

# Check status
systemctl status dexter-log-stream

# View logs
journalctl -u dexter-log-stream -f

# Disable service
systemctl disable dexter-log-stream

# Enable service
systemctl enable dexter-log-stream
```

## Success Criteria

The deployment is successful when:

1. ✅ Service shows "active (running)" status
2. ✅ Health endpoint returns JSON response
3. ✅ SSE endpoint streams log data
4. ✅ Frontend BrainWindow shows "NETWORK: ONLINE"
5. ✅ Log entries appear in real-time in the frontend
6. ✅ Test log files generate visible entries

## Next Steps

After successful deployment:

1. Set up proper log rotation for `/opt/dexter-ai/*.log` files
2. Configure monitoring/alerting for the service
3. Set up SSL/TLS if needed for production
4. Consider adding authentication for the log stream
5. Monitor memory usage and adjust limits if needed

## Files Deployed

- `/opt/dexter-ai/log-stream-server.js` - Main Node.js server
- `/etc/systemd/system/dexter-log-stream.service` - Systemd service configuration
- Updated: `/frontend/app/api/logs/route.ts` - Frontend API proxy (port 3003)