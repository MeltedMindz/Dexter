# Dexter AI Log Stream Server

A Node.js Server-Sent Events (SSE) service that provides real-time log streaming from the Dexter AI system on the VPS.

## Features

- **Real-time log streaming** using Server-Sent Events (SSE)
- **Multiple log file monitoring** with automatic discovery
- **JSON formatted log messages** with timestamps and log levels
- **Multiple client support** with proper connection management
- **Automatic reconnection** and error handling
- **Health check endpoint** for monitoring
- **Systemd service** for automatic startup and management

## Architecture

```
Frontend (Next.js) 
    ↓ (SSE proxy)
Next.js API Route (/api/logs)
    ↓ (HTTP SSE)
VPS Log Stream Server (port 3003)
    ↓ (tail -f)
Log Files (/opt/dexter-ai/*.log)
```

## Installation

### 1. Deploy to VPS

Run the deployment script from your local machine:

```bash
./deploy-log-stream.sh [VPS_IP]
```

This will:
- Create `/opt/dexter-ai/` directory
- Copy the server files to the VPS
- Install and start the systemd service
- Configure automatic startup

### 2. Manual Installation

If you prefer manual installation:

```bash
# On VPS
sudo mkdir -p /opt/dexter-ai
sudo cp log-stream-server.js /opt/dexter-ai/
sudo chmod +x /opt/dexter-ai/log-stream-server.js

# Install systemd service
sudo cp dexter-log-stream.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dexter-log-stream
sudo systemctl start dexter-log-stream
```

## Usage

### Service Management

```bash
# Start the service
sudo systemctl start dexter-log-stream

# Stop the service
sudo systemctl stop dexter-log-stream

# Restart the service
sudo systemctl restart dexter-log-stream

# Check status
sudo systemctl status dexter-log-stream

# View logs
sudo journalctl -u dexter-log-stream -f
```

### Endpoints

- **SSE Stream**: `http://VPS_IP:3003/logs`
- **Health Check**: `http://VPS_IP:3003/health`

### Testing

Run the test script to verify everything works:

```bash
./test-log-stream.sh [VPS_IP]
```

### Manual Testing

Test the SSE stream directly:

```bash
curl -N -H "Accept: text/event-stream" "http://VPS_IP:3003/logs"
```

Create test log entries:

```bash
# On VPS
echo "$(date) INFO: Test log entry" >> /opt/dexter-ai/dexter.log
echo "$(date) ERROR: Test error" >> /opt/dexter-ai/liquidity.log
```

## Log Format

The service monitors all `*.log` files in `/opt/dexter-ai/` and formats them as JSON:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "message": "Liquidity management initialized",
  "source": "dexter.log",
  "raw": "2024-01-15T10:30:00 INFO: Liquidity management initialized"
}
```

### Log Level Detection

The service automatically detects log levels based on common patterns:
- `ERROR`, `ERR` → `ERROR`
- `WARN`, `WARNING` → `WARN`
- `DEBUG`, `DBG` → `DEBUG`
- `CRITICAL`, `CRIT`, `FATAL` → `CRITICAL`
- Default → `INFO`

## Frontend Integration

The Next.js frontend API route at `/app/api/logs/route.ts` has been updated to connect to port 3003:

```typescript
const response = await fetch('http://5.78.71.231:3003/logs', {
  method: 'GET',
  headers: {
    'Accept': 'text/event-stream',
    'Cache-Control': 'no-cache'
  }
});
```

## Configuration

### Environment Variables

The service accepts these environment variables:

- `PORT`: Server port (default: 3003)
- `LOG_DIR`: Directory to monitor (default: /opt/dexter-ai/)
- `NODE_ENV`: Environment mode (default: production)

### Systemd Service Configuration

The service is configured with:
- Automatic restart on failure
- Memory limit of 512MB
- Security restrictions
- Journal logging

## Monitoring

### Health Check

```bash
curl http://VPS_IP:3003/health
```

Returns:
```json
{
  "status": "healthy",
  "clients": 2,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "logDir": "/opt/dexter-ai/"
}
```

### Metrics

- **Active clients**: Number of connected SSE clients
- **Log files monitored**: Automatically discovered `.log` files
- **Service uptime**: Via systemd status
- **Memory usage**: Limited to 512MB

## Troubleshooting

### Service Not Starting

Check service status and logs:
```bash
sudo systemctl status dexter-log-stream
sudo journalctl -u dexter-log-stream -f
```

### No Log Files Found

Ensure log files exist:
```bash
ls -la /opt/dexter-ai/*.log
```

Create test log files:
```bash
echo "$(date) INFO: Test entry" >> /opt/dexter-ai/dexter.log
```

### Connection Issues

Check if port 3003 is open:
```bash
sudo netstat -tlnp | grep 3003
```

Test connection:
```bash
curl http://localhost:3003/health
```

### Permission Issues

Ensure proper permissions:
```bash
sudo chown -R root:root /opt/dexter-ai/
sudo chmod +x /opt/dexter-ai/log-stream-server.js
```

## Security

The service includes security measures:
- Runs with minimal privileges
- No new privileges allowed
- Private temporary directory
- Protected system access
- Read-only access to most filesystem
- Resource limits enforced

## Performance

- **Memory limit**: 512MB
- **File descriptors**: 65536
- **Client connections**: Unlimited (within memory constraints)
- **Keepalive interval**: 30 seconds
- **Reconnection delay**: 5 seconds

## Files

- `log-stream-server.js`: Main Node.js server
- `dexter-log-stream.service`: Systemd service configuration
- `deploy-log-stream.sh`: Deployment script
- `test-log-stream.sh`: Testing script
- `frontend/app/api/logs/route.ts`: Next.js API proxy route