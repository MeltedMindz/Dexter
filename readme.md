# DexBrain - AI-Powered DLMM Analytics

DexBrain provides real-time analytics and AI-generated strategies for Dynamic Liquidity Market Makers (DLMMs) on Solana. This guide will help you set up the system for both local development and production deployment.

## Project Overview

DexBrain consists of several components working together:

The backend provides the core analytics engine, featuring protocol adapters for different DLMMs, an AI strategy generator, and a REST API. The frontend offers an intuitive dashboard for visualizing pool data and strategies. The system uses PostgreSQL for persistent storage, Redis for caching and rate limiting, and includes comprehensive monitoring with Prometheus and Grafana.

## Local Development Setup

### Prerequisites

You will need these tools installed on your development machine:

- Python 3.9 or higher
- Node.js 16 or higher
- Docker and Docker Compose
- PostgreSQL 14 (if running outside Docker)
- Redis 7 (if running outside Docker)

### Step 1: Clone and Configure

First, clone the repository and set up your environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/dexbrain.git
cd dexbrain

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create local environment file
cp .env.example .env
```

Edit .env with your configurations:

```env
# Database
DB_PASSWORD=your_secure_password
POSTGRES_URL=postgresql://postgres:your_secure_password@localhost:5432/dexbrain

# Redis
REDIS_URL=redis://localhost:6379/0

# Solana
SOLANA_RPC_URL=https://your-rpc-endpoint.com

# Security
SECRET_KEY=your_secure_secret_key

# Monitoring
GRAFANA_PASSWORD=your_secure_password
SLACK_WEBHOOK_URL=your_slack_webhook  # Optional
```

### Step 2: Database Setup

Initialize the database and populate test data:

```bash
# Start database services
docker-compose up -d db redis

# Run migrations
python -m alembic upgrade head

# Seed test data
python scripts/seed_database.py
```

The seed script will generate API keys for testing. Save these for logging into the dashboard.

### Step 3: Frontend Setup

Set up the frontend development environment:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Step 4: Start Backend Services

In a new terminal:

```bash
# Start the API server in development mode
python -m uvicorn app.main:app --reload --port 8000
```

### Step 5: Access Local Environment

Your local development environment is now ready:

- Frontend Dashboard: http://localhost:3000 
- API Documentation: http://localhost:8000/docs
- Grafana Dashboards: http://localhost:3000/grafana
- Prometheus: http://localhost:9090

## Production Deployment

### Prerequisites

- A Linux server with Docker and Docker Compose installed
- Domain name with SSL certificate
- Solana RPC endpoint with sufficient rate limits
- (Optional) Monitoring setup (Slack, PagerDuty, etc.)

### Step 1: Server Setup

Prepare your server:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker if not present
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Step 2: Application Deployment

Deploy the application:

```bash
# Clone repository
git clone https://github.com/yourusername/dexbrain.git
cd dexbrain

# Configure production environment
cp .env.example .env
nano .env  # Edit with production values

# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Run database migrations
docker-compose exec api python -m alembic upgrade head

# Create initial admin user
docker-compose exec api python scripts/create_admin.py
```

### Step 3: Configure Nginx

Set up Nginx as a reverse proxy:

```bash
# Install Nginx
sudo apt install nginx

# Configure SSL with Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### Step 4: Monitoring Setup

Configure monitoring alerts:

1. Update alertmanager.yml with your notification channels
2. Configure Grafana dashboards
3. Set up uptime monitoring (e.g., UptimeRobot, Pingdom)

## Maintenance and Troubleshooting

### Common Issues

Database Connection Errors:
```bash
# Check database logs
docker-compose logs db

# Verify database connection
docker-compose exec api python -c "from app.db import get_db; print(get_db().is_connected())"
```

Rate Limiting Issues:
```bash
# Check Redis rate limit keys
docker-compose exec redis redis-cli keys "ratelimit:*"

# Clear rate limit cache if needed
docker-compose exec redis redis-cli FLUSHDB
```

### Backup and Restore

Create database backups:
```bash
# Backup
docker-compose exec db pg_dump -U postgres dexbrain > backup.sql

# Restore
cat backup.sql | docker-compose exec -T db psql -U postgres dexbrain
```

### Updating the Application

To update to a new version:

```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose -f docker-compose.prod.yml build

# Update with zero downtime
docker-compose -f docker-compose.prod.yml up -d --no-deps --build api
```

### Health Checks

Monitor system health:

```bash
# Check service status
docker-compose ps

# View API logs
docker-compose logs -f api

# Monitor system metrics
docker stats
```

## Security Considerations

- Keep your .env file secure and never commit it to version control
- Regularly update dependencies and system packages
- Monitor system logs for suspicious activity
- Use strong passwords for all services
- Implement regular backup procedures
- Keep your SSL certificates up to date

## Getting Help

If you encounter issues:

1. Check the logs using `docker-compose logs`
2. Review the troubleshooting section above
3. Check the GitHub issues for similar problems
4. Open a new issue with detailed information about your problem

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
