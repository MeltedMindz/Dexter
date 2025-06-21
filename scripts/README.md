# Scripts Directory

This directory contains organized utility and operational scripts for the Dexter project.

## Structure

### `/utilities/`
- **backup_system.py** - Database backup and recovery utilities
- **demo_alchemy_collection.py** - Alchemy API data collection demo
- **enhanced_alchemy_service.py** - Enhanced Alchemy SDK integration
- **run_integrated_learning.py** - ML training pipeline runner
- **simple_learning_demo.py** - Simple ML demonstration
- **uniswap_position_harvester.py** - Uniswap position data harvester
- **vercel-env-config.txt** - Vercel environment configuration template

### `/monitoring/`
- **grafana_quick_fix.sh** - Grafana setup and configuration script
- **test_grafana_setup.py** - Grafana connectivity tests
- **metrics_exporter.py** - Prometheus metrics exporter
- **import_dashboard.json** - Grafana dashboard configuration
- **docker-compose.monitoring.yml** - Monitoring stack Docker compose

### `/demos/`
- Demo scripts and examples (to be organized)

## Usage

All scripts should be run from the repository root directory:

```bash
# Run from repository root
python scripts/utilities/backup_system.py
bash scripts/monitoring/grafana_quick_fix.sh
```

## Security Note

These scripts may reference environment variables for API keys and credentials. Ensure proper environment configuration before running.