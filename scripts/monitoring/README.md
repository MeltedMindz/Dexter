# Monitoring Scripts

## Purpose
The monitoring scripts directory contains utilities and scripts for setting up and managing monitoring infrastructure, including Grafana dashboards, Prometheus exporters, and monitoring tests.

## What Lives Here
- **grafana_quick_fix.sh** - Grafana setup and configuration script
- **test_grafana_setup.py** - Grafana connectivity tests
- **metrics_exporter.py** - Prometheus metrics exporter
- **import_dashboard.json** - Grafana dashboard configuration
- **docker-compose.monitoring.yml** - Monitoring stack Docker compose

## How It Fits Into the System
- **Interacts with**: Monitoring infrastructure (`../../monitoring/`), Prometheus, Grafana
- **Depends on**: Docker, Grafana, Prometheus
- **Provides**: Monitoring setup scripts, metrics export, dashboard configuration
- **Part of**: The scripts directory, providing operational utilities

## Current Status
✅ **Active / In use** - Monitoring scripts operational

## What This Is NOT
- This is not the monitoring configuration files (those are in `../../monitoring/`)
- This is not the monitoring infrastructure itself (that's external services)

## Relevant Docs / Entry Points
- **Monitoring setup**: `grafana_quick_fix.sh`
- **Metrics export**: `metrics_exporter.py`
- **Scripts documentation**: See `../README.md`
- **Monitoring documentation**: See `../../monitoring/README.md`
- **Root documentation**: See `../../README.md`

