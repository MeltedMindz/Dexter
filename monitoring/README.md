# Monitoring

## Purpose
The monitoring directory contains infrastructure configuration for observability of the Dexter Protocol system. This includes Prometheus metrics collection, Grafana dashboards, and monitoring setup for tracking system health, performance, and AI model metrics.

## What Lives Here
- **prometheus/** - Prometheus configuration files for metrics collection
- **grafana/** - Grafana provisioning configuration for dashboards and datasources
- **dexbrain-dashboard.json** - Pre-configured Grafana dashboard for DexBrain intelligence network

## How It Fits Into the System
- **Interacts with**: All backend services in `backend/` that expose metrics endpoints
- **Depends on**: Prometheus and Grafana services (typically run via Docker Compose)
- **Provides**: Real-time monitoring, alerting, and visualization of system performance
- **Part of**: The production infrastructure stack, enabling observability for the entire Dexter Protocol

## Current Status
✅ **Active / In use** - Monitoring infrastructure configured and operational for production services

## What This Is NOT
- This is not the monitoring scripts (those are in `scripts/monitoring/`)
- This is not the production monitoring configs (those are in `monitoring-configs/` which may be gitignored)
- This is not the actual running services (these are configuration files)

## Relevant Docs / Entry Points
- **Prometheus config**: `prometheus/prometheus.yml`
- **Grafana provisioning**: `grafana/provisioning/`
- **Dashboard**: `dexbrain-dashboard.json`
- **Root documentation**: See `../README.md`
- **Docker setup**: See `../docker-compose.streaming.yml` for service orchestration

## Setup
These configuration files are typically used with Docker Compose to set up the monitoring stack:
- Prometheus scrapes metrics from Dexter services
- Grafana visualizes metrics using the provided dashboards
- Dashboards are auto-provisioned on Grafana startup

## Key Metrics Tracked
- DexBrain API status and performance
- Active AI agents count
- API request rates
- Data submission rates
- Intelligence quality scores
- Blockchain coverage and network activity

