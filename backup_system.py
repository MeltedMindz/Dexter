#!/usr/bin/env python3
"""
Dexter AI Backup System
Automated backup of configurations, data, and logs
"""

import os
import subprocess
import datetime
import json
import logging
import asyncio
import tarfile
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DexterBackupSystem:
    """Comprehensive backup system for Dexter AI infrastructure"""
    
    def __init__(self):
        self.backup_root = "/opt/dexter-ai/backups"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = f"{self.backup_root}/{self.timestamp}"
        
        # Ensure backup directory exists
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Backup system initialized: {self.backup_dir}")
    
    async def run_full_backup(self):
        """Run complete backup of all Dexter components"""
        logger.info("🔄 Starting full Dexter AI backup...")
        
        backup_manifest = {
            "timestamp": self.timestamp,
            "backup_type": "full",
            "components": [],
            "status": "in_progress"
        }
        
        try:
            # 1. Backup configurations
            await self._backup_configurations()
            backup_manifest["components"].append("configurations")
            
            # 2. Backup databases
            await self._backup_databases()
            backup_manifest["components"].append("databases")
            
            # 3. Backup logs
            await self._backup_logs()
            backup_manifest["components"].append("logs")
            
            # 4. Backup Docker volumes
            await self._backup_docker_volumes()
            backup_manifest["components"].append("docker_volumes")
            
            # 5. Backup service configurations
            await self._backup_systemd_services()
            backup_manifest["components"].append("systemd_services")
            
            # 6. Create system snapshot
            await self._create_system_snapshot()
            backup_manifest["components"].append("system_snapshot")
            
            # 7. Compress backup
            await self._compress_backup()
            backup_manifest["components"].append("compression")
            
            backup_manifest["status"] = "completed"
            backup_manifest["size_mb"] = await self._get_backup_size()
            
            # Save manifest
            with open(f"{self.backup_dir}/manifest.json", "w") as f:
                json.dump(backup_manifest, f, indent=2)
            
            logger.info(f"✅ Full backup completed: {backup_manifest['size_mb']} MB")
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            backup_manifest["status"] = "failed"
            backup_manifest["error"] = str(e)
            
            with open(f"{self.backup_dir}/manifest.json", "w") as f:
                json.dump(backup_manifest, f, indent=2)
            raise
    
    async def _backup_configurations(self):
        """Backup all configuration files"""
        logger.info("📄 Backing up configurations...")
        
        config_dir = f"{self.backup_dir}/configurations"
        Path(config_dir).mkdir(exist_ok=True)
        
        # Dexter AI configurations
        configs_to_backup = [
            "/opt/dexter-ai",
            "/etc/systemd/system/dexter-*.service",
            "/var/log/dexter"
        ]
        
        for config_path in configs_to_backup:
            if os.path.exists(config_path):
                if os.path.isdir(config_path):
                    shutil.copytree(config_path, f"{config_dir}/{Path(config_path).name}", dirs_exist_ok=True)
                else:
                    # Handle glob patterns
                    subprocess.run(f"cp {config_path} {config_dir}/", shell=True, check=False)
        
        logger.info("✅ Configurations backed up")
    
    async def _backup_databases(self):
        """Backup databases (if any)"""
        logger.info("🗄️ Backing up databases...")
        
        db_dir = f"{self.backup_dir}/databases"
        Path(db_dir).mkdir(exist_ok=True)
        
        # Check for PostgreSQL
        try:
            result = subprocess.run(["which", "pg_dump"], capture_output=True, text=True)
            if result.returncode == 0:
                # PostgreSQL backup
                subprocess.run([
                    "pg_dumpall", "-U", "postgres", "-f", f"{db_dir}/postgres_backup.sql"
                ], check=False)
                logger.info("PostgreSQL backup created")
        except Exception as e:
            logger.warning(f"PostgreSQL backup failed: {e}")
        
        # Check for Redis data
        redis_data = "/var/lib/redis"
        if os.path.exists(redis_data):
            shutil.copytree(redis_data, f"{db_dir}/redis", dirs_exist_ok=True)
            logger.info("Redis data backed up")
        
        logger.info("✅ Databases backed up")
    
    async def _backup_logs(self):
        """Backup log files"""
        logger.info("📋 Backing up logs...")
        
        logs_dir = f"{self.backup_dir}/logs"
        Path(logs_dir).mkdir(exist_ok=True)
        
        # System logs
        log_files = [
            "/var/log/dexter/liquidity.log",
            "/var/log/syslog",
            "/var/log/auth.log"
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                shutil.copy2(log_file, logs_dir)
        
        # Journal logs for Dexter services
        try:
            services = [
                "dexter-position-harvester",
                "dexter-enhanced-alchemy", 
                "dexter-analysis",
                "dexter-metrics-exporter"
            ]
            
            for service in services:
                subprocess.run([
                    "journalctl", "-u", f"{service}.service", "--no-pager", 
                    "-o", "json", "--since", "1 week ago"
                ], stdout=open(f"{logs_dir}/{service}_journal.json", "w"), check=False)
        except Exception as e:
            logger.warning(f"Journal backup failed: {e}")
        
        logger.info("✅ Logs backed up")
    
    async def _backup_docker_volumes(self):
        """Backup Docker volumes"""
        logger.info("🐳 Backing up Docker volumes...")
        
        docker_dir = f"{self.backup_dir}/docker"
        Path(docker_dir).mkdir(exist_ok=True)
        
        try:
            # Get list of Docker volumes
            result = subprocess.run(
                ["docker", "volume", "ls", "-q"], 
                capture_output=True, text=True, check=True
            )
            
            volumes = result.stdout.strip().split('\n')
            
            for volume in volumes:
                if volume and 'dexter' in volume:
                    # Create volume backup
                    subprocess.run([
                        "docker", "run", "--rm", 
                        "-v", f"{volume}:/source:ro",
                        "-v", f"{docker_dir}:/backup",
                        "alpine:latest",
                        "tar", "-czf", f"/backup/{volume}.tar.gz", "-C", "/source", "."
                    ], check=False)
                    
                    logger.info(f"Backed up Docker volume: {volume}")
        
        except Exception as e:
            logger.warning(f"Docker volume backup failed: {e}")
        
        logger.info("✅ Docker volumes backed up")
    
    async def _backup_systemd_services(self):
        """Backup systemd service files"""
        logger.info("⚙️ Backing up systemd services...")
        
        services_dir = f"{self.backup_dir}/systemd"
        Path(services_dir).mkdir(exist_ok=True)
        
        # Find all Dexter services
        try:
            result = subprocess.run(
                ["find", "/etc/systemd/system", "-name", "dexter-*.service"],
                capture_output=True, text=True, check=True
            )
            
            service_files = result.stdout.strip().split('\n')
            
            for service_file in service_files:
                if service_file and os.path.exists(service_file):
                    shutil.copy2(service_file, services_dir)
                    logger.info(f"Backed up service: {Path(service_file).name}")
        
        except Exception as e:
            logger.warning(f"Systemd backup failed: {e}")
        
        logger.info("✅ Systemd services backed up")
    
    async def _create_system_snapshot(self):
        """Create system configuration snapshot"""
        logger.info("📸 Creating system snapshot...")
        
        snapshot_dir = f"{self.backup_dir}/system_snapshot"
        Path(snapshot_dir).mkdir(exist_ok=True)
        
        # System information
        system_info = {
            "timestamp": self.timestamp,
            "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
            "kernel": subprocess.run(["uname", "-r"], capture_output=True, text=True).stdout.strip(),
            "os_release": {},
            "installed_packages": [],
            "running_services": [],
            "docker_containers": [],
            "disk_usage": {},
            "memory_info": {},
            "network_config": {}
        }
        
        try:
            # OS release info
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release") as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            system_info["os_release"][key] = value.strip('"')
            
            # Installed packages (dpkg)
            try:
                result = subprocess.run(
                    ["dpkg", "-l"], capture_output=True, text=True, check=True
                )
                system_info["installed_packages"] = result.stdout.split('\n')[:100]  # First 100 lines
            except:
                pass
            
            # Running services
            try:
                result = subprocess.run(
                    ["systemctl", "list-units", "--type=service", "--state=active", "--no-pager"],
                    capture_output=True, text=True, check=True
                )
                system_info["running_services"] = result.stdout.split('\n')[:50]  # First 50 lines
            except:
                pass
            
            # Docker containers
            try:
                result = subprocess.run(
                    ["docker", "ps", "--format", "json"], capture_output=True, text=True, check=True
                )
                system_info["docker_containers"] = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
            except:
                pass
            
            # Save system info
            with open(f"{snapshot_dir}/system_info.json", "w") as f:
                json.dump(system_info, f, indent=2)
        
        except Exception as e:
            logger.warning(f"System snapshot failed: {e}")
        
        logger.info("✅ System snapshot created")
    
    async def _compress_backup(self):
        """Compress the backup directory"""
        logger.info("🗜️ Compressing backup...")
        
        archive_path = f"{self.backup_root}/dexter_backup_{self.timestamp}.tar.gz"
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(self.backup_dir, arcname=f"dexter_backup_{self.timestamp}")
        
        # Remove uncompressed directory
        shutil.rmtree(self.backup_dir)
        
        logger.info(f"✅ Backup compressed: {archive_path}")
    
    async def _get_backup_size(self):
        """Get backup size in MB"""
        try:
            archive_path = f"{self.backup_root}/dexter_backup_{self.timestamp}.tar.gz"
            if os.path.exists(archive_path):
                size_bytes = os.path.getsize(archive_path)
                return round(size_bytes / (1024 * 1024), 2)
            else:
                # Calculate directory size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(self.backup_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                return round(total_size / (1024 * 1024), 2)
        except:
            return 0
    
    async def _cleanup_old_backups(self, keep_count=7):
        """Keep only the latest N backups"""
        try:
            backup_files = []
            for file in os.listdir(self.backup_root):
                if file.startswith("dexter_backup_") and file.endswith(".tar.gz"):
                    filepath = os.path.join(self.backup_root, file)
                    backup_files.append((filepath, os.path.getctime(filepath)))
            
            # Sort by creation time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            for filepath, _ in backup_files[keep_count:]:
                os.remove(filepath)
                logger.info(f"Removed old backup: {Path(filepath).name}")
        
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

async def main():
    """Run backup system"""
    backup_system = DexterBackupSystem()
    await backup_system.run_full_backup()

if __name__ == "__main__":
    asyncio.run(main())