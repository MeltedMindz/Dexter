#!/usr/bin/env python3
"""
Enhanced ML Deployment Script for Dexter AI
Deploys advanced ML models and continuous learning system to VPS
"""

import asyncio
import subprocess
import logging
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMLDeployment:
    """
    Deployment manager for enhanced ML components
    """
    
    def __init__(self):
        self.vps_ip = "5.78.71.231"
        self.vps_user = "root"
        self.local_repo_path = Path(__file__).parent.parent
        self.remote_path = "/opt/dexter-ai"
        
        # Component paths
        self.components = {
            'enhanced_ml_models': 'backend/dexbrain/models/enhanced_ml_models.py',
            'training_pipeline': 'backend/dexbrain/training_pipeline.py',
            'enhanced_continuous_learning': 'backend/dexbrain/enhanced_continuous_learning.py',
            'advanced_uniswap_fetcher': 'dexter-liquidity/data/fetchers/advanced_uniswap_fetcher.py',
            'fetcher_init': 'dexter-liquidity/data/fetchers/__init__.py'
        }
        
        self.api_key = "c6f241c1dd5aea81977a63b2614af70d"
        self.subgraph_url = "https://gateway-arbitrum.network.thegraph.com/api/c6f241c1dd5aea81977a63b2614af70d/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
    
    async def deploy_enhanced_components(self):
        """
        Deploy all enhanced ML components to VPS
        """
        logger.info("🚀 Starting enhanced ML deployment to VPS...")
        
        try:
            # Step 1: Upload new ML components
            await self._upload_components()
            
            # Step 2: Install additional Python dependencies
            await self._install_dependencies()
            
            # Step 3: Create ML model storage directories
            await self._setup_model_storage()
            
            # Step 4: Deploy enhanced continuous learning service
            await self._deploy_enhanced_learning_service()
            
            # Step 5: Test ML integration
            await self._test_ml_integration()
            
            # Step 6: Start enhanced system
            await self._start_enhanced_system()
            
            logger.info("✅ Enhanced ML deployment completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            raise
    
    async def _upload_components(self):
        """Upload new ML components to VPS"""
        logger.info("📤 Uploading enhanced ML components...")
        
        for name, path in self.components.items():
            local_file = self.local_repo_path / path
            remote_file = f"{self.remote_path}/{path}"
            
            if local_file.exists():
                logger.info(f"  Uploading {name}...")
                
                # Create remote directory if needed
                remote_dir = str(Path(remote_file).parent)
                await self._run_ssh_command(f"mkdir -p {remote_dir}")
                
                # Upload file
                await self._run_scp_command(str(local_file), remote_file)
            else:
                logger.warning(f"  ⚠️  Local file not found: {local_file}")
    
    async def _install_dependencies(self):
        """Install additional Python dependencies for ML"""
        logger.info("📦 Installing ML dependencies...")
        
        # Additional ML dependencies
        ml_packages = [
            "torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu",
            "scikit-learn>=1.3.0", 
            "numpy>=1.24.0",
            "joblib>=1.3.0",
            "aiohttp>=3.8.0"
        ]
        
        for package in ml_packages:
            logger.info(f"  Installing {package}...")
            timeout = 300 if "torch" in package else 60  # 5 minutes for torch, 1 minute for others
            await self._run_ssh_command(f"cd {self.remote_path} && pip install {package} --break-system-packages", timeout=timeout)
    
    async def _setup_model_storage(self):
        """Create model storage directories"""
        logger.info("📁 Setting up ML model storage...")
        
        directories = [
            f"{self.remote_path}/model_storage",
            f"{self.remote_path}/model_storage/uniswap_optimizer",
            f"{self.remote_path}/knowledge_base",
            "/var/log/dexter"
        ]
        
        for directory in directories:
            await self._run_ssh_command(f"mkdir -p {directory}")
            await self._run_ssh_command(f"chmod 755 {directory}")
    
    async def _deploy_enhanced_learning_service(self):
        """Deploy enhanced continuous learning as systemd service"""
        logger.info("🔧 Deploying enhanced learning service...")
        
        # Create systemd service file
        service_content = f"""[Unit]
Description=Dexter Enhanced Continuous Learning System
After=network.target
Requires=network.target

[Service]
Type=simple
User=root
WorkingDirectory={self.remote_path}
Environment=PYTHONPATH={self.remote_path}
ExecStart=/usr/bin/python3 -m backend.dexbrain.enhanced_continuous_learning
Restart=always
RestartSec=10
StandardOutput=append:/var/log/dexter/enhanced_learning.log
StandardError=append:/var/log/dexter/enhanced_learning_error.log

[Install]
WantedBy=multi-user.target
"""
        
        # Write service file
        service_file = "/tmp/dexter-enhanced-learning.service"
        escaped_content = service_content.replace("'", "'\"'\"'")
        await self._run_ssh_command(f"cat > {service_file} << 'EOF'\n{service_content}EOF")
        
        # Install service
        await self._run_ssh_command(f"mv {service_file} /etc/systemd/system/")
        await self._run_ssh_command("systemctl daemon-reload")
        await self._run_ssh_command("systemctl enable dexter-enhanced-learning.service")
    
    async def _test_ml_integration(self):
        """Test ML model integration"""
        logger.info("🧪 Testing ML integration...")
        
        # Create test script  
        test_script = f"""import sys
sys.path.insert(0, "{self.remote_path}")

try:
    from backend.dexbrain.models.enhanced_ml_models import UniswapLPOptimizer
    from backend.dexbrain.training_pipeline import AdvancedTrainingPipeline
    from backend.dexbrain.models.knowledge_base import KnowledgeBase
    
    print("✅ Enhanced ML models imported successfully")
    
    # Test ML optimizer initialization
    optimizer = UniswapLPOptimizer()
    print("✅ ML optimizer initialized")
    
    # Test knowledge base
    kb = KnowledgeBase()
    print("✅ Knowledge base initialized")
    
    # Test training pipeline
    pipeline = AdvancedTrainingPipeline(kb)
    print("✅ Training pipeline initialized")
    
    print("🎉 All ML components working correctly!")
    
except Exception as e:
    print(f"❌ ML integration test failed: {{e}}")
    sys.exit(1)
"""
        
        # Write test script to file
        test_file = "/tmp/test_ml_integration.py"
        await self._run_ssh_command(f"cat > {test_file} << 'EOF'\n{test_script}EOF")
        
        # Run test
        await self._run_ssh_command(f"cd {self.remote_path} && python3 {test_file}")
    
    async def _start_enhanced_system(self):
        """Start the enhanced learning system"""
        logger.info("🚀 Starting enhanced learning system...")
        
        # Stop any existing service
        await self._run_ssh_command("systemctl stop dexter-enhanced-learning.service", ignore_errors=True)
        
        # Start enhanced service
        await self._run_ssh_command("systemctl start dexter-enhanced-learning.service")
        
        # Check status
        result = await self._run_ssh_command("systemctl status dexter-enhanced-learning.service --no-pager -l", capture_output=True)
        logger.info(f"Service status: {result}")
        
        # Wait a moment and check logs
        await asyncio.sleep(5)
        logs = await self._run_ssh_command("tail -20 /var/log/dexter/enhanced_learning.log", capture_output=True)
        logger.info(f"Recent logs:\n{logs}")
    
    async def _run_ssh_command(self, command: str, ignore_errors: bool = False, capture_output: bool = False, timeout: int = 60) -> str:
        """Run SSH command on VPS"""
        ssh_cmd = f"ssh {self.vps_user}@{self.vps_ip} '{command}'"
        
        try:
            if capture_output:
                result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
                if result.returncode != 0 and not ignore_errors:
                    raise subprocess.CalledProcessError(result.returncode, ssh_cmd, result.stderr)
                return result.stdout.strip()
            else:
                result = subprocess.run(ssh_cmd, shell=True, timeout=timeout)
                if result.returncode != 0 and not ignore_errors:
                    raise subprocess.CalledProcessError(result.returncode, ssh_cmd)
                return ""
        except subprocess.TimeoutExpired:
            logger.error(f"SSH command timed out: {command}")
            raise
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                logger.error(f"SSH command failed: {command} - {e}")
                raise
            return ""
    
    async def _run_scp_command(self, local_path: str, remote_path: str):
        """Run SCP command to upload file"""
        scp_cmd = f"scp {local_path} {self.vps_user}@{self.vps_ip}:{remote_path}"
        
        try:
            result = subprocess.run(scp_cmd, shell=True, timeout=120)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, scp_cmd)
        except subprocess.TimeoutExpired:
            logger.error(f"SCP command timed out: {scp_cmd}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"SCP command failed: {scp_cmd} - {e}")
            raise
    
    async def check_deployment_status(self):
        """Check status of deployed enhanced ML system"""
        logger.info("📊 Checking enhanced ML deployment status...")
        
        try:
            # Check service status
            service_status = await self._run_ssh_command(
                "systemctl is-active dexter-enhanced-learning.service", 
                capture_output=True, 
                ignore_errors=True
            )
            
            # Check recent logs
            recent_logs = await self._run_ssh_command(
                "tail -10 /var/log/dexter/enhanced_learning.log", 
                capture_output=True, 
                ignore_errors=True
            )
            
            # Check website logs
            website_logs = await self._run_ssh_command(
                "tail -5 /var/log/dexter/liquidity.log | grep DexBrain", 
                capture_output=True, 
                ignore_errors=True
            )
            
            # Check model files
            model_files = await self._run_ssh_command(
                f"ls -la {self.remote_path}/model_storage/", 
                capture_output=True, 
                ignore_errors=True
            )
            
            status_report = {
                'timestamp': datetime.now().isoformat(),
                'service_status': service_status,
                'recent_logs': recent_logs.split('\n')[-5:] if recent_logs else [],
                'website_logs': website_logs.split('\n')[-3:] if website_logs else [],
                'model_files_present': 'checkpoint.pth' in model_files,
                'deployment_healthy': service_status == 'active'
            }
            
            logger.info(f"📈 Deployment Status: {json.dumps(status_report, indent=2)}")
            return status_report
            
        except Exception as e:
            logger.error(f"Error checking deployment status: {e}")
            return {'error': str(e)}
    
    async def restart_enhanced_system(self):
        """Restart the enhanced learning system"""
        logger.info("🔄 Restarting enhanced learning system...")
        
        await self._run_ssh_command("systemctl restart dexter-enhanced-learning.service")
        await asyncio.sleep(3)
        
        status = await self.check_deployment_status()
        return status


async def main():
    """Main deployment function"""
    deployment = EnhancedMLDeployment()
    
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "deploy":
            await deployment.deploy_enhanced_components()
        elif command == "status":
            await deployment.check_deployment_status()
        elif command == "restart":
            await deployment.restart_enhanced_system()
        else:
            print("Usage: python deploy_enhanced_ml.py [deploy|status|restart]")
    else:
        # Default: deploy
        await deployment.deploy_enhanced_components()


if __name__ == "__main__":
    asyncio.run(main())