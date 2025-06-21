#!/usr/bin/env python3
"""
Test script to verify Grafana setup and metrics connectivity
"""

import requests
import json
import time
import sys

def test_prometheus_targets():
    """Test Prometheus targets"""
    print("🔍 Testing Prometheus targets...")
    
    try:
        response = requests.get("http://localhost:9090/api/v1/targets", timeout=10)
        if response.status_code == 200:
            data = response.json()
            targets = data.get('data', {}).get('activeTargets', [])
            
            dexter_targets = [t for t in targets if 'dexter' in t.get('labels', {}).get('job', '')]
            
            print(f"  ✅ Prometheus API accessible")
            print(f"  📊 Found {len(dexter_targets)} Dexter targets:")
            
            for target in dexter_targets:
                job = target.get('labels', {}).get('job', 'unknown')
                health = target.get('health', 'unknown')
                url = target.get('scrapeUrl', 'unknown')
                status = "✅" if health == "up" else "❌"
                print(f"    {status} {job}: {health} ({url})")
            
            return len([t for t in dexter_targets if t.get('health') == 'up']) > 0
        else:
            print(f"  ❌ Prometheus API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Prometheus connection failed: {e}")
        return False

def test_grafana_api():
    """Test Grafana API and datasources"""
    print("\n📊 Testing Grafana API...")
    
    try:
        # Test health
        response = requests.get("http://localhost:3000/api/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✅ Grafana health: {health.get('database', 'unknown')}")
            print(f"  📍 Version: {health.get('version', 'unknown')}")
        else:
            print(f"  ❌ Grafana health check failed: {response.status_code}")
            return False
        
        # Test datasources (basic auth admin:dexteradmin123)
        auth = ('admin', 'dexteradmin123')
        response = requests.get("http://localhost:3000/api/datasources", auth=auth, timeout=10)
        if response.status_code == 200:
            datasources = response.json()
            prometheus_ds = [ds for ds in datasources if ds.get('type') == 'prometheus']
            
            if prometheus_ds:
                print(f"  ✅ Prometheus datasource configured")
                for ds in prometheus_ds:
                    print(f"    📍 Name: {ds.get('name')}, URL: {ds.get('url')}")
            else:
                print(f"  ⚠️  No Prometheus datasource found")
        else:
            print(f"  ❌ Grafana datasources API error: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"  ❌ Grafana connection failed: {e}")
        return False

def test_metrics_endpoint():
    """Test custom metrics endpoint"""
    print("\n🔧 Testing custom metrics endpoint...")
    
    try:
        response = requests.get("http://localhost:9091/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.text
            dexter_metrics = [line for line in metrics.split('\n') if 'dexter_' in line and not line.startswith('#')]
            
            print(f"  ✅ Metrics endpoint accessible")
            print(f"  📊 Found {len(dexter_metrics)} Dexter metrics")
            
            # Show sample metrics
            for metric in dexter_metrics[:5]:
                if metric.strip():
                    print(f"    📍 {metric.strip()}")
            
            return len(dexter_metrics) > 0
        else:
            print(f"  ❌ Metrics endpoint error: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Metrics endpoint connection failed: {e}")
        return False

def test_health_endpoint():
    """Test health endpoint"""
    print("\n💓 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:9091/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✅ Health endpoint accessible")
            print(f"  📍 Status: {health.get('status', 'unknown')}")
            print(f"  📍 Timestamp: {health.get('timestamp', 'unknown')}")
            return health.get('status') == 'healthy'
        else:
            print(f"  ❌ Health endpoint error: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Health endpoint connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Dexter AI Grafana Setup Test")
    print("=" * 50)
    
    tests = [
        ("Metrics Endpoint", test_metrics_endpoint),
        ("Health Endpoint", test_health_endpoint),
        ("Prometheus Targets", test_prometheus_targets),
        ("Grafana API", test_grafana_api),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Grafana setup is ready.")
        print("\n📋 Next Steps:")
        print("  1. Access Grafana at: http://YOUR_VPS_IP:3000")
        print("  2. Login with: admin / dexteradmin123")
        print("  3. Navigate to 'Dexter AI - Comprehensive Dashboard'")
        print("  4. Verify metrics are displaying correctly")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Check the issues above.")
        
        if not results.get("Prometheus Targets", False):
            print("\n🔧 Troubleshooting Prometheus:")
            print("  - Check if Prometheus container is running")
            print("  - Verify prometheus.yml configuration")
            print("  - Ensure firewall allows port 9090")
        
        if not results.get("Metrics Endpoint", False):
            print("\n🔧 Troubleshooting Metrics:")
            print("  - Check if dexter-metrics-exporter service is running")
            print("  - Verify port 9091 is accessible")
            print("  - Check service logs: journalctl -u dexter-metrics-exporter")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)