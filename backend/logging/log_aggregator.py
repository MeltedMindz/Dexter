#!/usr/bin/env python3
"""
Real-time Log Aggregator for Dexter Protocol
Streams structured logs to frontend BrainWindow component
"""

import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from flask import Flask, jsonify, Response
from flask_cors import CORS
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from collections import deque

app = Flask(__name__)
CORS(app)

# Global log buffer
LOG_BUFFER = deque(maxlen=1000)
LOG_LOCK = threading.Lock()

class LogHandler(FileSystemEventHandler):
    """Handles file system events for log files"""
    
    def __init__(self, log_aggregator):
        self.log_aggregator = log_aggregator
        self.last_positions = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.log'):
            self.log_aggregator.process_log_file(event.src_path)

class LogAggregator:
    """Aggregates logs from multiple sources"""
    
    def __init__(self):
        self.log_directory = '/opt/dexter-ai'
        self.observer = Observer()
        self.handler = LogHandler(self)
        self.file_positions = {}
        
    def start_monitoring(self):
        """Start monitoring log files"""
        if os.path.exists(self.log_directory):
            self.observer.schedule(self.handler, self.log_directory, recursive=False)
            self.observer.start()
            print(f"📁 Started monitoring {self.log_directory}")
        else:
            print(f"⚠️ Log directory {self.log_directory} does not exist")
        
        # Process existing log files
        self.process_existing_logs()
    
    def process_existing_logs(self):
        """Process existing log files on startup"""
        if not os.path.exists(self.log_directory):
            return
        
        for filename in os.listdir(self.log_directory):
            if filename.endswith('.log'):
                file_path = os.path.join(self.log_directory, filename)
                self.process_log_file(file_path, initial=True)
    
    def process_log_file(self, file_path: str, initial: bool = False):
        """Process a log file and extract new entries"""
        try:
            if not os.path.exists(file_path):
                return
            
            # Get file size
            current_size = os.path.getsize(file_path)
            last_position = self.file_positions.get(file_path, 0)
            
            # If initial load, only read last 10KB
            if initial:
                last_position = max(0, current_size - 10240)
            
            if current_size <= last_position:
                return
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_position)
                new_content = f.read()
                
                # Process each line
                for line in new_content.strip().split('\\n'):
                    if line.strip():
                        self.parse_log_line(line.strip(), file_path)
                
                # Update file position
                self.file_positions[file_path] = f.tell()
                
        except Exception as e:
            print(f"❌ Error processing log file {file_path}: {e}")
    
    def parse_log_line(self, line: str, file_path: str):
        """Parse a log line and add to buffer"""
        try:
            # Try to parse as JSON (structured log)
            try:
                log_entry = json.loads(line)
                if isinstance(log_entry, dict):
                    self.add_log_entry(log_entry)
                    return
            except json.JSONDecodeError:
                pass
            
            # Parse as plain text log
            timestamp = datetime.now().isoformat()
            log_type = 'log'
            module = os.path.basename(file_path).replace('.log', '')
            
            # Detect log type from content
            if 'ERROR' in line.upper() or 'error' in line.lower():
                log_type = 'error'
            elif 'COMPOUND_SUCCESS' in line:
                log_type = 'compound_success'
            elif 'Strategy prediction complete' in line:
                log_type = 'vault_strategy'
            elif 'Dual position optimization' in line:
                log_type = 'vault_optimization'
            elif 'Generated vault intelligence' in line:
                log_type = 'vault_intelligence'
            elif 'compound opportunities' in line.lower():
                log_type = 'compound_opportunities'
            elif 'Intelligence served' in line:
                log_type = 'intelligence_feed'
            
            log_entry = {
                'timestamp': timestamp,
                'level': 'INFO',
                'module': module,
                'message': line,
                'type': log_type
            }
            
            self.add_log_entry(log_entry)
            
        except Exception as e:
            print(f"❌ Error parsing log line: {e}")
    
    def add_log_entry(self, log_entry: Dict[str, Any]):
        """Add log entry to buffer"""
        with LOG_LOCK:
            LOG_BUFFER.append(log_entry)
        
        print(f"📝 Added log entry: {log_entry.get('type', 'log')} - {log_entry.get('message', '')[:100]}")
    
    def get_recent_logs(self, limit: int = 100, log_type: str = 'all') -> List[Dict[str, Any]]:
        """Get recent logs from buffer"""
        with LOG_LOCK:
            logs = list(LOG_BUFFER)
        
        # Filter by type if specified
        if log_type != 'all':
            logs = [log for log in logs if log.get('type') == log_type]
        
        # Sort by timestamp (most recent first)
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return logs[:limit]

# Global log aggregator instance
log_aggregator = LogAggregator()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'log_buffer_size': len(LOG_BUFFER)
    })

@app.route('/api/logs/recent')
def get_recent_logs():
    """Get recent logs endpoint"""
    try:
        limit = min(int(request.args.get('limit', 100)), 500)
        log_type = request.args.get('type', 'all')
        since_timestamp = request.args.get('since')
        
        logs = log_aggregator.get_recent_logs(limit, log_type)
        
        # Filter by timestamp if provided
        if since_timestamp:
            try:
                since_dt = datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
                logs = [
                    log for log in logs 
                    if datetime.fromisoformat(log['timestamp']) > since_dt
                ]
            except ValueError:
                pass  # Invalid timestamp format, ignore filter
        
        return jsonify({
            'logs': logs,
            'total_count': len(logs),
            'filters': {
                'limit': limit,
                'type': log_type,
                'since': since_timestamp
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get logs: {str(e)}',
            'logs': []
        }), 500

@app.route('/api/logs/stream')
def stream_logs():
    """Server-sent events endpoint for real-time log streaming"""
    def generate():
        last_count = len(LOG_BUFFER)
        
        while True:
            current_count = len(LOG_BUFFER)
            
            if current_count > last_count:
                # Get new logs
                with LOG_LOCK:
                    new_logs = list(LOG_BUFFER)[last_count:]
                
                for log in new_logs:
                    yield f"data: {json.dumps(log)}\\n\\n"
                
                last_count = current_count
            
            time.sleep(1)  # Check for new logs every second
    
    return Response(generate(), mimetype='text/plain')

@app.route('/api/logs/demo')
def get_demo_logs():
    """Generate demo logs for testing"""
    current_time = datetime.now()
    
    demo_logs = [
        {
            'timestamp': (current_time - timedelta(minutes=2)).isoformat(),
            'level': 'INFO',
            'module': 'VaultMLEngine',
            'message': 'Strategy prediction complete | Recommended: gamma_balanced | Confidence: 87% | Expected APR: 18.5% | Ranges: 2',
            'type': 'vault_strategy',
            'metadata': {
                'strategy': 'gamma_balanced',
                'confidence': 0.87,
                'expected_apr': 0.185,
                'ranges_count': 2
            }
        },
        {
            'timestamp': (current_time - timedelta(minutes=3)).isoformat(),
            'level': 'INFO',
            'module': 'CompoundService',
            'message': 'COMPOUND_SUCCESS | Token ID: 12345 | TX Hash: 0xabc...def | Gas Used: 150,000 | Net Profit: $45.67',
            'type': 'compound_success',
            'metadata': {
                'token_id': 12345,
                'tx_hash': '0xabc...def',
                'gas_used': 150000,
                'net_profit': 45.67
            }
        },
        {
            'timestamp': (current_time - timedelta(minutes=5)).isoformat(),
            'level': 'INFO',
            'module': 'GammaStyleOptimizer',
            'message': 'Dual position optimization complete | Base Range: [98000, 102000] (75%) | Limit Range: [99500, 100500] (25%)',
            'type': 'vault_optimization',
            'metadata': {
                'base_range': [98000, 102000],
                'base_allocation': 0.75,
                'limit_range': [99500, 100500],
                'limit_allocation': 0.25
            }
        },
        {
            'timestamp': (current_time - timedelta(minutes=1)).isoformat(),
            'level': 'INFO',
            'module': 'DexBrain',
            'message': 'Generated vault intelligence for 0x123...abc | Strategy: gamma_balanced | Confidence: 87% | Compound Opportunities: 5',
            'type': 'vault_intelligence',
            'metadata': {
                'vault_address': '0x123...abc',
                'strategy': 'gamma_balanced',
                'confidence': 0.87,
                'compound_opportunities': 5
            }
        }
    ]
    
    # Add demo logs to buffer
    for log in demo_logs:
        log_aggregator.add_log_entry(log)
    
    return jsonify({
        'logs': demo_logs,
        'message': 'Demo logs added to buffer'
    })

def main():
    """Main function"""
    print("🚀 Starting Dexter Log Aggregator")
    
    # Start log monitoring
    log_aggregator.start_monitoring()
    
    # Add some initial demo logs if no real logs exist
    if len(LOG_BUFFER) == 0:
        print("📝 Adding initial demo logs")
        get_demo_logs()
    
    try:
        app.run(host='0.0.0.0', port=8084, debug=False)
    except KeyboardInterrupt:
        print("🛑 Shutting down log aggregator")
        log_aggregator.observer.stop()
        log_aggregator.observer.join()

if __name__ == "__main__":
    # Add Flask request import for the endpoint
    from flask import request
    main()