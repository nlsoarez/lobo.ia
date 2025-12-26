#!/usr/bin/env python3
"""
Entry point for Railway Web Service.
Runs the Streamlit dashboard.
"""

import os
import subprocess
import sys

def main():
    port = os.environ.get('PORT', '8501')

    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'dashboard.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--server.fileWatcherType', 'none'
    ]

    print(f"Starting Streamlit dashboard on port {port}...")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
