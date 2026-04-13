#!/usr/bin/env python3
"""
Kronos Web UI startup script
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import plotly
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def install_dependencies():
    """Install dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installation completed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Dependencies installation failed")
        return False

def main():
    """Main function"""
    print("🚀 Starting Kronos Web UI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nAuto-install dependencies? (y/n): ", end="")
        if input().lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("Please manually install dependencies and retry")
            return
    
    # Check model availability
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos model library available")
        model_available = True
    except ImportError:
        print("⚠️  Kronos model library not available, will use simulated prediction")
        model_available = False
    
    # Start Flask application
    print("\n🌐 Starting Web server...")
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    # Start server
    try:
        from app import app
        print("✅ Web server started successfully!")
        print(f"🌐 Access URL: http://localhost:7070")
        print("💡 Tip: Press Ctrl+C to stop server")
        
        # Auto-open browser
        time.sleep(2)
        webbrowser.open('http://localhost:7070')
        
        # Start Flask application
        app.run(debug=True, host='0.0.0.0', port=7070)
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        print("Please check if port 7070 is occupied")

if __name__ == "__main__":
    main()
