#!/bin/bash

# Kronos Web UI startup script

echo "🚀 Starting Kronos Web UI..."
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not installed, please install Python3 first"
    exit 1
fi

# Check if in correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this script in the webui directory"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
if ! python3 -c "import flask, flask_cors, pandas, numpy, plotly" &> /dev/null; then
    echo "⚠️  Missing dependencies, installing..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Dependencies installation failed"
        exit 1
    fi
    echo "✅ Dependencies installation completed"
else
    echo "✅ All dependencies installed"
fi

# Start application
echo "🌐 Starting Web server..."
echo "Access URL: http://${KRONOS_WEBUI_HOST:-127.0.0.1}:${KRONOS_WEBUI_PORT:-7070}"
echo "Press Ctrl+C to stop server"
echo ""

python3 app.py
