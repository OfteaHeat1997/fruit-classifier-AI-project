#!/bin/bash
# Start Jupyter for PyCharm with correct network settings
# This allows Windows PyCharm to connect to WSL Jupyter

# Activate the venv
source /home/paula/.virtualenvs/fruit-classifier-AI-project/bin/activate

# Get WSL IP address
WSL_IP=$(hostname -I | awk '{print $1}')

echo "============================================"
echo "STARTING JUPYTER FOR PYCHARM"
echo "============================================"
echo ""
echo "Jupyter will be accessible from Windows at:"
echo "  http://${WSL_IP}:8888"
echo ""
echo "PyCharm will connect automatically"
echo "Press Ctrl+C to stop"
echo ""

# Start Jupyter with config that allows Windows connections
cd /mnt/c/Users/maria/Desktop/fruit-classifier-AI-project
jupyter lab --config=jupyter_config.py --port=8888
