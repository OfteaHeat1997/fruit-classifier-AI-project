#!/bin/bash
# Start TensorBoard to View Training Progress
# Student: Maria Paula Salazar Agudelo
# Course: AI Minor - Personal Challenge

# WHAT THIS DOES:
# Starts TensorBoard web interface to view training progress in real-time

# HOW TO RUN:
#   chmod +x scripts/start_tensorboard.sh
#   ./scripts/start_tensorboard.sh

# Or directly:
#   bash scripts/start_tensorboard.sh

echo "======================================================================"
echo "TENSORBOARD - TRAINING VISUALIZATION"
echo "======================================================================"
echo ""
echo "Starting TensorBoard server..."
echo ""
echo "TensorBoard will show:"
echo "  - Training & Validation Accuracy graphs"
echo "  - Training & Validation Loss graphs"
echo "  - Model architecture diagram"
echo "  - Weight distributions (histograms)"
echo ""
echo "After server starts, open your browser to:"
echo "  http://localhost:6006"
echo ""
echo "Press CTRL+C to stop TensorBoard"
echo ""
echo "======================================================================"
echo ""

# Start TensorBoard
tensorboard --logdir logs/fit --port 6006
