#!/bin/bash
# Quick launcher for SRISK Dashboard

echo "🚀 Launching SRISK Dashboard..."
echo "📊 Dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

uv run streamlit run scripts/dashboard_srisk.py
