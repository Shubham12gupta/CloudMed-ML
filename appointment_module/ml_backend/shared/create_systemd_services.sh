#!/bin/bash
# ============================================================
# shared/create_systemd_services.sh
# Creates systemd service files for ML v2 and ML v3.
# Run once on EC2 as root/sudo.
# Usage: sudo bash appointment_module/ml_backend/shared/create_systemd_services.sh
# ============================================================

set -e

# ── Detect paths ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
V2_DIR="$ML_BACKEND_DIR/medical_history"
V3_DIR="$ML_BACKEND_DIR/consultation"
EC2_USER="${SUDO_USER:-ubuntu}"

echo "Using EC2 user: $EC2_USER"
echo "ML v2 path: $V2_DIR"
echo "ML v3 path: $V3_DIR"

# ── Create ml-v2.service ─────────────────────────────────────
cat > /etc/systemd/system/ml-v2.service << EOF
[Unit]
Description=Medical ML v2 — Medical History Suggestions (port 5000)
After=network.target

[Service]
User=$EC2_USER
WorkingDirectory=$V2_DIR
ExecStart=$V2_DIR/venv/bin/python $V2_DIR/app.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo "[v2] systemd service created: /etc/systemd/system/ml-v2.service"

# ── Create ml-v3.service ─────────────────────────────────────
cat > /etc/systemd/system/ml-v3.service << EOF
[Unit]
Description=Medical ML v3 — Consultation Suggestions (port 5001)
After=network.target

[Service]
User=$EC2_USER
WorkingDirectory=$V3_DIR
ExecStart=$V3_DIR/venv/bin/python $V3_DIR/app.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo "[v3] systemd service created: /etc/systemd/system/ml-v3.service"

# ── Enable and start ──────────────────────────────────────────
systemctl daemon-reload
systemctl enable ml-v2
systemctl enable ml-v3
systemctl start ml-v2
systemctl start ml-v3

echo ""
echo "=================================================="
echo "  Services started. Check status:"
echo "  sudo systemctl status ml-v2"
echo "  sudo systemctl status ml-v3"
echo ""
echo "  Health checks:"
echo "  curl http://localhost:5000/health"
echo "  curl http://localhost:5001/health"
echo "=================================================="
