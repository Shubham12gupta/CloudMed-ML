#!/bin/bash
# ============================================================
# shared/ec2_setup.sh
# One-time EC2 setup script for both ML services.
# Run once after pulling the repo on EC2.
# Usage: bash appointment_module/ml_backend/shared/ec2_setup.sh
# ============================================================

set -e

# ── Config (edit before running) ─────────────────────────────
S3_BUCKET="${MODELS_S3_BUCKET:-your-bucket-name}"
V2_S3_PREFIX="${MODELS_S3_PREFIX_V2:-ml-models/v2/latest/}"
V3_S3_PREFIX="${MODELS_S3_PREFIX_V3:-ml-models/v3/latest/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
V2_DIR="$ML_BACKEND_DIR/medical_history"
V3_DIR="$ML_BACKEND_DIR/consultation"

echo ""
echo "=================================================="
echo "  ML Backend EC2 Setup"
echo "  v2 dir: $V2_DIR"
echo "  v3 dir: $V3_DIR"
echo "=================================================="
echo ""

# ── ML v2: Medical History (port 5000) ───────────────────────
echo "[v2] Setting up medical_history service..."
cd "$V2_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
mkdir -p models
echo "[v2] Downloading models from S3: s3://$S3_BUCKET/$V2_S3_PREFIX"
aws s3 sync "s3://$S3_BUCKET/$V2_S3_PREFIX" models/
deactivate
echo "[v2] Done."

# ── ML v3: Consultation (port 5001) ──────────────────────────
echo ""
echo "[v3] Setting up consultation service..."
cd "$V3_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
mkdir -p models
echo "[v3] Downloading models from S3: s3://$S3_BUCKET/$V3_S3_PREFIX"
aws s3 sync "s3://$S3_BUCKET/$V3_S3_PREFIX" models/
deactivate
echo "[v3] Done."

echo ""
echo "=================================================="
echo "  Setup complete. Now create systemd services:"
echo "  bash shared/create_systemd_services.sh"
echo "=================================================="
