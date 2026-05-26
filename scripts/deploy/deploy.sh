#!/usr/bin/env bash
# scripts/deploy/deploy.sh
#
# Re-deploy: pull latest code and restart the service.
# Run this for every subsequent deploy after setup_ec2.sh has been run once.
#
# Usage (on the EC2 instance):
#   ./scripts/deploy/deploy.sh
#
# Usage (from your local machine):
#   ssh -i key.pem ubuntu@YOUR_IP "cd contract-review && ./scripts/deploy/deploy.sh"

set -euo pipefail

APP_DIR="/home/ubuntu/contract-review"
SERVICE_NAME="contract-review"

cd "$APP_DIR"

echo "==> Pulling latest code..."
git pull origin main

echo "==> Updating Python dependencies..."
source venv/bin/activate
uv pip install -r requirements.txt

echo "==> Restarting service..."
sudo systemctl restart "$SERVICE_NAME"

echo "==> Waiting for service to come up..."
sleep 3
sudo systemctl is-active --quiet "$SERVICE_NAME" \
    && echo "Service is running." \
    || { echo "ERROR: Service failed to start. Check: journalctl -u ${SERVICE_NAME} -n 50"; exit 1; }

echo "==> Deploy complete."
