#!/usr/bin/env bash
# scripts/deploy/setup_ec2.sh
#
# One-time setup for a fresh Ubuntu 22.04 EC2 instance.
# Run as the ubuntu user (sudo access assumed).
#
# Usage:
#   chmod +x scripts/deploy/setup_ec2.sh
#   ./scripts/deploy/setup_ec2.sh
#
# What it does:
#   1. Updates system packages
#   2. Installs Python 3.11, nginx, git, curl
#   3. Installs uv (fast Python package manager)
#   4. Clones the repo and creates a venv
#   5. Installs Python deps (including torch — slow first time)
#   6. Writes the systemd service and nginx config
#   7. Enables HTTPS instructions (certbot)
#
# After this runs, you still need to:
#   a. Add .env with your API keys
#   b. Run build_index.py (or rsync pre-built chroma_db/ from local)
#   c. Point nginx server_name to your domain
#   d. Run certbot if you have a domain

set -euo pipefail

REPO_URL="${REPO_URL:-}"          # set before running, or fill in below
APP_DIR="/home/ubuntu/contract-review"
SERVICE_NAME="contract-review"
APP_PORT=8000

# ── 1. System packages ────────────────────────────────────────────────────────
echo "==> Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-venv python3-pip nginx git curl

# ── 2. uv (fast package manager used locally) ────────────────────────────────
echo "==> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
if [ -z "$REPO_URL" ]; then
    echo "ERROR: Set REPO_URL before running this script."
    echo "  export REPO_URL=https://github.com/yourname/yourrepo.git"
    exit 1
fi

echo "==> Cloning $REPO_URL → $APP_DIR"
git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"

# ── 4. Python venv + deps ────────────────────────────────────────────────────
echo "==> Creating venv and installing dependencies (torch will take a few minutes)..."
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt

# ── 5. Env file placeholder ──────────────────────────────────────────────────
if [ ! -f "$APP_DIR/.env" ]; then
    cat > "$APP_DIR/.env" <<'EOF'
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME
PINECONE_API_KEY=pc-REPLACE_ME
PINECONE_INDEX_NAME=cuad-contracts
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EOF
    echo "==> .env placeholder written. Edit it with your real keys before starting the service."
fi

# ── 6. Required output directories ───────────────────────────────────────────
mkdir -p "$APP_DIR/output/json" "$APP_DIR/output/html"

# ── 7. Systemd service ───────────────────────────────────────────────────────
echo "==> Installing systemd service..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Contract Review FastAPI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/venv/bin/uvicorn api.routes:app --host 127.0.0.1 --port ${APP_PORT} --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

# ── 8. Nginx config ──────────────────────────────────────────────────────────
echo "==> Writing nginx config..."
sudo tee /etc/nginx/sites-available/${SERVICE_NAME} > /dev/null <<EOF
server {
    listen 80;
    server_name _;          # replace _ with your domain name

    client_max_body_size 10M;

    location / {
        proxy_pass         http://127.0.0.1:${APP_PORT};
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;             # review jobs can be slow
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/${SERVICE_NAME} /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo ""
echo "==> Setup complete."
echo ""
echo "Next steps:"
echo "  1. Edit .env:       nano ${APP_DIR}/.env"
echo "  2. Populate index:  cd ${APP_DIR} && source venv/bin/activate"
echo "                      python3 scripts/prepare_data.py   # if data not pre-built"
echo "                      python3 scripts/build_index.py --pinecone"
echo "                      python3 scripts/build_index.py --chromadb   # optional"
echo "  3. Start service:   sudo systemctl start ${SERVICE_NAME}"
echo "  4. Check logs:      journalctl -u ${SERVICE_NAME} -f"
echo "  5. HTTPS (optional):sudo apt install certbot python3-certbot-nginx -y"
echo "                      sudo certbot --nginx -d yourdomain.com"
