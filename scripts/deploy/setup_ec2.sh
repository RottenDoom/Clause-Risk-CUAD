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
mkdir -p "$APP_DIR/output/json" "$APP_DIR/output/html" "$APP_DIR/output/logs"

# ── 7. Systemd service ───────────────────────────────────────────────────────
# IMPORTANT:
#   --workers 1 — each worker would load its own 4-min embedder AND have its
#                 own in-memory _jobs dict (SSE stream might land on a
#                 different worker than the job).
#   TimeoutStartSec=600 — startup blocks on the synchronous embedder prewarm
#                         which takes 3-4 minutes on cold caches; systemd's
#                         default 90s timeout would SIGTERM us mid-load.
echo "==> Installing systemd service..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Contract Review FastAPI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/venv/bin/uvicorn api.routes:app --host 127.0.0.1 --port ${APP_PORT} --workers 1
Restart=always
RestartSec=5
TimeoutStartSec=600
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

# ── 8. Nginx config ──────────────────────────────────────────────────────────
# SSE-aware configuration:
#   proxy_buffering off            — required, otherwise nginx holds the SSE
#                                    stream until the buffer fills and the
#                                    frontend never sees individual cards
#   proxy_http_version 1.1 + ""    — proper keep-alive for long-lived streams
#   proxy_read_timeout 600s        — matches the SSE idle timeout in routes.py
echo "==> Writing nginx config..."
sudo tee /etc/nginx/sites-available/${SERVICE_NAME} > /dev/null <<EOF
server {
    listen 80;
    server_name _;          # replace _ with your domain name

    client_max_body_size 10M;

    # Main app
    location / {
        proxy_pass         http://127.0.0.1:${APP_PORT};
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_http_version 1.1;
        proxy_set_header   Connection        "";
    }

    # SSE streaming endpoint — disable buffering so events flush instantly
    location ~ ^/review/.+/stream$ {
        proxy_pass         http://127.0.0.1:${APP_PORT};
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_http_version 1.1;
        proxy_set_header   Connection        "";
        proxy_buffering    off;
        proxy_cache        off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        chunked_transfer_encoding off;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/${SERVICE_NAME} /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# ── 9. Pre-download + warm the embedder ──────────────────────────────────────
# Pulls the sentence-transformers model into ~/.cache/huggingface and warms
# the OS page cache. Without this, the FIRST systemd start blocks ~4 minutes
# inside the prewarm step.
echo "==> Pre-downloading + warming the embedding model (one-time, may take a few minutes)..."
cd "$APP_DIR"
source venv/bin/activate
python3 -c "from agent.embedder import embed; embed(['warmup sentence']); print('embedder ready')" || \
    echo "WARNING: embedder warmup failed — first service start will be slow but should still work."

echo ""
echo "==> Setup complete."
echo ""
echo "Next steps:"
echo "  1. Edit .env:       nano ${APP_DIR}/.env"
echo "  2. Populate data:   cd ${APP_DIR} && source venv/bin/activate"
echo "                      python3 scripts/prepare_data.py    # 80/20 split"
echo "                      python3 scripts/build_index.py     # upserts to Pinecone"
echo "  3. Start service:   sudo systemctl start ${SERVICE_NAME}"
echo "                      # Watch logs and wait for 'STARTUP: complete' before testing."
echo "  4. Check logs:      journalctl -u ${SERVICE_NAME} -f"
echo "  5. HTTPS (optional):sudo apt install certbot python3-certbot-nginx -y"
echo "                      sudo certbot --nginx -d yourdomain.com"
