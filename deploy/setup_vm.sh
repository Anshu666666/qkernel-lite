#!/usr/bin/env bash
set -e

echo "=== 1. Verifying Environment & Packages ==="
cd /home/ubuntu/qkernel-lite

/home/ubuntu/qkernel-lite/venv/bin/python3 -c "
import streamlit
import torch
import pennylane
import medmnist
import sklearn
import pandas
import numpy
print('Python packages verified successfully!')
"

echo "=== 2. Creating Systemd Service (/etc/systemd/system/qkernel.service) ==="
sudo tee /etc/systemd/system/qkernel.service > /dev/null << 'EOF'
[Unit]
Description=QKernel-Lite Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/qkernel-lite
ExecStart=/home/ubuntu/qkernel-lite/venv/bin/streamlit run app.py --server.port 8502 --server.address 127.0.0.1 --server.headless true --browser.serverAddress 152.67.6.234 --server.enableCORS false --server.enableXsrfProtection false
Restart=always
RestartSec=5
MemoryMax=500M

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable qkernel.service
sudo systemctl restart qkernel.service
echo "qkernel.service restarted successfully."

echo "=== 3. Configuring Caddyfile for Port 8501 ==="
# Check if :8501 block already exists in /etc/caddy/Caddyfile
if ! grep -q ":8501" /etc/caddy/Caddyfile; then
    sudo tee -a /etc/caddy/Caddyfile > /dev/null << 'EOF'

:8501 {
    handle_path /report* {
        root * /home/ubuntu/qkernel-lite/report
        file_server
    }
    handle_path /slides* {
        root * /home/ubuntu/qkernel-lite/slides
        file_server
    }
    handle {
        reverse_proxy 127.0.0.1:8502
    }
}
EOF
    echo "Added :8501 block to /etc/caddy/Caddyfile"
fi

sudo systemctl reload caddy
echo "Caddy reloaded successfully."

echo "=== 4. Verifying Services ==="
sleep 5
sudo systemctl status qkernel.service --no-pager
curl -I http://127.0.0.1:8502 || true
curl -I http://127.0.0.1:8501 || true

echo "=== Deployment Completed Successfully! ==="
