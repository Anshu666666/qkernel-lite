# 🚀 QKernel-Lite Deployment Playbook: Oracle Cloud Ubuntu VM

This guide provides step-by-step instructions for deploying **QKernel-Lite** to your Oracle Cloud VM (`152.67.6.234`), conforming to the specifications in `VM_CENTRAL_MANAGEMENT.md`.

---

## 📌 Architecture & Resource Budgeting

| Specification | Allocated Setting |
|---|---|
| **Host IP** | `152.67.6.234` (Oracle Cloud Always-Free) |
| **SSH User** | `ubuntu` |
| **Internal Binding** | `127.0.0.1:8090` |
| **Public Domain** | `https://qkernel.duckdns.org` (or subpath on existing domain) |
| **Memory Ceiling** | **350 MiB limit** enforced via systemd `MemoryMax=350M` |
| **Static Serving** | Caddy serves `report/` and `slides/` directly with near-zero RAM usage |

---

## ⚡ Quick 5-Step Deployment Process

### Step 1: Connect to Your VM
From your local Windows PowerShell:
```powershell
ssh -i "c:\Users\anshu\Downloads\ssh-key-2026-09-09 (1).key" ubuntu@152.67.6.234
```

---

### Step 2: Clone the Repository to `/home/ubuntu/qkernel-lite`
```bash
cd /home/ubuntu
git clone https://github.com/Anshu666666/qkernel-lite.git
cd qkernel-lite
```

---

### Step 3: Set Up a Lightweight Production Virtual Environment
Because the Oracle Cloud VM has 1 GB RAM total, we install a lightweight runtime for the dashboard (Streamlit, Scikit-Learn, Plotly, Pandas, Matplotlib) without heavy GPU/CUDA overhead:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install streamlit pandas numpy scikit-learn matplotlib plotly
```

---

### Step 4: Install and Enable the `systemd` Service
Copy the included service unit file:
```bash
sudo cp deploy/qkernel.service /etc/systemd/system/qkernel.service
sudo systemctl daemon-reload
sudo systemctl enable qkernel.service
sudo systemctl start qkernel.service
```

Verify service status and memory consumption:
```bash
systemctl status qkernel.service
```
*(Expected memory usage: ~75–120 MiB, safely below the 350 MiB ceiling.)*

---

### Step 5: Configure Reverse Proxy & Automatic SSL in Caddy
1. Create or map a DuckDNS subdomain (e.g. `qkernel.duckdns.org` -> `152.67.6.234`).
2. Append the block from `deploy/Caddyfile.snippet` to `/etc/caddy/Caddyfile`:
```bash
sudo nano /etc/caddy/Caddyfile
```
Add:
```caddy
qkernel.duckdns.org {
    # Direct high-speed static asset delivery
    handle_path /report* {
        root * /home/ubuntu/qkernel-lite/report
        file_server
    }
    handle_path /slides* {
        root * /home/ubuntu/qkernel-lite/slides
        file_server
    }

    # Reverse proxy Streamlit app with WebSockets
    handle {
        reverse_proxy 127.0.0.1:8090
    }

    encode gzip zstd
}
```
3. Test and reload Caddy:
```bash
caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

---

## 🛠️ Routine Monitoring Commands

- **Inspect live logs:**
  ```bash
  sudo journalctl -u qkernel.service -f -n 50
  ```
- **Restart service:**
  ```bash
  sudo systemctl restart qkernel.service
  ```
- **Live Memory Check:**
  ```bash
  ps -eo pid,cmd,%mem,%cpu --sort=-%mem | head -n 10
  ```
