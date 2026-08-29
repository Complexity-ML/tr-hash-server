# TR-Hash-Server

Operations repository for the physical TR-HASH inference server at
`192.168.1.16`. It does not contain another inference engine: TR-Hash-i64 owns
model loading, batching, KV cache, CUDA Graphs and the OpenAI-compatible API.
This project owns the Fedora host lifecycle around that engine.

## Host profile

| Role | Device |
| --- | --- |
| Display | GPU 0 — NVIDIA GeForce GTX 1660 Ti, 6 GB |
| Inference | GPU 1 — NVIDIA GeForce RTX 5060 Ti, 16 GB |
| Runtime | `/home/boris/pytorch` |
| Service manager | systemd |
| Default model | TR-HASH MoE 200M Full SFT v1, 32K |
| Quantization | none |

`CUDA_DEVICE_ORDER=PCI_BUS_ID` keeps CUDA ordinals aligned with `nvidia-smi`,
so device `1` reliably selects the RTX 5060 Ti rather than the display GPU.

## Why systemd

Fedora already starts and supervises services with systemd. Running the model
as `boris` avoids a root-owned inference process, while the root healthcheck can
restart the complete service after three consecutive `/ready` failures.

## Install on the server

```bash
sudo dnf install -y git python3
git clone https://github.com/Complexity-ML/TR-Hash-Server.git /home/boris/TR-Hash-Server
cd /home/boris/TR-Hash-Server
python3 -m venv .venv
.venv/bin/pip install .
sudo .venv/bin/tr-hash-server install --project "$PWD"
```

Create the API key without exposing it in the process command line:

```bash
sudo install -o root -g boris -m 0640 /dev/null /etc/tr-hash-server/api.key
sudo sh -c 'openssl rand -hex 32 > /etc/tr-hash-server/api.key'
sudo chown root:boris /etc/tr-hash-server/api.key
sudo chmod 0640 /etc/tr-hash-server/api.key
sudoedit /etc/tr-hash-server/server.env
```

Install or update TR-Hash-i64 inside the existing runtime:

```bash
source /home/boris/pytorch/bin/activate
python -m pip install -U 'git+https://github.com/Complexity-ML/TR-Hash-i64.git@main'
```

Validate before enabling the long-running process:

```bash
sudo -u boris --preserve-env=TR_HASH_EXECUTABLE,TR_HASH_DEVICES \
  /usr/local/bin/tr-hash-server doctor
sudo systemctl enable --now tr-hash-i64.service
sudo systemctl enable --now tr-hash-healthcheck.timer
```

The first `doctor` call is expected to report readiness as unavailable until the
model service has started. Run it again after `/ready` becomes available.

## Daily commands

```bash
sudo systemctl status tr-hash-i64.service
sudo journalctl -u tr-hash-i64.service -f
sudo systemctl restart tr-hash-i64.service
sudo /usr/local/bin/tr-hash-server doctor
systemctl list-timers tr-hash-healthcheck.timer
```

From the Mac, verify the authenticated endpoint with the key copied through a
secure channel:

```bash
curl http://192.168.1.16:7860/ready
```

Do not forward port `7860` on the Internet. LAN exposure still requires the API
key, and remote public access should terminate TLS through a reverse proxy or a
VPN such as WireGuard/Tailscale.
