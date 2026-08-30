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
| GPU transport | Thunderbolt 3 eGPU enclosure, 40 Gb/s negotiated link |
| Runtime | `/home/boris/pytorch` |
| Service manager | systemd |
| Default model | TR-HASH MoE 200M Full SFT v1, 32K |
| Runtime dtype | FP32, matching the released checkpoint metadata |
| Quantization | none |

`CUDA_DEVICE_ORDER=PCI_BUS_ID` keeps CUDA ordinals aligned with `nvidia-smi`,
so device `1` reliably selects the RTX 5060 Ti rather than the display GPU.
CUDA Graphs start disabled on this host until an eager-generation smoke test
passes; they can then be enabled deliberately in `server.env`.

## Why systemd

Fedora already starts and supervises services with systemd. Running the model
as `boris` avoids a root-owned inference process, while the root healthcheck can
restart the complete service after three consecutive `/ready` failures. The
installer places its small management runtime under `/usr/local/lib` so SELinux
never needs to execute Python from a home directory.
The unit waits for the configured GPU through `nvidia-smi` without importing
PyTorch in a disposable process. This matters for the Thunderbolt eGPU: a
PyTorch preflight would create and immediately destroy a CUDA context just
before the inference server creates its long-lived context. The selected GPU
must remain continuously visible for 15 seconds before launch.

Restarts are deliberately slow and bounded. A stop includes a 10-second eGPU
cooldown, automatic restarts wait 20 seconds, and systemd permits only two
attempts per five minutes. The readiness watchdog refuses to restart the
service when NVML can no longer reach the GPU, avoiding a destructive restart
loop after an NVIDIA Xid 79 / PCIe link loss.

## Install on the server

```bash
sudo dnf install -y git python3
git clone https://github.com/Complexity-ML/TR-Hash-Server.git /home/boris/TR-Hash-Server
cd /home/boris/TR-Hash-Server
python3 -m venv .venv
.venv/bin/pip install .
sudo .venv/bin/tr-hash-server install --project "$PWD"
```

The installer creates `/etc/tr-hash-server/api.key` as `boris:boris` with mode
`0600`, which is the strict secret-file contract enforced by TR-Hash-i64. It
never places the secret itself in the process command line. Review the host
configuration with `sudoedit /etc/tr-hash-server/server.env`.

The readiness probe uses the same LAN address as the narrowly bound inference
listener, rather than broadening the service to every network interface.

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
sudo systemctl enable --now tr-hash-tensorboard.service
```

Optional private Hub credentials for supervised jobs belong in
`/etc/tr-hash-server/jobs.env` (root-owned, mode `0640`), never in a job TOML:

```bash
HF_TOKEN=hf_...
```

The first `doctor` call is expected to report readiness as unavailable until the
model service has started. Run it again after `/ready` becomes available.

## Daily commands

```bash
sudo systemctl status tr-hash-i64.service
sudo journalctl -u tr-hash-i64.service -f
sudo systemctl restart tr-hash-i64.service
sudo systemctl status tr-hash-tensorboard.service
sudo /usr/local/bin/tr-hash-server doctor
systemctl list-timers tr-hash-healthcheck.timer
```

TensorBoard listens only on the server loopback interface. From the Mac:

```bash
ssh -N -L 6006:127.0.0.1:6006 boris@192.168.1.16
```

Then open `http://localhost:6006`. Training jobs and TensorBoard are independent:
restarting the dashboard never interrupts a run.

Managed jobs may declare an absolute `tensorboard_logdir` in their TOML. On
submission, JobManager registers that directory under
`/var/lib/tr-hash-server/tensorboard/<job-name>`. TensorBoard watches this stable
registry, so runs from different framework checkouts appear automatically and
no project-specific artifact root is hard-coded in `server.env`.

From the Mac, verify the authenticated endpoint with the key copied through a
secure channel:

```bash
curl http://192.168.1.16:7860/ready
```

Do not forward port `7860` on the Internet. LAN exposure still requires the API
key, and remote public access should terminate TLS through a reverse proxy or a
VPN such as WireGuard/Tailscale.

## Generic eGPU jobs

`TR-Hash-Server` can supervise any long-running GPU command. The manager has no
model-size, dataset-size, token-budget, or training-framework assumptions. Copy
`examples/training-job.toml`, then set an argv array, working directory, GPU
ordinal and optional checkpoint contract.

```bash
sudo tr-hash-server job submit training.toml --enable-on-boot
tr-hash-server job list
tr-hash-server job status example-training
tr-hash-server job logs example-training -f
sudo tr-hash-server job stop example-training
sudo tr-hash-server job resume example-training
sudo tr-hash-server job remove example-training
```

Commands are executed directly without a shell. Environment variables must be
declared in the config. If matching checkpoints exist, `{checkpoint}` in each
`resume_arguments` entry expands to the newest artifact. External checkpoints
are never deleted by `job remove`.

Each job receives its own systemd unit and a lightweight parent supervisor. It
probes the selected eGPU through NVML without opening a second CUDA context. Two
consecutive probe failures terminate the child, persist `recovery_required`,
return exit code 79 and suppress immediate restart. An optionally enabled unit
starts again after host reboot and resumes from the newest checkpoint. A job
that already reached `completed` is not run again automatically.

For an eGPU that is unstable at its stock board-power limit, set
`egpu.power_limit_w` in the job TOML. The generated systemd unit reapplies the
limit with root privileges after the GPU stability check, including after a
reboot or driver reload, and fails closed if the limit cannot be applied.
