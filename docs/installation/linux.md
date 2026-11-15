# Linux Installation

This page installs ModelForge v4 on a Linux machine with one or more NVIDIA GPUs. Linux is the primary platform: it is what CI runs on, and it is the only native platform where multi-GPU training, FSDP and Liger kernels are available.

## Prerequisites

- A 64-bit Linux distribution (x86_64 or aarch64).
- An NVIDIA GPU and the proprietary NVIDIA driver. ModelForge reads GPUs through NVML, which ships with the driver.
- **Python 3.13** or newer. v4 does not run on Python 3.11 or 3.12.
- Disk space for base models (Hugging Face cache), checkpoints and outputs.

CPU-only machines also work, but the planner caps models at 500M parameters (less on machines with little RAM) and training is very slow. See [Hardware and Memory](../configuration/hardware-profiles.md).

## 1. Install the NVIDIA driver

Check whether a driver is already loaded:

```bash
nvidia-smi
```

If the command is missing or fails, install the driver your distribution recommends:

=== "Ubuntu / Debian"

    ```bash
    sudo ubuntu-drivers install
    sudo reboot
    ```

=== "Fedora"

    ```bash
    # Enable RPM Fusion first: https://rpmfusion.org/Configuration
    sudo dnf install akmod-nvidia
    sudo reboot
    ```

=== "Arch"

    ```bash
    sudo pacman -S nvidia nvidia-utils
    sudo reboot
    ```

You don't need the CUDA Toolkit. PyTorch wheels bundle the CUDA runtime libraries; only the driver is required.

## 2. Install Python 3.13

Many distributions still ship an older Python. Check with `python3 --version`. If it is below 3.13:

=== "uv (any distribution)"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.13
    uv venv --python 3.13 ~/modelforge/.venv
    ```

=== "Ubuntu (deadsnakes PPA)"

    ```bash
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.13 python3.13-venv
    python3.13 -m venv ~/modelforge/.venv
    ```

=== "Fedora"

    ```bash
    sudo dnf install python3.13
    python3.13 -m venv ~/modelforge/.venv
    ```

uv-created environments don't include pip. Use `uv pip install …` wherever this page says `pip install …`, or run `python -m ensurepip` inside the environment.

## 3. Install ModelForge

```bash
source ~/modelforge/.venv/bin/activate
pip install --upgrade pip
pip install "modelforge-finetuning[cli,gpu]"
```

| Extra | Adds | Notes |
|---|---|---|
| none | The browser console, planner and all trainers | Includes bitsandbytes for QLoRA. |
| `[cli]` | click, rich | Needed for `modelforge cli` and `modelforge tensorboard`. |
| `[gpu]` | liger-kernel (and Triton) | Fused kernels that cut memory on CUDA. Optional; ModelForge falls back when it is absent. |

## PyTorch and CUDA

PyTorch is not a direct dependency. It is installed as a dependency of `accelerate`, `peft` and `bitsandbytes`. On Linux the default `torch` wheel from PyPI is built with CUDA and pulls in the NVIDIA runtime libraries as pip packages, so the install above is usually enough.

Verify:

```
$ python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
2.11.0 13.0 True
```

Your exact versions will differ. If the last value is `False`:

- **The driver is too old for the wheel's CUDA version.** The `nvidia-smi` header shows the highest CUDA version your driver supports. Either update the driver, or install a wheel built for an older CUDA version from the PyTorch index. Pick the command from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/); it looks like:

    ```bash
    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
    ```

- **You are on a machine without a GPU** (or in a container without GPU access). See below.

!!! warning "The planner trusts NVML, not PyTorch"
    GPU detection uses NVML from the driver. If NVML sees a GPU but PyTorch can't use it, the console plans for the GPU's VRAM but training falls back to the CPU. Fix `torch.cuda.is_available()` before you train.

## 4. Start ModelForge

```bash
modelforge run
```

On a desktop, the browser opens the printed link. On a remote server, see the next section. Then continue with [Post-Installation Setup](post-installation.md).

## Remote servers

`modelforge run` binds `0.0.0.0` by default, so it is reachable from the network by anyone who has the token. On a remote GPU box, bind to localhost and tunnel instead:

```bash
# on the server
modelforge run --host 127.0.0.1 --no-browser

# on your laptop
ssh -L 8000:127.0.0.1:8000 user@gpu-server
```

Open the printed `http://127.0.0.1:8000/app?token=…` link on your laptop. For a fully headless workflow, use the terminal wizard (`modelforge cli`) instead. See [Terminal Wizard](../user-guide/terminal-cli.md).

## Running as a systemd service

To keep the console running across logouts, run it as a user service with a fixed token:

```ini
# ~/.config/systemd/user/modelforge.service
[Unit]
Description=ModelForge browser console

[Service]
WorkingDirectory=%h/modelforge
ExecStart=%h/modelforge/.venv/bin/modelforge run --host 127.0.0.1 --no-browser --token REPLACE_WITH_A_LONG_RANDOM_TOKEN
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now modelforge
journalctl --user -u modelforge -f
loginctl enable-linger "$USER"   # keep running after logout
```

`WorkingDirectory` matters: relative dataset paths and the default `./modelforge_output` are resolved from it. Generate a token with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Anyone who can read the unit file can read the token.

If the service restarts during a run, that run is marked failed on startup and can be resumed from its last checkpoint. See [Checkpoints and Resuming](../user-guide/checkpoints.md).

## Multiple GPUs

ModelForge sums the VRAM of all GPUs NVML reports. The browser console trains on every detected GPU; the terminal wizard asks which ones to use, and the REST API accepts `gpu_indices`. ModelForge sets `CUDA_VISIBLE_DEVICES` itself from that selection, so setting it in your shell before `modelforge run` does not restrict the browser console. See [Multi-GPU](../strategies/multi-gpu.md).

## Installing from source

A source checkout builds the browser UI during install, so it needs **Node.js and npm** on `PATH` in addition to Python 3.13. The repository pins Node in `.node-version` (22.x).

```bash
git clone https://github.com/ForgeOpus/ModelForge.git
cd ModelForge
pip install -e ".[dev]"      # runs npm ci && npm run build via setup.py
modelforge run --no-browser
```

If Node.js or npm is missing, or the frontend build fails, the install stops with an error instead of producing a package without a UI. Installing a published wheel from PyPI does not need Node. See [Development Setup](../contributing/development.md).

## Troubleshooting

### `nvidia-smi` works but no GPU appears in Settings

NVML could not be initialized from the ModelForge process, which then reports no GPUs and plans for CPU. Check that the server runs as a user with access to `/dev/nvidia*`, and in containers that the GPU is passed through.

### Port 8000 already in use

```bash
modelforge run --port 8080
```

### `ImportError` for click or rich

`modelforge cli` and `modelforge tensorboard` need the `[cli]` extra: `pip install "modelforge-finetuning[cli]"`.

### Out of memory during training

See [Performance and Memory](../troubleshooting/performance.md).

## Next steps

- [Post-Installation Setup](post-installation.md)
- [Quick Start](../getting-started/quickstart.md)
- [Memory Optimizations](../strategies/memory-optimizations.md)
