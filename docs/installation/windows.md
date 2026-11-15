# Windows Installation

This page installs ModelForge v4 on Windows 10 or 11 with an NVIDIA GPU. You can run it natively, which works for single-GPU training, or inside WSL2, which behaves like Linux and supports everything ModelForge does.

## Choosing native Windows or WSL2

| | Native Windows | WSL2 (Ubuntu) |
|---|---|---|
| Single-GPU training (SFT, DPO, GRPO, summarization, QA) | Yes | Yes |
| QLoRA 8-bit and 4-bit (bitsandbytes) | Yes, bitsandbytes ships a Windows wheel | Yes |
| Liger kernels (`[gpu]` extra) | No. Liger depends on Triton, which has no Windows wheels | Yes |
| Multi-GPU (data parallel, FSDP) | Not supported, see below | Yes |
| PyTorch install | Must install a CUDA build from pytorch.org | Default PyPI wheel includes CUDA |

!!! note "Multi-GPU on native Windows"
    Multi-GPU runs are started with Accelerate's `notebook_launcher`, which forks worker processes and uses NCCL for communication. Native Windows has neither `fork` nor NCCL in PyTorch's builds. If you have more than one GPU, use WSL2, or train on one GPU: pick it in the terminal wizard, or pass `gpu_indices` to the REST API. The browser console always uses every detected GPU.

The ModelForge CI runs on Linux only, so native Windows is less tested than Linux or WSL2.

## Option 1: Native Windows

### Prerequisites

- Windows 10 or 11, 64-bit.
- An NVIDIA GPU with a current driver from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx). You don't need the separate CUDA Toolkit; PyTorch wheels bundle the CUDA runtime.
- **Python 3.13** (64-bit) from [python.org](https://www.python.org/downloads/windows/). Tick **Add python.exe to PATH** in the installer. ModelForge v4 does not run on Python 3.11 or 3.12.

Check both:

```powershell
py -3.13 --version
nvidia-smi
```

### 1. Create a virtual environment

```powershell
mkdir C:\modelforge
cd C:\modelforge
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell refuses to run the activation script, allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Install PyTorch with CUDA first

PyTorch is not a direct dependency of ModelForge; it comes in through `accelerate`, `peft` and `bitsandbytes`. On Windows, the PyTorch wheel on PyPI is CPU-only. If you install ModelForge first, you get that CPU build and training silently runs on the CPU.

Install a CUDA build from the PyTorch index before ModelForge. Pick the command for your CUDA version from the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) (Stable, Windows, Pip, Python, a CUDA version). It looks like this:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Replace `cu128` with the CUDA tag the selector shows. Your driver must support that CUDA version; `nvidia-smi` prints the highest CUDA version the driver supports in its header.

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The last value must be `True`.

### 3. Install ModelForge

```powershell
pip install modelforge-finetuning
# terminal wizard and TensorBoard helper
pip install "modelforge-finetuning[cli]"
```

Don't install the `[gpu]` extra on native Windows. `liger-kernel` itself installs, but it needs Triton, which has no Windows wheels, so the install fails. ModelForge works without it and only enables Liger when the package is importable.

Because a CUDA build of `torch` is already installed and satisfies the requirement, pip keeps it. If `torch.cuda.is_available()` turns `False` after this step, pip replaced it; repeat step 2 with `--force-reinstall`.

### 4. Start ModelForge

```powershell
modelforge run
```

The browser opens the printed link with the startup token. Then continue with [Post-Installation Setup](post-installation.md).

!!! warning "Check what the planner sees"
    ModelForge detects GPUs through NVML (the NVIDIA driver), not through PyTorch. With a CPU-only PyTorch build the console still shows your GPU and plans for its VRAM, but training runs on the CPU. Always confirm `torch.cuda.is_available()` is `True`.

## Option 2: WSL2

WSL2 runs a real Linux kernel with access to your NVIDIA GPU through the Windows driver. Everything on the [Linux](linux.md) page applies inside it.

### Prerequisites

- Windows 10 21H2 or newer, or Windows 11.
- The current NVIDIA Windows driver. Install it on Windows only. Don't install an NVIDIA Linux driver inside WSL; the Windows driver provides GPU access.

### 1. Install WSL2 and Ubuntu

In an administrator PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart when asked, open **Ubuntu** from the Start menu, and create a Linux user. Check the GPU from inside Ubuntu:

```bash
nvidia-smi
```

### 2. Install Python 3.13

Ubuntu 24.04 ships Python 3.12, so install 3.13 separately. Two common ways:

=== "uv"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    uv python install 3.13
    uv venv --python 3.13 ~/modelforge/.venv
    ```

=== "deadsnakes PPA"

    ```bash
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.13 python3.13-venv
    python3.13 -m venv ~/modelforge/.venv
    ```

### 3. Install ModelForge

```bash
source ~/modelforge/.venv/bin/activate
pip install "modelforge-finetuning[cli,gpu]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

(With a uv-created environment, use `uv pip install` instead of `pip install`; uv environments don't include pip.)

On Linux the default PyTorch wheel from PyPI includes CUDA, so no extra index is needed. If `torch.cuda.is_available()` is `False`, see [Linux](linux.md#pytorch-and-cuda).

### 4. Start ModelForge

```bash
modelforge run
```

WSL2 forwards `localhost` to Windows, so the printed `http://127.0.0.1:8000/app?token=…` link opens in a Windows browser. If the browser doesn't open by itself, copy the link.

!!! tip "Keep data inside the Linux file system"
    Datasets, output directories and the Hugging Face cache are much faster under `~` than under `/mnt/c/`. Paths you type into the console are resolved inside WSL.

## Troubleshooting

### `torch.cuda.is_available()` is `False`

You have the CPU build, or a CUDA build newer than your driver supports. Reinstall from the PyTorch index with a CUDA tag your driver supports:

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### `liger-kernel` or `triton` fails to install

You installed the `[gpu]` extra on native Windows. Install without it: `pip install modelforge-finetuning`.

### `modelforge` is not recognized

The virtual environment is not active. Run `.venv\Scripts\Activate.ps1` again, or call `.venv\Scripts\modelforge.exe run`.

### Long path errors during install

Enable long paths once in an administrator PowerShell, then reinstall:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

More in [Windows-Specific Issues](../troubleshooting/windows-issues.md).

## Next steps

- [Post-Installation Setup](post-installation.md)
- [Quick Start](../getting-started/quickstart.md)
- [Hardware and Memory](../configuration/hardware-profiles.md)
