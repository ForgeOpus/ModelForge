# Windows Installation Guide

Complete setup guide for ModelForge on Windows, including native installation and options for full feature support.

## 🚨 Important: Platform Limitations

**The Unsloth provider is NOT supported on native Windows.** Unsloth requires a Linux environment.

If you want to use Unsloth for 2x faster training, you have two options:
1. **[Windows Subsystem for Linux (WSL)](#option-2-wsl-installation-recommended)** - Recommended
2. **[Docker with NVIDIA Container Toolkit](#option-3-docker-installation)** - Alternative

For standard training with the HuggingFace provider, native Windows works perfectly.

---

## Option 1: Native Windows Installation

Use this if you're okay with the HuggingFace provider only (no Unsloth support).

### Prerequisites

1. **Windows 10/11** (64-bit)
2. **Python 3.11.x** - [Download from python.org](https://www.python.org/downloads/)
   - ⚠️ Python 3.12 is NOT supported yet
   - During installation, check "Add Python to PATH"
3. **NVIDIA GPU** with 4GB+ VRAM (6GB+ recommended)
4. **NVIDIA Drivers** - Latest Game Ready or Studio drivers
5. **CUDA Toolkit** - [Download CUDA 11.8 or 12.x](https://developer.nvidia.com/cuda-downloads)
6. **HuggingFace Account** - [Create account](https://huggingface.co/) and [generate access token](https://huggingface.co/settings/tokens)

### Verify Python Installation

Open **PowerShell** or **Command Prompt**:

```powershell
python --version
```

Should show: `Python 3.11.x`

### Verify CUDA Installation

```powershell
nvcc --version
```

Should show your CUDA version (e.g., `release 12.6`)

### Installation Steps

#### 1. Create a Virtual Environment (Recommended)

```powershell
# Create project directory
mkdir ModelForge
cd ModelForge

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate
```

#### 2. Install ModelForge

```powershell
pip install modelforge-finetuning

# Optional extras
pip install modelforge-finetuning[cli]           # CLI wizard
pip install modelforge-finetuning[quantization]   # 4-bit/8-bit quantization
```

#### 3. Install PyTorch with CUDA Support

Visit [PyTorch Installation Page](https://pytorch.org/get-started/locally/) and select:
- **PyTorch Build**: Stable
- **Your OS**: Windows
- **Package**: Pip
- **Language**: Python
- **Compute Platform**: CUDA 11.8 or CUDA 12.6

**For CUDA 12.6:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**For CUDA 11.8:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Verify GPU Detection

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Should show:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060
```

#### 5. Set HuggingFace Token

**Option A: Environment Variable (PowerShell)**
```powershell
$env:HUGGINGFACE_TOKEN="your_token_here"
```

**Option B: Environment Variable (CMD)**
```cmd
set HUGGINGFACE_TOKEN=your_token_here
```

**Option C: .env File (Persistent)**
```powershell
echo HUGGINGFACE_TOKEN=your_token_here > .env
```

#### 6. Run ModelForge

```powershell
modelforge          # Launch web UI
modelforge cli      # Launch CLI wizard (headless/SSH alternative)
```

Open browser to: `http://localhost:8000`

### ⚠️ Native Windows Limitations

- **No Unsloth provider** - Only HuggingFace provider available
- **Standard training speed** - Cannot use 2x speedup from Unsloth
- **All other features work** - SFT, QLoRA, RLHF, DPO strategies are available

---

## Option 2: WSL Installation (Recommended)

Windows Subsystem for Linux provides a full Linux environment on Windows, enabling **all ModelForge features** including Unsloth.

### Prerequisites

- Windows 10 (Build 19041+) or Windows 11
- NVIDIA GPU with latest drivers (525.60+)
- At least 16GB RAM recommended

### Step 1: Install WSL

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-22.04
```

This installs WSL 2 with Ubuntu 22.04. Restart your computer when prompted.

### Step 2: Set Up WSL

After restart, Ubuntu will open automatically:

1. Create a username and password
2. Update packages:

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 3: Install NVIDIA CUDA on WSL

**IMPORTANT**: Do NOT install NVIDIA drivers in WSL - use your Windows drivers!

Install CUDA Toolkit in WSL:

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install -y cuda-toolkit-12-6
```

Add CUDA to PATH:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
nvcc --version
nvidia-smi
```

### Step 4: Install Python 3.11 in WSL

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

Make Python 3.11 default:
```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

Verify:
```bash
python3 --version  # Should show Python 3.11.x
```

### Step 5: Install ModelForge in WSL

```bash
# Create virtual environment
mkdir ~/ModelForge
cd ~/ModelForge
python3 -m venv venv
source venv/bin/activate

# Install ModelForge
pip install modelforge-finetuning

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install Unsloth
pip install unsloth
```

### Step 6: Set HuggingFace Token

```bash
export HUGGINGFACE_TOKEN="your_token_here"

# Or add to .env file
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### Step 7: Run ModelForge

```bash
modelforge
```

Access from Windows browser: `http://localhost:8000`

### ✅ WSL Benefits

- **Full Unsloth support** - 2x faster training
- **All features available** - No limitations
- **Better performance** - Native Linux environment
- **Easy file access** - Access WSL files from Windows Explorer at `\\wsl$\Ubuntu-22.04\`

---

## Option 3: Docker Installation

Use Docker for isolated environments and easy deployment.

### Prerequisites

1. **Docker Desktop for Windows** - [Download](https://www.docker.com/products/docker-desktop/)
2. **NVIDIA GPU** with latest drivers
3. **WSL 2** backend (enabled by default in Docker Desktop)

### Step 1: Install Docker Desktop

1. Download and install Docker Desktop
2. Enable WSL 2 backend in Settings
3. Restart computer

### Step 2: Install NVIDIA Container Toolkit

Open **PowerShell as Administrator**:

```powershell
# Install WSL if not already installed
wsl --install

# Switch to WSL Ubuntu
wsl -d Ubuntu-22.04
```

In WSL Ubuntu terminal:

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
```

Restart Docker Desktop.

### Step 3: Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Should show your GPU information.

### Step 4: Create Dockerfile

Create `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install ModelForge
RUN pip install --no-cache-dir modelforge-finetuning

# Install PyTorch with CUDA
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install Unsloth
RUN pip install --no-cache-dir unsloth

# Set working directory
WORKDIR /workspace

# Expose port
EXPOSE 8000

# Run ModelForge
CMD ["modelforge", "--host", "0.0.0.0"]
```

### Step 5: Build and Run

```powershell
# Build image
docker build -t modelforge:latest .

# Run container
docker run --gpus all -p 8000:8000 -e HUGGINGFACE_TOKEN=your_token_here modelforge:latest
```

Access at: `http://localhost:8000`

### Docker with Persistent Storage

To preserve data between container restarts:

```powershell
docker run --gpus all -p 8000:8000 \
  -v modelforge-data:/root/.local/share/modelforge \
  -e HUGGINGFACE_TOKEN=your_token_here \
  modelforge:latest
```

---

## Operational Constraints

### Important: max_sequence_length in Unsloth

When using the **Unsloth provider**, you **MUST** specify a fixed `max_sequence_length`. Auto-inference (value `-1`) is **not supported**.

**Example Configuration:**

```json
{
  "provider": "unsloth",
  "model_name": "meta-llama/Llama-3.2-3B",
  "max_seq_length": 2048,  // REQUIRED: Must be a positive integer
  "strategy": "sft",
  ...
}
```

**Valid values:**
- `512`, `1024`, `2048`, `4096`, `8192`, etc.

**Invalid values:**
- `-1` (auto-inference - NOT supported)
- `0` or negative numbers

This limitation is specific to Unsloth and does not apply to the HuggingFace provider.

---

## Troubleshooting

### CUDA Not Detected

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Check CUDA installation: `nvcc --version`
4. Restart computer after installing CUDA

### Unsloth Import Error on Native Windows

**Problem**: `ImportError: No module named 'unsloth'`

**Solution**: Unsloth requires Linux. Use WSL or Docker (see above).

### WSL GPU Not Accessible

**Problem**: `nvidia-smi` works in Windows but not in WSL

**Solutions**:
1. Update Windows to latest version
2. Update NVIDIA drivers (525.60+)
3. Ensure WSL 2 is installed: `wsl --status`
4. Restart WSL: `wsl --shutdown` then reopen

### Docker GPU Not Accessible

**Problem**: Docker can't access GPU

**Solutions**:
1. Ensure WSL 2 backend is enabled in Docker Desktop
2. Install nvidia-container-toolkit in WSL (see above)
3. Restart Docker Desktop
4. Verify: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`

### Permission Denied Errors

**Problem**: Cannot write to directories

**Solutions**:
- **WSL**: Ensure you own the directory: `sudo chown -R $USER:$USER ~/ModelForge`
- **Docker**: Use volume mounts with correct permissions

---

## Next Steps

- **[Post-Installation Setup](post-installation.md)** - Configure ModelForge
- **[Quick Start Guide](../getting-started/quickstart.md)** - Run your first training job
- **[Configuration Guide](../configuration/configuration-guide.md)** - Learn all options
- **[Unsloth Provider](../providers/unsloth.md)** - Learn about Unsloth features

---

**Need Help?** Check the [Windows-Specific Troubleshooting](../troubleshooting/windows-issues.md) guide.
