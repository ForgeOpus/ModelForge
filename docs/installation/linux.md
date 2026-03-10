# Linux Installation Guide

Complete setup guide for ModelForge on Linux with full feature support.

## Prerequisites

- **Linux Distribution**: Ubuntu 20.04+, Debian 11+, Fedora 35+, or similar
- **Python 3.11.x** (Python 3.12 not yet supported)
- **NVIDIA GPU** with 4GB+ VRAM (6GB+ recommended)
- **NVIDIA Drivers**: Version 525.60 or newer
- **CUDA Toolkit**: 11.8 or 12.x
- **HuggingFace Account**: [Create account](https://huggingface.co/) and [generate access token](https://huggingface.co/settings/tokens)

## Installation Steps

### 1. Update System Packages

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt upgrade -y
```

**Fedora:**
```bash
sudo dnf update -y
```

**Arch Linux:**
```bash
sudo pacman -Syu
```

### 2. Install Python 3.11

**Ubuntu/Debian:**
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

**Fedora:**
```bash
sudo dnf install -y python3.11 python3.11-devel
```

**Arch Linux:**
```bash
sudo pacman -S python311
```

Verify installation:
```bash
python3.11 --version
```

### 3. Install NVIDIA Drivers

Check if drivers are already installed:
```bash
nvidia-smi
```

If not installed:

**Ubuntu/Debian:**
```bash
# Add graphics-drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Install latest driver (or specific version like nvidia-driver-535)
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot
```

**Fedora:**
```bash
# Enable RPM Fusion repositories
sudo dnf install -y https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install -y https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Install NVIDIA drivers
sudo dnf install -y akmod-nvidia

# Reboot
sudo reboot
```

After reboot, verify:
```bash
nvidia-smi
```

### 4. Install CUDA Toolkit

**Ubuntu/Debian (CUDA 12.6):**
```bash
# Download and install CUDA repository package
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install -y cuda-toolkit-12-6
```

**Fedora (CUDA 12.6):**
```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
sudo dnf clean all
sudo dnf install -y cuda-toolkit-12-6
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
```

### 5. Create Virtual Environment

```bash
# Create project directory
mkdir ~/ModelForge
cd ~/ModelForge

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 6. Install ModelForge

```bash
pip install modelforge-finetuning

# Optional extras
pip install modelforge-finetuning[cli]           # CLI wizard
pip install modelforge-finetuning[quantization]   # 4-bit/8-bit quantization
```

### 7. Install PyTorch with CUDA Support

Visit [PyTorch Installation Page](https://pytorch.org/get-started/locally/) for the latest command.

**For CUDA 12.6:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 8. Install Unsloth (Optional, for 2x Faster Training)

```bash
pip install unsloth
```

### 9. Verify GPU Detection

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060
```

### 10. Set HuggingFace Token

**Option A: Export (temporary):**
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

**Option B: .env file (persistent):**
```bash
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

**Option C: .bashrc (permanent):**
```bash
echo 'export HUGGINGFACE_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### 11. Run ModelForge

```bash
modelforge          # Launch web UI
modelforge cli      # Launch CLI wizard (headless/SSH alternative)
```

Open browser to: `http://localhost:8000`

## Running as a Service (Optional)

To run ModelForge as a systemd service:

### 1. Create Service File

```bash
sudo nano /etc/systemd/system/modelforge.service
```

Add:
```ini
[Unit]
Description=ModelForge Fine-Tuning Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/ModelForge
Environment="PATH=/home/your_username/ModelForge/venv/bin"
Environment="HUGGINGFACE_TOKEN=your_token_here"
ExecStart=/home/your_username/ModelForge/venv/bin/modelforge
Restart=always

[Install]
WantedBy=multi-user.target
```

Replace `your_username` and `your_token_here` with your values.

### 2. Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable modelforge
sudo systemctl start modelforge
```

### 3. Check Status

```bash
sudo systemctl status modelforge
```

### 4. View Logs

```bash
sudo journalctl -u modelforge -f
```

## Troubleshooting

### CUDA Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Verify NVIDIA drivers:
   ```bash
   nvidia-smi
   ```
2. Check CUDA installation:
   ```bash
   nvcc --version
   ```
3. Reinstall PyTorch with correct CUDA version
4. Check LD_LIBRARY_PATH includes CUDA libs:
   ```bash
   echo $LD_LIBRARY_PATH
   ```

### Driver Version Mismatch

**Problem**: `CUDA driver version is insufficient`

**Solution**:
Update NVIDIA drivers to version 525.60 or newer.

### Permission Denied

**Problem**: Cannot create files/directories

**Solution**:
```bash
sudo chown -R $USER:$USER ~/ModelForge
```

### Port Already in Use

**Problem**: `Address already in use: 8000`

**Solutions**:
1. Check what's using port 8000:
   ```bash
   sudo lsof -i :8000
   ```
2. Kill the process or use a different port:
   ```bash
   modelforge --port 8080
   ```

### Out of Memory (OOM)

**Problem**: Training crashes with OOM error

**Solutions**:
1. Use QLoRA strategy for memory efficiency
2. Reduce batch size
3. Use gradient checkpointing
4. Use a smaller model
5. See [Performance Optimization](../troubleshooting/performance.md)

## Docker Installation (Alternative)

### 1. Install Docker

**Ubuntu:**
```bash
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

Log out and back in for group changes to take effect.

### 2. Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Create Dockerfile and Build

Create a `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install CUDA (if needed for GPU support)
# Skip if using NVIDIA base image
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-6 \
    && rm cuda-keyring_1.1-1_all.deb

# Install ModelForge from PyPI
RUN pip install --no-cache-dir modelforge-finetuning

# Install PyTorch with CUDA
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install Unsloth (optional)
RUN pip install --no-cache-dir unsloth

# Set working directory
WORKDIR /workspace

# Expose port
EXPOSE 8000

# Run ModelForge
CMD ["modelforge", "--host", "0.0.0.0"]
```

Build and run:

```bash
# Build the image
docker build -t modelforge:latest .

# Run container
docker run --gpus all -p 8000:8000 \
  -e HUGGINGFACE_TOKEN=your_token_here \
  -v modelforge-data:/root/.local/share/modelforge \
  modelforge:latest
```

## Next Steps

- **[Post-Installation Setup](post-installation.md)** - Configure ModelForge
- **[Quick Start Guide](../getting-started/quickstart.md)** - Run your first training
- **[Configuration Guide](../configuration/configuration-guide.md)** - Learn all options
- **[Unsloth Provider](../providers/unsloth.md)** - Enable 2x faster training

---

**Need Help?** Check [Common Issues](../troubleshooting/common-issues.md) or ask in [GitHub Discussions](https://github.com/forgeopus/modelforge/discussions).
