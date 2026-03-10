# Windows-Specific Troubleshooting

Solutions for common Windows-specific issues with ModelForge.

## Unsloth Provider Issues

### Unsloth Not Supported on Native Windows

**Problem**: "Unsloth is not installed" or import errors on native Windows

**Cause**: Unsloth requires Linux-specific libraries and compilation

**Solutions**:

1. **Use WSL 2 (Recommended)**: [WSL Installation Guide](../installation/windows.md#option-2-wsl-installation-recommended)
2. **Use Docker**: [Docker Installation Guide](../installation/windows.md#option-3-docker-installation)
3. **Use HuggingFace Provider**: Works perfectly on native Windows

### CLI Wizard on Native Windows

The CLI wizard (`modelforge cli`) works on native Windows — no WSL needed. Only the Unsloth provider requires WSL or Docker.

```powershell
pip install modelforge-finetuning[cli]
modelforge cli
```

## WSL Issues

### WSL GPU Not Accessible

**Problem**: `nvidia-smi` works in Windows but not in WSL

**Solutions**:
1. Update Windows to latest version (Build 21H2 or later)
2. Update NVIDIA drivers to 525.60 or newer
3. Ensure WSL 2 is installed:
   ```powershell
   wsl --status
   ```
4. Restart WSL:
   ```powershell
   wsl --shutdown
   ```
   Then reopen Ubuntu

### CUDA Not Found in WSL

**Problem**: `nvcc: command not found` in WSL

**Solution**: Install CUDA Toolkit in WSL (not Windows):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

## Docker Issues

### Docker GPU Not Accessible

**Problem**: Docker container can't access GPU

**Solutions**:
1. Ensure WSL 2 backend is enabled in Docker Desktop
2. Install nvidia-container-toolkit in WSL:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```
3. Restart Docker Desktop

### Docker Build Fails

**Problem**: Dockerfile build errors

**Solution**: Ensure WSL integration is enabled:
1. Docker Desktop → Settings → Resources → WSL Integration
2. Enable integration for Ubuntu distribution
3. Restart Docker Desktop

## Installation Issues

### Permission Denied Errors

**Problem**: Cannot create files/directories

**Solutions**:

**In WSL**:
```bash
sudo chown -R $USER:$USER ~/ModelForge
```

**In Windows**:
Run PowerShell as Administrator

### PATH Issues

**Problem**: `modelforge` command not found

**Solutions**:

**In PowerShell**:
```powershell
# Add to PATH
$env:PATH += ";$env:USERPROFILE\AppData\Local\Programs\Python\Python311\Scripts"
```

**In WSL**:
```bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## Environment Variables

### HuggingFace Token Not Recognized

**Problem**: Token not found despite setting

**Solutions**:

**PowerShell (Session)**:
```powershell
$env:HUGGINGFACE_TOKEN="your_token"
```

**PowerShell (Permanent)**:
```powershell
[System.Environment]::SetEnvironmentVariable('HUGGINGFACE_TOKEN', 'your_token', 'User')
```

**CMD**:
```cmd
setx HUGGINGFACE_TOKEN "your_token"
```

Restart terminal after setting permanent variables.

## Performance Issues

### Slow Training on Windows

**Solutions**:
1. Use WSL instead of native Windows (better I/O performance)
2. Install dataset on NVMe SSD
3. Disable Windows Defender for project folder
4. Close unnecessary Windows applications

### High CPU Usage

**Solutions**:
1. Limit Windows background processes
2. Disable Windows Search indexing for project folder
3. Use WSL for better resource management

## File System Issues

### Line Ending Issues

**Problem**: JSONL files have wrong line endings

**Solution**: Convert CRLF to LF:
```powershell
(Get-Content dataset.jsonl) | Set-Content dataset.jsonl
```

### Path Issues

**Problem**: Paths with backslashes don't work

**Solution**: Use forward slashes or raw strings:
```python
# Good
path = "C:/Users/username/dataset.jsonl"
path = r"C:\Users\username\dataset.jsonl"

# Bad
path = "C:\Users\username\dataset.jsonl"  # Escaping issues
```

### Long Path Issues

**Problem**: "File path too long" errors

**Solution**: Enable long paths in Windows:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Restart computer.

## Firewall and Antivirus

### Windows Defender Blocks ModelForge

**Solution**: Add exception:
1. Windows Security → Virus & threat protection → Manage settings
2. Add exclusion for ModelForge folder

### Firewall Blocks Port 8000

**Solution**: Add firewall rule:
```powershell
New-NetFirewallRule -DisplayName "ModelForge" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

## Specific Error Messages

### "vcruntime140.dll not found"

**Solution**: Install Visual C++ Redistributable:
Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### "DLL load failed"

**Solutions**:
1. Install Visual C++ Redistributable
2. Update Windows
3. Reinstall Python

### "Access is denied"

**Solutions**:
1. Run PowerShell as Administrator
2. Check folder permissions
3. Disable antivirus temporarily

## WSL-Specific

### WSL Runs Out of Memory

**Solution**: Configure WSL memory limit in `.wslconfig`:

Create `C:\Users\<username>\.wslconfig`:
```ini
[wsl2]
memory=16GB
processors=8
swap=8GB
```

Restart WSL:
```powershell
wsl --shutdown
```

### File Access Between Windows and WSL

**Access WSL from Windows**:
```
\\wsl$\Ubuntu-22.04\home\username\
```

**Access Windows from WSL**:
```
/mnt/c/Users/username/
```

## Getting More Help

- [General Troubleshooting](common-issues.md)
- [FAQ](faq.md)
- [Windows Installation Guide](../installation/windows.md)
- [GitHub Issues](https://github.com/forgeopus/modelforge/issues)

---

**Windows-specific issue not listed?** Ask in [GitHub Discussions](https://github.com/forgeopus/modelforge/discussions).
