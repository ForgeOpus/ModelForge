# Post-Installation Setup

Complete these steps after installing ModelForge to ensure everything is configured correctly.

## Verify Installation

### 1. Check ModelForge Version

```bash
modelforge --version
```

Should display: `ModelForge v3` (or current version)

### 2. Verify GPU Detection

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output (with GPU):
```
CUDA Available: True
GPU Count: 1
GPU Name: NVIDIA GeForce RTX 3060
```

### 3. Test Basic Import

```bash
python -c "from ModelForge import app; print('ModelForge imported successfully!')"
```

### 4. Verify CLI Wizard (Optional)

If you installed the `[cli]` extra:

```bash
pip install modelforge-finetuning[cli]
modelforge cli
```

This should launch the interactive CLI wizard.

### 5. Verify Quantization (Optional)

If you installed the `[quantization]` extra:

```bash
pip install modelforge-finetuning[quantization]
python -c "import bitsandbytes; print('bitsandbytes installed successfully!')"
```

## Configure HuggingFace Token

Your HuggingFace token is required to download models.

### Get Your Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "ModelForge")
4. Select type: **Fine-grained** or **Write**
5. Copy the token

### Set Token

**Linux (persistent):**
```bash
echo 'export HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

**Windows PowerShell (persistent):**
```powershell
[System.Environment]::SetEnvironmentVariable('HUGGINGFACE_TOKEN', 'hf_xxxxxxxxxxxx', 'User')
```

**Using .env file (all platforms):**
```bash
cd ~/ModelForge  # or your project directory
echo "HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx" > .env
```

## First Run

### Start ModelForge

```bash
modelforge
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Access Web Interface

Open browser: `http://localhost:8000`

You should see the ModelForge interface with:
- Dashboard
- Training tab
- Playground tab
- Models tab

## Verify Features

### Check Available Providers

Open browser console and make API call:
```javascript
fetch('http://localhost:8000/api/info')
  .then(r => r.json())
  .then(d => console.log(d))
```

Should show:
```json
{
  "providers": ["huggingface", "unsloth"],
  "strategies": ["sft", "qlora", "rlhf", "dpo"],
  "tasks": ["text-generation", "summarization", "extractive-question-answering"]
}
```

### Test Health Endpoint

```bash
curl http://localhost:8000/api/health
```

Should return:
```json
{
  "status": "healthy",
  "version": "v2"
}
```

## Directory Structure

ModelForge creates the following directories:

**Linux:**
```
~/.local/share/modelforge/
├── database/              # SQLite database
├── datasets/              # Uploaded datasets
├── model_checkpoints/     # Trained models
└── training_logs/         # TensorBoard logs
```

**Windows:**
```
C:\Users\<username>\AppData\Local\modelforge\
├── database\
├── datasets\
├── model_checkpoints\
└── training_logs\
```

## Optional: Install Additional Providers

### Install Unsloth (2x Faster Training)

**Linux/WSL:**
```bash
pip install unsloth
```

**Verify:**
```bash
python -c "import unsloth; print('Unsloth installed successfully!')"
```

> **Windows Users**: Unsloth requires WSL or Docker. See [Windows Installation Guide](windows.md).

## Configure Advanced Settings

### Custom Port

Run on different port:
```bash
modelforge --port 8080
```

### Bind to All Interfaces

Allow remote access:
```bash
modelforge --host 0.0.0.0
```

### Configure Database Path

Set custom database location:
```bash
export MODELFORGE_DB_PATH="/path/to/database"
modelforge
```

## Performance Tuning

### GPU Memory Optimization

Add to `.env`:
```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Disable TensorBoard (Save Memory)

```bash
export MODELFORGE_DISABLE_TENSORBOARD=1
```

### Set Default Batch Size

```bash
export MODELFORGE_DEFAULT_BATCH_SIZE=4
```

## Verify Sample Dataset

### Download Test Dataset

```bash
cd ~/ModelForge
curl -o test_dataset.jsonl https://raw.githubusercontent.com/forgeopus/modelforge/main/ModelForge/test_datasets/low_text_generation.jsonl
```

### Upload via UI

1. Go to Training tab
2. Click "Upload Dataset"
3. Select `test_dataset.jsonl`
4. Should see validation success

### Or Upload via API

```bash
curl -X POST http://localhost:8000/api/upload_dataset \
  -F "file=@test_dataset.jsonl"
```

## Common Post-Installation Tasks

### Update ModelForge

```bash
pip install --upgrade modelforge-finetuning
```

### Reset Database

```bash
rm -rf ~/.local/share/modelforge/database
modelforge  # Will recreate database
```

### Clear Cache

```bash
rm -rf ~/.cache/huggingface
```

### View Logs

```bash
# Real-time logs
modelforge --log-level debug

# Or check specific log file
cat ~/.local/share/modelforge/training_logs/latest.log
```

## Security Considerations

### API Key Protection

Never commit `.env` files with tokens to version control:

```bash
echo ".env" >> .gitignore
```

### Firewall Configuration

If running on server, configure firewall:

**Linux (UFW):**
```bash
sudo ufw allow 8000/tcp
```

**Expose only to localhost:**
```bash
modelforge --host 127.0.0.1
```

### HTTPS Setup (Production)

Use reverse proxy (nginx/Apache) with SSL:

```nginx
server {
    listen 443 ssl;
    server_name modelforge.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Next Steps

- **[Quick Start Guide](../getting-started/quickstart.md)** - Run your first training
- **[Configuration Guide](../configuration/configuration-guide.md)** - Learn all options
- **[Dataset Formats](../configuration/dataset-formats.md)** - Prepare your data
- **[Troubleshooting](../troubleshooting/common-issues.md)** - Common issues

---

**Installation Complete!** You're ready to start fine-tuning models. 🚀
