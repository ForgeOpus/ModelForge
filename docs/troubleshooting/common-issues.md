# Common Issues and Solutions

Troubleshooting guide for frequent ModelForge issues.

## Installation Issues

### CUDA Not Available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

### Python Version Issues

**Symptom**: Installation fails with "requires Python 3.11"

**Solution**: Install Python 3.11:
```bash
# Linux
sudo apt install python3.11

# Or use pyenv
pyenv install 3.11.0
```

## Training Issues

### CUDA Out of Memory

**Symptom**: Training crashes with OOM error

**Solutions** (in order of preference):
1. Use QLoRA strategy:
   ```json
   {"strategy": "qlora", "use_4bit": true}
   ```
2. Reduce batch size:
   ```json
   {"per_device_train_batch_size": 1}
   ```
3. Reduce sequence length:
   ```json
   {"max_seq_length": 1024}
   ```
4. Enable gradient checkpointing:
   ```json
   {"gradient_checkpointing": true}
   ```
5. Use smaller model

### Training Very Slow

**Solutions**:
1. Use Unsloth provider (2x faster)
2. Use bf16 on Ampere+ GPUs:
   ```json
   {"bf16": true}
   ```
3. Increase batch size if VRAM allows
4. Use NVMe SSD for dataset

### Model Not Found

**Symptom**: "Model X not found on HuggingFace Hub"

**Solutions**:
1. Check model ID is correct
2. Set HuggingFace token:
   ```bash
   export HUGGINGFACE_TOKEN=your_token
   ```
3. For gated models, accept license on HuggingFace

## Windows-Specific Issues

### Unsloth Not Working

**Symptom**: "Unsloth is not installed" on Windows

**Solution**: Unsloth requires Linux. Use WSL or Docker.

See [Windows Installation](../installation/windows.md) for details.

### WSL GPU Not Detected

**Solutions**:
1. Update to latest Windows version
2. Update NVIDIA drivers (525.60+)
3. Ensure WSL 2: `wsl --status`
4. Restart WSL: `wsl --shutdown`

## Dataset Issues

### Dataset Validation Failed

**Symptom**: "Missing required field 'output'"

**Solution**: Ensure all examples have required fields:
```jsonl
{"input": "text", "output": "text"}
```

### Invalid JSON

**Symptom**: "Invalid JSON on line X"

**Solution**: Validate JSON:
```bash
python -m json.tool dataset.jsonl
```

## Provider Issues

### Provider Not Found

**Symptom**: "Unknown provider 'unsloth'"

**Solution**: Install provider:
```bash
pip install unsloth
```

### max_seq_length Error with Unsloth

**Symptom**: "max_seq_length cannot be -1"

**Solution**: Set fixed value:
```json
{"max_seq_length": 2048}
```

## API Issues

### Port Already in Use

**Symptom**: "Address already in use: 8000"

**Solutions**:
1. Find process: `lsof -i :8000`
2. Kill process or use different port:
   ```bash
   modelforge --port 8080
   ```

### Connection Refused

**Solutions**:
1. Check ModelForge is running: `ps aux | grep modelforge`
2. Check firewall settings
3. Try localhost: `http://localhost:8000`

## Performance Issues

### High Memory Usage

**Solutions**:
1. Use gradient checkpointing
2. Use 4-bit quantization
3. Reduce batch size
4. Close other applications

### Slow Inference

**Solutions**:
1. Use smaller model
2. Reduce max_seq_length
3. Use quantization
4. Batch requests

## Strategy-Task Incompatibility

### DPO/RLHF with Non-Text-Generation Task

**Symptom**: Validation error when starting training with DPO or RLHF strategy

**Solution**: DPO and RLHF strategies only support `"task": "text-generation"`. Change your task:
```json
{"strategy": "dpo", "task": "text-generation"}
```

### Unsloth with Non-Text-Generation Task

**Symptom**: Validation error when using Unsloth provider with summarization or QA

**Solution**: Unsloth only supports `"task": "text-generation"`. Use `"provider": "huggingface"` for other tasks.

## Missing Optional Dependencies

### bitsandbytes Not Installed

**Symptom**: `ImportError: bitsandbytes is required for quantization`

**Solution**: Install the quantization extra:
```bash
pip install modelforge-finetuning[quantization]
```

### CLI Wizard Not Starting

**Symptom**: `ImportError` for `questionary` or `rich` when running `modelforge cli`

**Solution**: Install the CLI extra:
```bash
pip install modelforge-finetuning[cli]
```

## More Help

- [Windows-Specific Issues](windows-issues.md)
- [FAQ](faq.md)
- [GitHub Issues](https://github.com/forgeopus/modelforge/issues)
- [GitHub Discussions](https://github.com/forgeopus/modelforge/discussions)

---

**Still having issues?** Create an issue on [GitHub](https://github.com/forgeopus/modelforge/issues/new).
