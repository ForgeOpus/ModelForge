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
   modelforge run --port 8080
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

## More Help

- [Windows-Specific Issues](windows-issues.md)
- [FAQ](faq.md)
- [GitHub Issues](https://github.com/ForgeOpus/ModelForge/issues)
- [GitHub Discussions](https://github.com/ForgeOpus/ModelForge/discussions)

---

## Apple Silicon (MPS) Issues

### MPS Not Available

**Symptom**: `torch.backends.mps.is_available()` returns `False`

**Solutions**:
1. Verify you're on macOS 12.3 or later
2. Verify you have Apple Silicon (M1/M2/M3/M4)
3. Update PyTorch to latest version:
   ```bash
   pip install --upgrade torch
   ```
4. Check MPS build:
   ```python
   import torch
   print(f"MPS built: {torch.backends.mps.is_built()}")
   ```

### "Unsloth provider is not supported on Apple MPS"

**Symptom**: Error when trying to use Unsloth on macOS

**Solution**: Unsloth requires NVIDIA CUDA and is not compatible with Apple MPS. Use HuggingFace provider instead:
```json
{
  "provider": "huggingface",
  "device": "mps"
}
```

### "4-bit quantization via bitsandbytes is not supported on MPS"

**Symptom**: Error or warning about quantization on MPS

**Solution**: bitsandbytes library doesn't support MPS. Disable quantization:
```json
{
  "use_4bit": false,
  "use_8bit": false,
  "fp16": true
}
```

**Note**: ModelForge automatically disables quantization on MPS, but you may still see this warning.

### MPS Backend Out of Memory

**Symptom**: "MPS backend out of memory" error during training

**Solutions** (in order of preference):
1. Use a smaller model (3B instead of 7B)
2. Reduce `max_seq_length`:
   ```json
   {"max_seq_length": 512}
   ```
3. Reduce batch size:
   ```json
   {"per_device_train_batch_size": 1}
   ```
4. Enable gradient checkpointing:
   ```json
   {"gradient_checkpointing": true}
   ```
5. Close other applications to free unified memory

### MPS Training Very Slow

**Symptom**: Training takes much longer than expected

**Expected Behavior**: MPS is 3-5x slower than high-end NVIDIA GPUs, but still much faster than CPU.

**Tips to improve speed**:
- Use smaller models (1-3B parameters)
- Reduce `max_seq_length` to 512 or 1024
- Disable gradient checkpointing if you have enough memory:
  ```json
  {"gradient_checkpointing": false}
  ```
- Close other applications to free resources

### "RuntimeError: MPS does not support..."

**Symptom**: Operation not supported on MPS backend

**Cause**: Some PyTorch operations are not yet implemented for MPS.

**Solutions**:
1. Update to the latest PyTorch version:
   ```bash
   pip install --upgrade torch
   ```
2. Try a different model architecture
3. Fall back to CPU or use an NVIDIA GPU if available

**Note**: Report unsupported operations to PyTorch via their GitHub issues.

### Model Loading Fails on MPS

**Symptom**: Model fails to load or crashes on MPS

**Solutions**:
1. Ensure you're using HuggingFace provider (not Unsloth)
2. Disable quantization:
   ```json
   {"use_4bit": false, "use_8bit": false}
   ```
3. Try loading a different model (some architectures have better MPS support)
4. Check you have enough unified memory for the model

---

**Still having issues?** Create an issue on [GitHub](https://github.com/ForgeOpus/ModelForge/issues/new).
