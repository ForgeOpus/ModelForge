# Windows issues

This page covers problems specific to running ModelForge v4 on native Windows, and when to use WSL2 instead. Installation steps are in [Windows installation](../installation/windows.md); general errors are in [Common issues](common-issues.md).

ModelForge v4 has no Windows-specific code paths, and its CI runs on Linux only. Native Windows works for single-GPU training with the Hugging Face stack, with the caveats below. For multi-GPU training or Liger kernels, use WSL2.

## What works where

| Feature | Native Windows | WSL2 (Ubuntu) |
|---|---|---|
| Browser console, API, terminal wizard, notebooks | Yes | Yes |
| Single-GPU training (SFT, DPO, GRPO, QA, Summarization) | Yes, with a CUDA build of PyTorch | Yes |
| LoRA bf16 and full fine-tuning | Yes | Yes |
| QLoRA 8-bit / 4-bit (bitsandbytes) | Yes. bitsandbytes 0.49 ships a `win_amd64` wheel. | Yes |
| Liger kernels (`[gpu]` extra) | No. Triton has no Windows wheels. | Yes |
| Multi-GPU (data parallel, FSDP) | No. See [below](#multi-gpu-training-doesnt-work). | Yes |

There is no Unsloth provider in v4 and no Docker image, so neither is a workaround here.

## PyTorch installs without CUDA

**Symptom:** **Settings → Hardware** lists your NVIDIA GPU, but the plan shows `bf16: false` and `fp16: false`, and training is very slow because it runs on the CPU.

ModelForge doesn't pin PyTorch directly; it comes in as a dependency of Accelerate, PEFT and TRL. On Windows, the default `torch` wheel on PyPI is **CPU-only** (the CUDA libraries are only pulled in on Linux). GPU detection uses NVIDIA's NVML, which only needs the driver, so the GPU still shows up.

Check:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False`, install a CUDA build from the PyTorch index in the same environment, then restart `modelforge run` and generate the plan again. Get the exact command for your driver from the [PyTorch install selector](https://pytorch.org/get-started/locally/). It has this form:

```powershell
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/<cuda-tag>
```

Do this **after** installing ModelForge, or install PyTorch first and then ModelForge, so pip doesn't replace the CUDA build with the CPU one.

## Liger kernels won't install

```powershell
pip install "modelforge-finetuning[gpu]"
```

fails on Windows because `liger-kernel` requires `triton`, and Triton publishes Linux wheels only. Install ModelForge without the extra. Plans then set `use_liger_kernel: false` and training works normally, using somewhat more activation memory. If you want Liger, use WSL2.

## Multi-GPU training doesn't work

ModelForge launches multi-GPU runs with Accelerate's notebook launcher, which starts one worker per GPU by **forking** the server process. Windows has no `fork`, and PyTorch's NCCL backend is Linux-only. A machine with two or more GPUs will plan for all of them by default, and the run then fails to start.

On native Windows, train on one GPU: pass `gpu_indices` with a single index to both `POST /api/recommendations` and `POST /api/runs` (see [REST API](../api-reference/rest-api.md#post-apiruns)). The console always trains on every detected GPU, so on a multi-GPU Windows machine start runs through the API.

Setting `CUDA_VISIBLE_DEVICES` before starting the server does not help: GPU detection reads every card through NVML, and at launch ModelForge overwrites `CUDA_VISIBLE_DEVICES` with the GPUs it planned for. For real multi-GPU training, use WSL2 or Linux. See [Multi-GPU training](../strategies/multi-gpu.md).

## Firewall prompt when the server starts

The first time `modelforge run` starts, Windows Defender Firewall may ask whether to allow Python on private or public networks. That's because the server binds `0.0.0.0` (every interface) by default.

- If you only use the console on this PC, choose **Cancel** / don't allow, or avoid the prompt by binding to loopback:

    ```powershell
    modelforge run --host 127.0.0.1
    ```

- Allow private networks only if other machines on your LAN need to reach it. Anyone who has the token can then start runs and read file paths.

## Long paths

Hugging Face cache entries, checkpoint folders and TensorBoard logs create deep paths, for example `...\hub\models--org--model-name\snapshots\<40-char hash>\...` and `<output dir>\checkpoint-1500\...`. With Windows' default 260-character limit, downloads or saves can fail with `FileNotFoundError` or `OSError: [Errno 2]` on files that clearly exist.

Fixes:

- Enable long paths (administrator PowerShell, then sign out and in):

    ```powershell
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
      -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ```

- Keep output directories short, for example `C:\mf\runs\qwen-01` rather than a deeply nested project folder.
- Move the Hugging Face cache to a short path with `HF_HOME`, for example `$env:HF_HOME = "D:\hf"`.

## Hugging Face cache uses more disk than expected

`huggingface_hub` stores downloads as symlinks into a blob store. Without Developer Mode or administrator rights, Windows can't create symlinks, so the library warns once and stores full copies instead. Everything works, but repeated revisions of a model take more space. Turn on **Settings → System → For developers → Developer Mode** to let it use symlinks.

## Paths in API requests

The API is JSON, so backslashes in Windows paths must be escaped, or use forward slashes, which Windows accepts:

```json
{"path": "C:\\data\\train.jsonl"}
{"path": "C:/data/train.jsonl"}
```

Relative paths resolve against the folder `modelforge run` was started in, not the folder your browser or script is in.

## Disk space shown for the wrong drive

ModelForge measures disk space at the filesystem root (`/`), which on Windows means the root of the **current drive**. If you start the server from `C:` but write outputs to `D:`, the plan's free-space figure and its disk warning describe `C:`. Start `modelforge run` from the drive that will hold your outputs.

## Other Windows notes

- **Python 3.13 or newer is required.** Use the python.org installer or the `py` launcher (`py -3.13 -m venv .venv`).
- **Port already in use** shows as `[WinError 10048] only one usage of each socket address ... is normally permitted`. Use `--port`.
- **Where data lives:** run history and the model registry are in `modelforge.db` under `%LOCALAPPDATA%\ForgeOpus\ModelForge`.
- **Playground disk offload** goes to `%TEMP%` by default. Set `MODELFORGE_OFFLOAD_DIR` to a folder on a larger or faster drive.
- **Hugging Face login:** use `hf auth login`. The old `huggingface-cli login` no longer works with the `huggingface-hub` version v4 requires.

## Using WSL2 instead

WSL2 runs a real Linux kernel with GPU passthrough, so ModelForge behaves exactly as on Linux: Liger installs, multi-GPU works, and the default PyPI PyTorch includes CUDA.

1. Install a current NVIDIA driver **on Windows**. Don't install an NVIDIA driver inside WSL; the Windows driver provides the GPU to WSL.
2. Install WSL2 and Ubuntu from an administrator PowerShell:

    ```powershell
    wsl --install -d Ubuntu
    ```

3. Inside Ubuntu, check the GPU is visible with `nvidia-smi`, then follow [Linux installation](../installation/linux.md).
4. Start the server inside WSL. WSL2 forwards `localhost`, so the printed `http://127.0.0.1:8000/app?token=...` link opens from a Windows browser.

Keep datasets and output directories inside the Linux filesystem (for example `~/runs`), not under `/mnt/c`. Reading and writing through `/mnt/c` is much slower.

## See also

- [Windows installation](../installation/windows.md)
- [Common issues](common-issues.md)
- [Performance and out-of-memory](performance.md)
- [Multi-GPU training](../strategies/multi-gpu.md)
