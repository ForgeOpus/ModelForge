# macOS Installation (Apple Silicon)

This page installs ModelForge v4 on an Apple Silicon Mac (M1 or later) and explains what the planner does with the MPS backend. Macs work for learning the workflow and for small models; for anything larger, train on an NVIDIA GPU.

## What to expect on a Mac

| | Apple Silicon (MPS) |
|---|---|
| Memory the planner assumes | **8 GB**, whatever your unified memory is |
| Tiers | `full_finetune` or `lora_bf16` only |
| QLoRA 8-bit and 4-bit | Disabled. bitsandbytes quantization isn't supported on MPS |
| Precision | Full precision. bf16 and fp16 are both off |
| Liger kernels (`[gpu]` extra) | Not available. Liger needs CUDA and Triton |
| Multi-GPU and FSDP | Not applicable |
| Intel Macs | Not supported. Current PyTorch releases publish macOS wheels for Apple Silicon only |

ModelForge can't read how much of the unified memory the GPU may use, so it counts MPS as a fixed 8 GB. That gives these ceilings, worked out with the formulas on [Hardware and Memory](../configuration/hardware-profiles.md):

| Objective | Adjusted memory | Largest full fine-tune | Largest LoRA |
|---|---|---|---|
| Supervised (SFT), summarization, extractive QA | 8 GB | ≈ 0.33B parameters | ≈ 2.95B parameters |
| Preference (DPO) | 3 GB | ≈ 0.06B parameters | ≈ 0.45B parameters |
| Reinforcement (GRPO) | 2 GB | none | none |

The planner picks the largest popular Hub model under the LoRA ceiling and uses full fine-tuning only if that model also fits the full fine-tune ceiling. GRPO reserves room for four sampled completions, which leaves nothing for a model on the 8 GB budget, so expect no recommendation for the Reinforcement objective on a Mac.

The plan shows a warning on MPS: *"Apple MPS detected. bitsandbytes 4-bit and 8-bit quantisation are not supported on MPS."*

## Prerequisites

- An Apple Silicon Mac.
- **macOS 14 (Sonoma) or newer.** bitsandbytes is a required dependency, and its macOS wheel (at the version ModelForge locks, 0.49.2) is built for macOS 14 and later on arm64.
- **Python 3.13** or newer.

## 1. Install Python 3.13

=== "Homebrew"

    ```bash
    brew install python@3.13
    python3.13 -m venv ~/modelforge/.venv
    ```

=== "uv"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.13
    uv venv --python 3.13 ~/modelforge/.venv
    ```

    uv environments don't include pip. Use `uv pip install` below.

=== "python.org"

    Download the macOS installer for Python 3.13 from [python.org](https://www.python.org/downloads/macos/), then:

    ```bash
    python3.13 -m venv ~/modelforge/.venv
    ```

## 2. Install ModelForge

```bash
source ~/modelforge/.venv/bin/activate
pip install --upgrade pip
pip install "modelforge-finetuning[cli]"
```

Don't add the `[gpu]` extra; Liger depends on Triton, which has no macOS wheels.

PyTorch comes in as a dependency. The PyPI wheel for macOS arm64 includes MPS support, so no special index is needed. Check it:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

The last value must be `True`. If it is `False`, you are probably running an x86_64 Python under Rosetta. Check with `python -c "import platform; print(platform.machine())"`; it must print `arm64`.

## 3. Start ModelForge

```bash
modelforge run
```

Open **Settings** in the console. Under **Hardware**, the GPU reads **Apple Silicon GPU · N/A GB**. That `N/A` is expected: it is the entry the planner turns into 8 GB. Continue with [Post-Installation Setup](post-installation.md).

## Choosing models

- Keep to models under about 3B parameters for SFT, and under about 0.5B for DPO. Models above the ceiling are refused with *"Selected model exceeds the available training memory budget."*
- Smaller is often better here. A 0.5B model fine-tuned on good data trains several times faster than a 3B one.
- Close memory-heavy apps before training. The 8 GB figure is a planning bound, not a reservation.

## Optional PyTorch settings

ModelForge v4 does not set any MPS environment variables. (v3 set `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically; v4 does not.) If a run fails with an error saying an operator is not implemented for MPS, you can let PyTorch fall back to the CPU for that operator:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
modelforge run
```

This is a PyTorch setting, not a ModelForge one. It makes the unsupported operator run on the CPU, which is slower.

## Troubleshooting

### `MPS backend out of memory`

The model and activations exceeded what macOS let the GPU allocate. Choose a smaller model, shorten your examples, or close other applications. See [Performance and Memory](../troubleshooting/performance.md).

### bitsandbytes fails to install

You are on macOS 13 or older, or on an Intel Mac. bitsandbytes is a required dependency and has no wheel for those systems. Upgrade to macOS 14 or newer on Apple Silicon.

### No recommendation for Reinforcement (GRPO)

Expected on MPS; see the table above. Use Supervised or Preference, or train GRPO on a CUDA machine.

### Training is very slow

MPS training runs in full precision, without fused kernels, and with gradient checkpointing on. That is the trade for fitting into memory. For real workloads, use a machine with an NVIDIA GPU.

## Next steps

- [Post-Installation Setup](post-installation.md)
- [Quick Start](../getting-started/quickstart.md)
- [Hardware and Memory](../configuration/hardware-profiles.md)
