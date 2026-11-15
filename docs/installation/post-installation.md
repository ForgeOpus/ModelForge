# Post-Installation Setup

This page covers what to do after the package is installed: check the install, log in to Hugging Face, start the server and connect to it safely, confirm GPU detection, and find where ModelForge keeps its files. It applies to every platform.

## Verify the install

```
$ modelforge --help
usage: modelforge [-h] {run,cli,recommend,history,list-tasks,tensorboard} ...
```

```bash
pip show modelforge-finetuning          # Version: 4.x
which modelforge-nb                     # the notebook entry point (where modelforge-nb on Windows)
```

Then check PyTorch, because ModelForge's GPU detection doesn't depend on it but training does:

=== "NVIDIA (CUDA)"

    ```bash
    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
    ```

    Expect `True` and your GPU count. If not, see [Windows](windows.md) or [Linux](linux.md#pytorch-and-cuda).

=== "Apple Silicon"

    ```bash
    python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
    ```

    Expect `True`. See [macOS](macos-mps.md).

If you plan to use the terminal wizard, check the optional extra:

```bash
python -c "import click, rich; print('cli extra ok')"
```

## Log in to Hugging Face

Public models and datasets need no login. You need one for:

- **Gated models** such as Llama or Gemma. Accept the licence on the model's Hub page first, with the same account.
- **Private datasets or models.**
- **Pushing a trained model** to the Hub with `POST /api/models/{id}/push`. That needs a token with write access.

Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and log in once on the machine that runs the server:

```bash
hf auth login
```

The server uses this cached login whenever it talks to the Hub. Alternatively, set `HF_TOKEN` in the environment that starts ModelForge; `huggingface_hub` reads it, and the terminal wizard checks it before offering to log in.

!!! note "`huggingface-cli` no longer works"
    ModelForge v4 depends on `huggingface_hub` 1.x, where the old `huggingface-cli` command only prints a deprecation message. Use `hf auth login`, `hf auth whoami` and `hf auth logout`.

The terminal wizard (`modelforge cli`) checks for a token at start and offers to log in if none is found. The browser console doesn't prompt; log in from a terminal.

## First run

```
$ modelforge run
ModelForge v4 browser UI
Serving on: http://0.0.0.0:8000
Open:       http://127.0.0.1:8000/app?token=…
The startup token is required for API and WebSocket access.
```

The browser opens the **Open** link. If it doesn't (for example over SSH), copy the link yourself, or pass `--no-browser` to skip trying. Relative paths you enter later, such as `./data/train.jsonl` or the default `./modelforge_output`, are resolved from the directory you ran this command in.

| Flag | Default | Effect |
|---|---|---|
| `--host` | `0.0.0.0` | Interface to bind. |
| `--port` | `8000` | Port to bind. |
| `--no-browser` | off | Don't open a browser. |
| `--token` | random | Use a fixed startup token. |

## The startup token

Each start generates a random token (`secrets.token_urlsafe(32)`), unless you pass `--token`. The printed link carries it as `?token=…`. The console saves the token in the browser, removes it from the address bar, and sends it with every request.

- **Every API endpoint except `GET /api/health` requires it.** A missing or wrong token returns `401`.
- **WebSockets** (live run events, Playground chat) take it as a `?token=` query parameter.
- **A new token is generated on every restart**, so old links and other browsers stop working. Use `--token` for bookmarks, scripts and services.

To connect another browser, open the printed link in it, or paste the token under **Settings → Connection → API token** and reload. If the console says the connection is unavailable, the token is missing or out of date.

Scripts send the token in any one of these ways:

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/hardware
curl -H "X-ModelForge-Token: $TOKEN"   http://127.0.0.1:8000/api/hardware
curl "http://127.0.0.1:8000/api/hardware?token=$TOKEN"
```

See [REST API](../api-reference/rest-api.md).

## Network exposure

!!! warning "The server listens on every interface by default"
    `--host` defaults to `0.0.0.0`. Anyone who can reach the port and has the token can start runs, read file paths and datasets on the machine, and push models with your Hugging Face login. The server speaks plain HTTP, so the token travels unencrypted on the network.

- On a laptop or a shared network, bind to localhost: `modelforge run --host 127.0.0.1`.
- On a remote GPU machine, bind to localhost and use an SSH tunnel: `ssh -L 8000:127.0.0.1:8000 user@host`, then open the printed link locally.
- If you must expose it, put it behind a reverse proxy that terminates TLS, and use a long `--token`.
- Treat the token like a password. It appears in the printed link and in shell history if you pass it on the command line.

## Verify GPU detection

ModelForge plans from the hardware it detects, so check it before your first run. Open **Settings** in the console. **Hardware** lists each GPU with its memory, total RAM and free disk; **Environment** shows the version and providers. If the hardware looks wrong, the plan will be wrong too.

The same data is available from the API:

```
$ curl -s -H "X-ModelForge-Token: $TOKEN" http://127.0.0.1:8000/api/hardware
{
  "physical_core_count": 6,
  "disk_storage_gb": 853.06,
  "available_disk_storage_gb": 510.88,
  "ram_gb": 60.92,
  "gpus": [{"name": "NVIDIA GeForce RTX 5060 Ti", "total_memory_gb": 15.93}]
}
```

How each platform shows up:

| Machine | `gpus` | What the planner does |
|---|---|---|
| NVIDIA | One entry per GPU, memory from NVML | Sums the VRAM of the GPUs in use |
| Apple Silicon | `{"name": "Apple Silicon GPU", "total_memory_gb": "N/A"}` | Counts it as 8 GB, disables QLoRA |
| CPU only | `[]` | Caps models at 500M parameters, trains an adapter on the CPU |

GPU detection uses NVML from the NVIDIA driver, not PyTorch. If `gpus` lists your card but `torch.cuda.is_available()` is `False`, the plan is sized for the GPU while training runs on the CPU. Free disk is measured on the root file system (`/`, or the current drive on Windows), not on your output directory's drive.

To see a full plan from the terminal without training:

```bash
modelforge recommend --use-case instruction_following --objective supervised
```

## Where things are stored

| What | Where |
|---|---|
| Run history, checkpoint list, run events, model registry | `modelforge.db` (SQLite) in the user data directory, below |
| Trained models and checkpoints | The output directory you choose per run (default `./modelforge_output`); checkpoints in `<output dir>/checkpoint-<step>` |
| Training metadata | `<output dir>/modelforge_config.json` (browser and notebook runs) |
| TensorBoard logs | `<output dir>/tensorboard/` |
| Downloaded base models and Hub datasets | The standard Hugging Face cache (`~/.cache/huggingface` unless `HF_HOME` or `HF_HUB_CACHE` is set) |
| Hugging Face token | The Hugging Face cache directory, written by `hf auth login` |
| Playground disk offload (`POST /api/playground/infer` only) | A temporary folder under `MODELFORGE_OFFLOAD_DIR` or the system temp directory, deleted afterwards |

The user data directory comes from `platformdirs`:

| OS | User data directory |
|---|---|
| Linux | `~/.local/share/ModelForge` (or `$XDG_DATA_HOME/ModelForge`) |
| macOS | `~/Library/Application Support/ModelForge` |
| Windows | `%LOCALAPPDATA%\ForgeOpus\ModelForge` |

The browser console and the notebook entry point record runs in the same tables, which `modelforge history` prints. The terminal wizard records runs in a separate legacy table in the same file; see them with `modelforge history --legacy`.

## Environment variables

ModelForge itself reads very few environment variables. Most of the ones that matter belong to the libraries it uses.

| Variable | Read by | Effect |
|---|---|---|
| `MODELFORGE_OFFLOAD_DIR` | ModelForge (`web/playground.py`) | Folder for temporary disk offload when the `POST /api/playground/infer` endpoint can't fit a model in GPU and CPU memory. Defaults to the system temp directory. |
| `HF_TOKEN` | huggingface_hub; the terminal wizard checks it | Hugging Face access token, used instead of a cached login. |
| `HF_HOME`, `HF_HUB_CACHE` | huggingface_hub | Move the Hugging Face cache (models, datasets, token). |
| `HF_HUB_OFFLINE` | huggingface_hub | Disable Hub access. Recommendations search the Hub for models, so they can't find candidates while this is set. |
| `CUDA_VISIBLE_DEVICES` | PyTorch; **set by ModelForge** | ModelForge sets it from the GPUs a run uses, overwriting any value from your shell. Detection uses NVML and ignores it. Choose GPUs in the terminal wizard or with `gpu_indices` in the API instead. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | PyTorch | On Apple Silicon, run operators MPS doesn't implement on the CPU. Not set by v4. |

There is no configuration file. Server options are command-line flags; training settings come from the plan.

## Updating

```bash
pip install --upgrade modelforge-finetuning
```

On Windows, check `torch.cuda.is_available()` again after upgrading in case pip replaced your CUDA build of PyTorch.

## Resetting run history

Stop the server and delete `modelforge.db` from the user data directory. ModelForge creates an empty one on the next start. This forgets runs, checkpoints and registered models in the console, but does not delete any output directories or the Hugging Face cache.

## Next steps

- [Quick Start](../getting-started/quickstart.md): run a first fine-tune.
- [Browser Console](../user-guide/browser-console.md): every page of the console.
- [Hardware and Memory](../configuration/hardware-profiles.md): how detected hardware becomes a plan.
- [Common Issues](../troubleshooting/common-issues.md)
