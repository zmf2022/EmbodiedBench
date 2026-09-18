# Cosmos 3

RoboLab client for [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) (World-Action Model) over the OpenPI WebSocket protocol.

## Server (terminal 1)

```Shell
export CUDA_VISIBLE_DEVICES=0
conda run --no-capture-output -n cosmos3 python -u scripts/start_cosmos3_policy_server.py \
  --checkpoint-path /mnt/datadisk/models/Cosmos3-Nano-Policy-DROID \
  --port 8000
```

The wrapper disables the gated content-guardrail download and loads the
vendored `openpi_server` shim from `scripts/_shims`. Stop any Qwen vLLM on
8000 first, or use another port with `--remote-port` below.

## Client (terminal 2)

```Shell
conda activate pharm_flow
python policies/cosmos3/run.py --task BananaInBowlTask --remote-port 8000 --visualizer kit
```

Options: `--remote-uri` (full WS URI), `--remote-token` (or `COSMOS3_API_TOKEN`),
`--num-envs 10 --headless` (parallel).
