# IWFUSION-SLM-V1 Model Runtime

This folder contains the model-side deployment assets for `IWFUSION-SLM-V1`.

## Recommended Azure runtime

Use the GGUF artifact in:

- `gguf/master_sovereign_v51_gold_q4_k_m.gguf`

Run it behind `llama.cpp` server on the model host.

Use:

- `azure_llama_cpp_launch.sh`

Backend app settings should then point to that host with:

- `IWFUSION_INFERENCE_BACKEND=llama_cpp`
- `LOCAL_SLM_BASE_URL=http://<model-host>:8080`
- `IWFUSION_MODEL_NAME=IWFUSION-SLM-V1`

## Included assets

### GGUF deployment artifact

- File: `gguf/master_sovereign_v51_gold_q4_k_m.gguf`
- SHA256: `a5acce4b55dfc70f245668bf2fe0625bd60b6ec5e46350904bd12d5dbfe1a5cc`

### Exact local MLX adapter

- Folder: `mlx_adapter/`

This matches the adapter used by the currently running local app runtime. It is included for provenance and exact local parity reference.

## Notes

- The Azure deploy path should use the GGUF model host unless you are intentionally reproducing the Apple Silicon MLX runtime.
- The local Mac `llama-server` binary is not bundled here because Azure requires a Linux build or container image.
- See `model_runtime_env.example` and `CHECKSUMS.txt`.

