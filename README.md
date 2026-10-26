# comfyui_davcha_flux2klein
Flux2 Klein nodes for ComfyUI.

## Klein CLIP Loader (Pruned)
Loads the Qwen3 text encoder for Flux2 Klein (4B and 8B, plain or Qwen3-VL) with only what Klein reads: hidden states 9, 18 and 27, so layers 27 and up, the final norm and the lm_head are not built. For GGUF files the token embeddings stay quantized and only the prompt's rows are dequantized.

Conditioning is identical to the standard loaders. With Qwen3-8B Q4_K_M, the text encoder weights on GPU go from 5.51 GiB to 3.20 GiB.

GGUF files need [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).
