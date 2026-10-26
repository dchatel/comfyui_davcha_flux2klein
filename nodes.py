import sys
import torch
import folder_paths
import comfy.sd
import comfy.model_management
import nodes
from comfy_api.latest import io

class DavchaKleinCLIPLoader(io.ComfyNode):
    # Klein reads hidden states [9, 18, 27]: only Qwen3 layers 0-26 are used, the rest, the final norm and the lm_head are dropped at load.
    klein_model_options = {
        "qwen3_4b_model_config": {"num_hidden_layers": 27, "final_norm": False},
        "qwen3_8b_model_config": {"num_hidden_layers": 27, "final_norm": False, "lm_head": False},
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaKleinCLIPLoader",
            display_name="Klein CLIP Loader (Pruned)",
            category="davcha/flux2klein",
            inputs=[
                io.Combo.Input("clip_name", options=nodes.NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"].get_filename_list()),
            ],
            outputs=[
                io.Clip.Output(),
            ]
        )

    @classmethod
    def execute(cls, clip_name):
        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
        if not clip_path.endswith(".gguf"):
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.FLUX2, model_options=cls.klein_model_options)
            return io.NodeOutput(clip)

        loader = nodes.NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]
        gguf_nodes = sys.modules[loader.__module__]
        gguf_pkg = loader.__module__.rsplit(".", 1)[0]
        gguf_loader = sys.modules[gguf_pkg + ".loader"]
        gguf_ops = sys.modules[gguf_pkg + ".ops"]
        gguf_dequant = sys.modules[gguf_pkg + ".dequant"]

        # gguf_clip_loader dequantizes the whole token_embd table to fp16 (1.16 GiB for Qwen3-8B).
        # Keep it quantized and dequantize only the looked-up rows instead.
        sd, extra = gguf_loader.gguf_sd_loader(clip_path, is_text_model=True)
        if extra.get("arch_str") in ("qwen3", "qwen3vl"):
            sd = gguf_loader.sd_map_replace(sd, gguf_loader.LLAMA_SD_MAP)
        else:
            sd = gguf_loader.gguf_clip_loader(clip_path)

        class KleinGGMLOps(gguf_ops.GGMLOps):
            class Embedding(gguf_ops.GGMLOps.Embedding):
                def forward_ggml_cast_weights(self, input, out_dtype=None):
                    if self.weight.patches:
                        return super().forward_ggml_cast_weights(input, out_dtype)
                    rows = self.weight.data[input.flatten().to(self.weight.device)].to(input.device)
                    rows = gguf_ops.GGMLTensor(rows, tensor_type=self.weight.tensor_type, tensor_shape=(rows.shape[0], self.weight.tensor_shape[1]))
                    out = gguf_dequant.dequantize_tensor(rows, dtype=torch.float16)
                    return out.reshape(input.shape + out.shape[-1:]).to(dtype=out_dtype)

        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.FLUX2,
            state_dicts=[sd],
            model_options={
                "custom_operations": KleinGGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device(),
                **cls.klein_model_options,
            },
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = gguf_nodes.GGUFModelPatcher.clone(clip.patcher)
        return io.NodeOutput(clip)
