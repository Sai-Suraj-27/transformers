"""Convert JinaEmbeddingsV3 checkpoint from hub to HF format."""

import os
import torch
from transformers import AutoConfig
from .modeling_jina_embeddings_v3 import JinaEmbeddingsV3Model


def rename_keys(key):
    """Rename keys from roberta prefix to match our model architecture."""
    # 1. Base Prefix Mappings
    if key.startswith("roberta.embeddings."):
        key = key.replace("roberta.embeddings.", "embeddings.")

    # 2. LayerNorm Mappings
    if "emb_ln." in key:
        key = key.replace("emb_ln.", "embeddings.LayerNorm.")

    # 3. Encoder Layer Mappings
    if "roberta.encoder.layers." in key:
        key = key.replace("roberta.encoder.layers.", "encoder.layer.")

    # 4. Attention Mappings
    if "mixer.Wqkv." in key:
        key = key.replace("mixer.Wqkv.", "attention.attention_class.Wqkv.")
    if "mixer.out_proj." in key:
        key = key.replace("mixer.out_proj.", "attention.output.dense.")
    if "norm1." in key:
        key = key.replace("norm1.", "attention.output.LayerNorm.")

    # 5. MLP Mappings
    if "mlp.fc1." in key:
        key = key.replace("mlp.fc1.", "intermediate.dense.")
    if "mlp.fc2." in key:
        key = key.replace("mlp.fc2.", "output.dense.")
    if "norm2." in key:
        key = key.replace("norm2.", "output.LayerNorm.")

    # 6. Pooler Mappings
    if "roberta.pooler." in key:
        key = key.replace("roberta.pooler.", "pooler.")

    return key

def convert_jina_checkpoint():
    print("Loading original Jina model...")
    # original_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    # original_state_dict = original_model.state_dict()

    # 1. Get the folder where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Join it with the filename (Removed the leading dot typo here too)
    file_path = os.path.join(script_dir, "jina_v3_original.pt")

    print(f"Reading weights from: {file_path}")
    original_state_dict = torch.load(file_path, map_location="cpu")

    print("Loading my implementation...")
    # Load your empty model structure
    config = AutoConfig.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    my_model = JinaEmbeddingsV3Model(config)

    print("Converting keys...")
    new_state_dict = {}
    for old_key, tensor in original_state_dict.items():
        new_key = rename_keys(old_key)

        # Optional: Print mismatch to debug
        # if old_key != new_key:
        #     print(f"{old_key} -> {new_key}")

        new_state_dict[new_key] = tensor

    print("Loading converted weights into your model...")
    # strict=True ensures every key matches exactly. If this fails, check the error message for missing keys.
    missing_keys, unexpected_keys = my_model.load_state_dict(new_state_dict, strict=True)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return my_model

if __name__ == "__main__":
    my_model = convert_jina_checkpoint()

    # Save the converted model so you can use it later
    print("Saving converted model...")
    my_model.save_pretrained("./converted_jina_v3")
    print("Done!")

