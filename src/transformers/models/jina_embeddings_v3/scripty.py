import gc

import torch

from transformers import AutoTokenizer


device = torch.device("cpu")

if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Success! Using Intel GPU: {torch.xpu.get_device_name(0)}")
else:
    print("Warning: Using CPU. Check your installation.")


# model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
# torch.save(model.state_dict(), "hf_model")
#

sentences = ["How is the weather today?", "What is the current weather like today?"]
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
# task = 'retrieval.query'
# task_id = 0
# # task_id = model._adaptation_map[task]
# adapter_mask = torch.full((len(sentences),), task_id, dtype=torch.int32)

# with torch.no_grad():
#     model_output = model(**encoded_input, adapter_mask=adapter_mask)
#
# breakpoint()

from transformers.models.jina_embeddings_v3.configuration_jina_embeddings_v3 import JinaEmbeddingsV3Config
from transformers.models.jina_embeddings_v3.modeling_jina_embeddings_v3 import JinaEmbeddingsV3Model


config = JinaEmbeddingsV3Config()
model = JinaEmbeddingsV3Model(config)

old_state_dict = torch.load("src/transformers/models/jina_embeddings_v3/hf_model", map_location="cpu")
# print(model)
rename = {
    "embeddings.LayerNorm":"roberta.emb_ln",
    "embeddings":"roberta.embeddings",
    "encoder.layer":"roberta.encoder.layers",
    "attention.attention_class.Wqkv":"mixer.Wqkv",
    "attention.output.dense":"mixer.out_proj",
    "attention.output.LayerNorm":"norm1",
    "intermediate.dense":"mlp.fc1",
    "output.dense":"mlp.fc2",
    "output.LayerNorm":"norm2",
    "pooler":"roberta.pooler",
}

new_state_dict = {}

for old_key, old_value in old_state_dict.items():
    new_key = old_key # Start by assuming no change

    for pattern_new, pattern_old in rename.items():
        if pattern_old in new_key:
            new_key = new_key.replace(pattern_old, pattern_new)

    new_state_dict[new_key] = old_value

missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print(f"Missing: {missing}, unexpected: {unexpected}")

del old_state_dict
gc.collect()


model.to(device=device)
model.eval()

with torch.no_grad():
    # model_output = model(**encoded_input, adapter_mask=adapter_mask)
    model_output = model(**encoded_input, adapter_mask=None, output_attentions=True, output_hidden_states=True)
    print("Inference complete on device:", model.device)

breakpoint()

