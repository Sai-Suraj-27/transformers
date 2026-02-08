# from transformers import AutoTokenizer
# from .modeling_jina_embeddings_v3 import JinaEmbeddingsV3Model
# from .configuration_jina_embeddings_v3 import JinaEmbeddingsV3Config
from transformers import AutoConfig

from .modular_jina_embeddings_v3 import JinaEmbeddingsV3Model


# from ..jina_check.modeling_lora import XLMRobertaLoRA
# from ..jina_check.modeling_xlm_roberta import XLMRobertaModel
# from ..jina_check.configuration_xlm_roberta import XLMRobertaFlashConfig

# model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
# for key in model.state_dict().keys():
#     print(key)


JinaEmbeddingsV3Config = AutoConfig.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

# texts = [
#     "Follow the white rabbit.",  # English
#     "Sigue al conejo blanco.",  # Spanish
#     "Suis le lapin blanc.",  # French
#     "跟着白兔走。",  # Chinese
# ]

# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# encoded_input = tokenizer(texts, return_tensors='pt', padding=True)
# print("Encoded_Input: \n", encoded_input)

# model_jina = XLMRobertaLoRA(
#     config=JinaEmbeddingsV3Config,
#     roberta=XLMRobertaModel,
#     add_pooling_layer=True
# )
# print(model_jina)
# breakpoint()
# for key in model_jina.state_dict().keys():
#     if ("layer.0" in key) or ("embeddings" in key) or ("pooler" in key):
#         print(key, "       ", model_jina.state_dict()[key].shape)
#

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


model = JinaEmbeddingsV3Model(JinaEmbeddingsV3Config)

for key in model.state_dict().keys():
    for rename_key, rename_value in rename.items():
        # breakpoint()
        if rename_key in key:
            key = key.replace(rename_key, rename_value, 1)

    print(key)

    # print(key, "       ", model.state_dict()[key].shape)

    # if ("layer.0" in key) or ("embeddings" in key) or ("pooler" in key):
        # print(key, "       ", model.state_dict()[key].shape)



# output = model(**encoded_input)
# print("Output: \n", output)


# from transformers import AutoModel

# # Load the original Jina model (with trust_remote_code=True)
# model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

# # Print the keys of the first attention layer
# print("--- Attention Keys ---")
# for key in model.state_dict().keys():
#     if "layer.0" in key and "attention" in key:
#         print(key)
