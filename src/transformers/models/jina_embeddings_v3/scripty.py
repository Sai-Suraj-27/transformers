import gc
import torch
from transformers import AutoModel, AutoTokenizer


# model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
# torch.save(model.state_dict(), "hf_model")
#

sentences = ["How is the weather today?", "What is the current weather like today?"]
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
task = 'retrieval.query'
task_id = 0
# task_id = model._adaptation_map[task]
adapter_mask = torch.full((len(sentences),), task_id, dtype=torch.int32)

# with torch.no_grad():
#     model_output = model(**encoded_input, adapter_mask=adapter_mask)
#
# breakpoint()

# pp model_output.last_hidden_state[:, :3, :12]
hub_model= torch.tensor([[[-0.5846,  0.4540, -0.3864, -0.1645,  1.0021, -2.4062, -0.6457,
           1.0532,  0.0614, -0.2177, -1.0951, -1.4494],
         [-0.3704,  0.0773,  0.0720,  0.4730,  1.7826, -3.1329, -0.6470,
           1.7652,  0.0328, -0.6120, -1.7012, -1.4928],
         [-0.5613,  0.1269,  0.1533,  0.2271,  2.2457, -3.1786, -0.5471,
           1.8706,  0.1927, -0.7499, -1.9238, -1.7375]],

        [[-0.3541,  0.0986, -0.4278, -0.2811,  0.8666, -2.3081, -0.7612,
           1.0659, -0.0342, -0.5705, -1.2041, -1.4713],
         [-0.2480, -0.4323, -0.1916,  0.1471,  1.5355, -3.1510, -0.9542,
           1.6624,  0.2255, -1.1500, -2.0385, -1.8857],
         [-0.3643, -0.4196, -0.1048,  0.1165,  1.7631, -3.2737, -0.8841,
           1.7282,  0.1632, -1.1976, -2.0813, -2.0488]]])

my_model = torch.tensor([[[-4.2589e-01, -3.4301e-01, -1.5424e-01, -2.4937e-01,  7.1666e-01,
          -2.4199e+00, -5.3435e-01,  1.8074e+00, -3.6335e-02, -2.4025e-01,
          -1.1168e+00, -1.7075e+00],
         [-2.4236e-01, -5.9537e-04,  3.9819e-02,  2.1662e-01,  1.2775e+00,
          -2.3442e+00, -7.3929e-01,  1.8960e+00,  3.1878e-01, -7.0187e-01,
          -1.6294e+00, -1.7661e+00],
         [-8.6769e-01, -2.4549e-01,  4.3062e-01,  5.1737e-01,  1.5138e+00,
          -3.0687e+00, -6.4919e-01,  1.7481e+00, -2.1358e-01, -9.4096e-01,
          -1.9030e+00, -1.7239e+00]],

        [[-2.8589e-01,  1.9224e-01, -3.8558e-01, -2.4726e-01,  4.6594e-01,
          -2.2926e+00, -8.7748e-01,  8.5649e-01, -7.1104e-02, -7.0106e-01,
          -1.2906e+00, -1.2289e+00],
         [ 2.1806e-01, -7.9602e-02, -6.9187e-02,  2.9118e-01,  1.6030e+00,
          -2.9644e+00, -1.1610e+00,  1.6755e+00,  6.7469e-02, -8.7712e-01,
          -2.2431e+00, -2.1072e+00],
         [-3.5753e-01, -1.9169e-01, -1.1861e-01, -1.8302e-01,  1.5697e+00,
          -2.4902e+00, -7.0524e-01,  1.6016e+00, -6.0429e-02, -1.1129e+00,
          -2.3133e+00, -1.9838e+00]]])

# my model: pp model_output.last_hidden_state[:, 3, 12]
# tensor([2.0760, 1.5877])

is_match = torch.allclose(hub_model, my_model, atol=1e-5)

if is_match:
    print("✅ SUCCESS: The outputs match perfectly!")
else:
    print("❌ FAILURE: The outputs differ.")
    diff = (hub_model - my_model).abs().max().item()
    print(f"Max difference: {diff}")

breakpoint()

# from transformers.models.jina_embeddings_v3.configuration_jina_embeddings_v3 import JinaEmbeddingsV3Config
# from transformers.models.jina_embeddings_v3.modeling_jina_embeddings_v3 import JinaEmbeddingsV3Model
#
#
# config = JinaEmbeddingsV3Config()
# model = JinaEmbeddingsV3Model(config)
#
# old_state_dict = torch.load("hf_model", map_location="cpu")
# # print(model)
# rename = {
#     "embeddings.LayerNorm":"roberta.emb_ln",
#     "embeddings":"roberta.embeddings",
#     "encoder.layer":"roberta.encoder.layers",
#     "attention.attention_class.Wqkv":"mixer.Wqkv",
#     "attention.output.dense":"mixer.out_proj",
#     "attention.output.LayerNorm":"norm1",
#     "intermediate.dense":"mlp.fc1",
#     "output.dense":"mlp.fc2",
#     "output.LayerNorm":"norm2",
#     "pooler":"roberta.pooler",
# }
#
# new_state_dict = {}
#
# for old_key, old_value in old_state_dict.items():
#     new_key = old_key # Start by assuming no change
#
#     for pattern_new, pattern_old in rename.items():
#         if pattern_old in new_key:
#             new_key = new_key.replace(pattern_old, pattern_new)
#
#     new_state_dict[new_key] = old_value
#
# missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
#
# del old_state_dict
# gc.collect()
#
# with torch.no_grad():
#     model_output = model(**encoded_input, adapter_mask=adapter_mask)
#
# breakpoint()
#







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

# rename = {
#     "embeddings.LayerNorm":"roberta.emb_ln",
#     "embeddings":"roberta.embeddings",
#     "encoder.layer":"roberta.encoder.layers",
#     "attention.attention_class.Wqkv":"mixer.Wqkv",
#     "attention.output.dense":"mixer.out_proj",
#     "attention.output.LayerNorm":"norm1",
#     "intermediate.dense":"mlp.fc1",
#     "output.dense":"mlp.fc2",
#     "output.LayerNorm":"norm2",
#     "pooler":"roberta.pooler",
# }
#
#
# model = JinaEmbeddingsV3Model(JinaEmbeddingsV3Config)
#
# for key in model.state_dict().keys():
#     for rename_key, rename_value in rename.items():
#         # breakpoint()
#         if rename_key in key:
#             key = key.replace(rename_key, rename_value, 1)
#
#     print(key)

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
