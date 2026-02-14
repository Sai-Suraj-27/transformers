import torch
from transformers import AutoModel, AutoTokenizer


# 1. Setup
model_id = "jinaai/jina-embeddings-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()  # CRITICAL: Disables Dropout for deterministic results

# 2. Prepare Input
text = "Testing Jina V3 integration."
inputs = tokenizer(text, return_tensors="pt")

# 3. Prepare Adapter Mask (Force Task 0 for consistency)
# The remote code usually handles this, but let's be explicit if possible.
# For Jina V3, specific tasks are handled via 'task_id'.
# We will rely on default behavior or explicit inputs if the remote code supports it.
# (Assuming standard forward pass for now)

# 4. Capture Intermediate Layers using Hooks
activations = {}

def get_activation(name):
    def hook(model, input, output):
        # Detach and move to CPU to save memory/disk
        activations[name] = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
    return hook


# Register hooks on key layers
# Note: You might need to adjust these names based on 'print(model)' in this env
# Common guesses for Jina/XLM-R:
model.roberta.embeddings.word_embeddings.parametrizations.weight.original.register_forward_hook(get_activation("word_embeddings"))
model.roberta.embeddings.token_type_embeddings.parametrizations.weight.original.register_forward_hook(get_activation("token_type_embeddings"))
model.encoder.layer[0].register_forward_hook(get_activation('layer_0'))
model.encoder.layer[-1].register_forward_hook(get_activation('layer_last'))


# 5. Run Inference

with torch.no_grad():
    outputs = model(**inputs)


# 6. Save Everything
artifact = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "embeddings": activations["embeddings"],
    "layer_0": activations["layer_0"],
    "layer_last": activations["layer_last"],
    "final_output": outputs.last_hidden_state.detach().cpu()

}


torch.save(artifact, "jina_v3_debug_artifacts.pt")
print("✅ Artifacts saved to 'jina_v3_debug_artifacts.pt'")
