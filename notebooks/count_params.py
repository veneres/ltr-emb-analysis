# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")

# %%
for p in model.parameters():
    print(f"{p}: {p.numel()}")

# %%
model.num_parameters()

# %%
len(list(model.children()))

# %%
model_main_components = list(model.children())

# %%
base_model = model_main_components[0]
encoder = list(base_model.children())[0]
for name, part in encoder.named_children():
    print(name)

# %%
attention_params = 0
for name, p in model.named_parameters():
    if "attention" in name and p.requires_grad:
        attention_params += p.numel()

# %%
attention_params

# %%
# sanity check
all_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f"{all_params=}")
print(f"{model.num_parameters()=}")

# %%
attention_params / all_params

# %%
