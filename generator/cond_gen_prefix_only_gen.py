import os
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np
import random


# -----------------------
# Config
# -----------------------
MODEL_NAME = "nferruz/ProtGPT2"  # replace if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KV_GENERATOR_PATH = "kv_prefix_model.pt"

class KVPrefixGenerator(nn.Module):
    """
    Condition -> past_key_values generator.

    We output a vector that is reshaped into (num_layers, 2, batch, num_heads, prefix_len, head_dim)
    where 2 = key / value.

    For efficiency we implement as a linear MLP producing required flattened vector.
    """
    def __init__(self, cond_dim, num_layers, num_heads, prefix_len, head_dim, hidden=1024):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.head_dim = head_dim

        # total size per example to output:
        self.out_dim = num_layers * 2 * num_heads * prefix_len * head_dim

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.out_dim)
        )

    def forward(self, cond):
        """
        cond: (batch, cond_dim)
        returns: a tuple of length num_layers, each entry (key, value) shapes:
                 key: (batch, num_heads, prefix_len, head_dim)
                 value: (batch, num_heads, prefix_len, head_dim)
        """
        batch = cond.size(0)
        x = self.mlp(cond)  # (batch, out_dim)
        # reshape
        x = x.view(batch, self.num_layers, 2, self.num_heads, self.prefix_len, self.head_dim)
        # split per layer and return list of tuples (key, value)
        past_key_values = []
        for layer_idx in range(self.num_layers):
            key = x[:, layer_idx, 0, :, :, :].contiguous()   # (batch, num_heads, prefix_len, head_dim)
            value = x[:, layer_idx, 1, :, :, :].contiguous() # (batch, num_heads, prefix_len, head_dim)
            past_key_values.append((key, value))
        return tuple(past_key_values)
    
    # -----------------------
# Inference / generation with KV prefix
# -----------------------
# --- Эту функцию можно разместить перед функцией генерации ---

def concat_past_key_values(past1, past2):
    """
    Объединяет два набора past_key_values по оси длины последовательности.
    """
    if past1 is None:
        return past2
    if past2 is None:
        return past1
    
    # past - это кортеж из (key, value) для каждого слоя
    # key/value имеют форму: [batch, heads, seq_len, head_dim]
    concatenated_past = []
    for (k1, v1), (k2, v2) in zip(past1, past2):
        # Объединяем по оси seq_len (ось 2)
        new_k = torch.cat([k1, k2], dim=2)
        new_v = torch.cat([v1, v2], dim=2)
        concatenated_past.append((new_k, new_v))
        
    return tuple(concatenated_past)

def generate_sequences_with_kv_prefix(
    kv_generator, model, tokenizer, cond_vector, prompt_text=None, 
    max_new_tokens=60, temperature=0.8, top_k=50, 
    repetition_penalty=1.2, num_sequences=1, seed=None
):
    kv_generator.eval(); model.eval()
    if seed is not None: torch.manual_seed(seed)
    
    cond_tensor = torch.tensor(cond_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        past = kv_generator(cond_tensor)
        key_from_first_layer = past[0][0] # Shape: [batch, heads, prefix_len, head_dim]
        # Переставляем оси, чтобы получить [batch, prefix_len, heads, head_dim]
        key_from_first_layer = key_from_first_layer.permute(0, 2, 1, 3)
        # Склеиваем головы: [batch, prefix_len, hidden_size]
        prefix_as_hidden_states = key_from_first_layer.reshape(
            key_from_first_layer.size(0), PREFIX_LEN, -1
        )
        prefix_embeds = model.transformer.h[0].attn.c_proj(prefix_as_hidden_states)

        prompt = f"<|endoftext|>\n{prompt_text}" if prompt_text else "<|endoftext|>"
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_token_len = input_ids.size(1)

        # Создаем маску внимания для промпта и префикса
        # Для генерации она просто состоит из единиц
        input_emb = model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, input_emb], dim=1)
        
        # Если генерируем несколько последовательностей, повторяем inputs_embeds
        if num_sequences > 1:
            inputs_embeds = inputs_embeds.repeat(num_sequences, 1, 1)

        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_sequences, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        total_prompt_len = PREFIX_LEN + prompt_token_len

        sequences = []
        for i in range(num_sequences):
            gen_ids = outputs[i][total_prompt_len:]
            seq_str = tokenizer.decode(gen_ids, skip_special_tokens=True)
            sequences.append(seq_str)
        return sequences

if __name__ == "__main__":
    print(f"Loading models for inference on device: {DEVICE}")

    # 1. Загружаем сохраненный словарь
    checkpoint = torch.load(KV_GENERATOR_PATH, map_location=DEVICE)

    # 2. Восстанавливаем конфигурацию из словаря
    model_config = checkpoint['model_config']
    PREFIX_LEN = model_config['prefix_len']
    CONDITION_DIM = checkpoint['condition_dim']
    cell_line_mapping = checkpoint['cell_line_mapping']
    
    print(f"Loaded config: PREFIX_LEN={PREFIX_LEN}, CONDITION_DIM={CONDITION_DIM}")

    # 3. Создаем пустую модель KVPrefixGenerator с правильной конфигурацией
    kv_generator = KVPrefixGenerator(
        cond_dim=CONDITION_DIM,
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        prefix_len=PREFIX_LEN,
        head_dim=model_config['head_dim']
    ).to(DEVICE)

    # 4. Загружаем веса в созданную модель
    kv_generator.load_state_dict(checkpoint['state_dict'])
    kv_generator.eval()

    # 5. Загружаем базовую модель ProtGPT2 и токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    print("Models loaded successfully.")

print("\n--- Generation for Multiple Cell Lines and Saving to CSV ---")
target_prompt = ""
total_sequences_to_generate = 500
num_to_print = 5
output_csv_path = "generated_cpp_sequences_prefix.csv"

all_available_lines = list(cell_line_mapping.keys())
target_cell_lines_list = random.choices(all_available_lines, k=total_sequences_to_generate)
print(f"Generating a total of {total_sequences_to_generate} sequences...")
all_results = []
for target_cell_line in tqdm(target_cell_lines_list, desc="Generating sequences"):
    cond_vector = [0.0] * CONDITION_DIM
    cond_vector[0] = 1.0
    cond_vector[1 + cell_line_mapping[target_cell_line]] = 1.0
    
    generated_list = generate_sequences_with_kv_prefix(
        kv_generator, model, tokenizer, cond_vector,
        prompt_text=target_prompt, num_sequences=1,
        max_new_tokens=100, temperature=0.9, top_k=50
    )
    
    # Собираем полную, чистую последовательность
    # prompt + сгенерированная часть
    full_sequence = target_prompt + generated_list[0]
    all_results.append({"sequence": full_sequence, "cell_line": target_cell_line})

print(f"\n--- Showing first {num_to_print} generated sequences ---")
for i, result in enumerate(all_results[:num_to_print]):
    print(f"\nSequence {i+1} (for cell line: {result['cell_line']}):")
    print(result['sequence'])

print("\nSaving results...")
results_df = pd.DataFrame(all_results)
results_df.to_csv(output_csv_path, index=False)
print(f"Successfully saved all {total_sequences_to_generate} sequences to '{output_csv_path}'")