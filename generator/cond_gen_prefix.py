"""
KV-prefix generator for GPT2-style models (ProtGPT2). This skeleton
creates per-layer K/V prefixes from a condition vector and uses them
as `past_key_values` for the model during generation and training.

Notes & limitations:
- This code assumes a GPT2-like internal layout and that model.forward()
  accepts `past_key_values` argument (standard HF GPT2 supports this).
- During training, passing `past_key_values` with tensors that require_grad=True
  will allow gradients to flow into the generator.
- Check your model.config: num_hidden_layers, n_head, hidden_size -> derive head_dim.
"""

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

PREFIX_LEN = 8   # smaller than soft-prompt because KV per layer is heavy
LEARNING_RATE = 5e-4
BATCH_SIZE = 4
EPOCHS = 3
MAX_SEQ_LEN = 256

# -----------------------
# Load model / tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
# We freeze base model weights and only train the generator (PEFT-like)
for p in model.parameters():
    p.requires_grad = False

# Extract transformer sizes
num_layers = getattr(model.config, 'n_layer', getattr(model.config, 'num_hidden_layers', 36))
num_heads = getattr(model.config, 'n_head', getattr(model.config, 'num_attention_heads', 20))
hidden_size = getattr(model.config, 'n_embd', getattr(model.config, 'hidden_size', 1280))
head_dim = hidden_size // num_heads
assert head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

# -----------------------
# KV Generator
# -----------------------
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
# Training data skeleton (replace with your real data)
# -----------------------
class PeptideDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        cond, ids = self.items[idx]
        return torch.tensor(cond, dtype=torch.float32), torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):
    conds = torch.stack([b[0] for b in batch], dim=0)
    ids_tensors = [b[1] for b in batch]
    
    # GPT2 обычно не имеет pad_token_id, используем eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0 
        
    padded = pad_sequence(ids_tensors, batch_first=True, padding_value=pad_id)
    mask = (padded != pad_id).long()
    
    return conds, padded, mask

def load_dataset_from_csv(csv_path, tokenizer, max_seq_len):
    """
    Загружает данные из CSV файла с отформатированными последовательностями
    
    Args:
        csv_path: путь к CSV файлу с колонками 'sequence', 'is_cpp', 'cell_line'
        tokenizer: токенизатор ProtGPT2
        max_seq_len: максимальная длина последовательности в токенах
    
    Returns:
        dataset_items: список кортежей (condition_vector, token_ids)
        cell_line_mapping: словарь {cell_line_name: index}
        condition_dim: размерность условного вектора
    """
    import pandas as pd
    from tqdm import tqdm
    
    print(f"Loading data from {csv_path}...")
    
    # Улучшение: проверка существования файла
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Фильтруем только CPP пептиды для обучения
    df_cpp = df[df['is_cpp'] == 1].copy()
    print(f"Found {len(df_cpp)} CPP sequences out of {len(df)} total")
    
    if len(df_cpp) == 0:
        raise ValueError("No CPP sequences found in the dataset!")
    
    # Создаем mapping клеточных линий
    unique_cell_lines = sorted(df_cpp['cell_line'].unique())
    cell_line_to_idx = {cl: idx for idx, cl in enumerate(unique_cell_lines)}
    num_cell_lines = len(unique_cell_lines)
    condition_dim = 1 + num_cell_lines
    
    print(f"Found {num_cell_lines} unique cell lines: {unique_cell_lines[:5]}..." 
          if len(unique_cell_lines) > 5 else unique_cell_lines)
    print(f"Condition dimension: {condition_dim}")
    
    # Подготавливаем данные
    dataset_items = []
    skipped_count = 0
    error_types = {'unknown_cell_line': 0, 'tokenization_error': 0, 'empty_sequence': 0}
    
    for _, row in tqdm(df_cpp.iterrows(), total=len(df_cpp), desc="Preparing dataset"):
        # 1. Создаем условный вектор
        condition_vector = [0.0] * condition_dim
        condition_vector[0] = 1.0  # is_cpp = 1
        
        # Устанавливаем флаг для клеточной линии
        cell_line = row['cell_line']
        if cell_line not in cell_line_to_idx:
            error_types['unknown_cell_line'] += 1
            skipped_count += 1
            continue
            
        cell_line_idx = 1 + cell_line_to_idx[cell_line]
        condition_vector[cell_line_idx] = 1.0
        
        # 2. Токенизируем последовательность
        sequence = row['sequence']
        try:
            # add_special_tokens=False так как последовательности уже отформатированы
            token_ids = tokenizer.encode(sequence, add_special_tokens=False)
        except Exception as e:
            error_types['tokenization_error'] += 1
            skipped_count += 1
            continue
        
        # 3. Проверяем длину
        if len(token_ids) == 0:
            error_types['empty_sequence'] += 1
            skipped_count += 1
            continue
            
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        
        dataset_items.append((condition_vector, token_ids))
    
    # Выводим статистику
    print(f"\nDataset prepared: {len(dataset_items)} sequences")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} sequences:")
        for error_type, count in error_types.items():
            if count > 0:
                print(f"  - {error_type}: {count}")
    
    # Статистика по клеточным линиям
    cell_line_counts = df_cpp['cell_line'].value_counts()
    print("\nSequences per cell line:")
    for cl in unique_cell_lines[:5]:
        print(f"  {cl}: {cell_line_counts.get(cl, 0)} sequences")
    if len(unique_cell_lines) > 5:
        print(f"  ... and {len(unique_cell_lines) - 5} more")
    
    return dataset_items, cell_line_to_idx, condition_dim

# Загружаем данные ПЕРЕД созданием модели
dataset_items, cell_line_mapping, CONDITION_DIM = load_dataset_from_csv(
    'cpp_only_167.csv',
    tokenizer,
    MAX_SEQ_LEN
)

# Проверяем, что данные загружены
if not dataset_items:
    raise ValueError("No valid training data found!")

# ТЕПЕРЬ создаем KV generator с правильным CONDITION_DIM
kv_generator = KVPrefixGenerator(
    CONDITION_DIM, 
    num_layers, 
    num_heads, 
    PREFIX_LEN, 
    head_dim
).to(DEVICE)

optimizer = AdamW(kv_generator.parameters(), lr=LEARNING_RATE)

# Создаем PyTorch dataset и dataloader
dataset = PeptideDataset(dataset_items)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn
)

print(f"\nReady for training:")
print(f"  Total sequences: {len(dataset)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of batches: {len(dataloader)}")
print(f"  Condition dimension: {CONDITION_DIM}")
print(f"  Prefix length: {PREFIX_LEN}")

# -----------------------
# Training loop (using past_key_values)
# -----------------------
print("Start KV-prefix training on device:", DEVICE)
for epoch in range(EPOCHS):
    kv_generator.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for cond_batch, input_ids, att_mask in loop:
        cond_batch = cond_batch.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        att_mask = att_mask.to(DEVICE)

        # We will create past_key_values from cond_batch
        past = kv_generator(cond_batch)
        # HF GPT2 expects past_key_values in tuple((k,v), (k,v), ...), where each k/v has shape:
        # (batch, num_heads, seq_len, head_dim)
        # Our generator already outputs in that shape.

        # When using past_key_values, the model will consider that the tokens represented by past
        # already exist and will only compute logits for the new input_ids. Thus we need to ensure
        # labels and attention masks align.
        # For training we will feed the actual tokens (input_ids) and let the model predict them
        # conditioned on the prefix represented by past (prefix_len).
        # Note: the model will not return logits corresponding to past positions.

        # Создаем маску для префикса (всегда 1, так как префикс всегда "виден")
        batch_size = input_ids.shape[0]
        prefix_mask = torch.ones(batch_size, PREFIX_LEN, device=DEVICE, dtype=torch.long)
        # Объединяем маску префикса с маской для input_ids
        combined_att_mask = torch.cat([prefix_mask, att_mask], dim=1)

        # Build labels: normal causal LM labels for input_ids
        labels = input_ids.clone()
        labels[att_mask == 0] = -100  # ignore paddings

        outputs = model(input_ids=input_ids,
                        attention_mask=combined_att_mask,
                        labels=labels,
                        past_key_values=past)  # pass trainable past
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} average loss: {avg:.4f}")

# Save KV generator
save_dict = {
    'state_dict': kv_generator.state_dict(),
    'cell_line_mapping': cell_line_mapping,
    'condition_dim': CONDITION_DIM,
    'model_config': {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'prefix_len': PREFIX_LEN
    }
}
torch.save(save_dict, "kv_prefix_model.pt")

# -----------------------
# Inference / generation with KV prefix
# -----------------------
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
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past):
            cache.update(k, v, layer_idx)

        prompt = f"<|endoftext|>\n{prompt_text}" if prompt_text else "<|endoftext|>"
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_token_len = input_ids.size(1)

        # Если генерируем несколько последовательностей, нужно повторить past и input_ids
        if num_sequences > 1:
            input_ids = input_ids.expand(num_sequences, -1)
            # Для DynamicCache нужно обработать batch размерность по-другому
            # Создаем новый cache для батча
            batch_cache = DynamicCache()
            for layer_idx in range(len(past)):
                k, v = cache.key_cache[layer_idx], cache.value_cache[layer_idx]
                batch_k = k.repeat(num_sequences, 1, 1, 1)
                batch_v = v.repeat(num_sequences, 1, 1, 1)
                batch_cache.key_cache.append(batch_k)
                batch_cache.value_cache.append(batch_v)
            cache = batch_cache

        outputs = model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_sequences, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        sequences = []
        for i in range(num_sequences):
            # Срезаем промпт, чтобы получить только новую часть
            gen_ids = outputs[i][prompt_token_len:]
            seq_str = tokenizer.decode(gen_ids, skip_special_tokens=True)
            sequences.append(seq_str)
        return sequences

print("\n--- Generation for Multiple Cell Lines and Saving to CSV ---")
target_prompt = ""
total_sequences_to_generate = 500
num_to_print = 5
output_csv_path = "generated_cpp_sequences_prefix_tune.csv"

all_available_lines = list(cell_line_mapping.keys())
target_cell_lines_list = random.choices(all_available_lines, k=total_sequences_to_generate)
print(f"Generating a total of {total_sequences_to_generate} sequences...")
all_results = []
for target_cell_line in tqdm(target_cell_lines_list, desc="Generating sequences"):
    cond_vector = [0.0] * CONDITION_DIM
    cond_vector[0] = 1.0
    cond_vector[1 + cell_line_mapping[target_cell_line]] = 1.0
    
    # Вызываем нашу НОВУЮ функцию генерации
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