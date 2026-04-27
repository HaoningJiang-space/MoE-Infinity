import os
import torch
from transformers import AutoTokenizer
from moe_infinity import MoE

model_path = '/data/ziheng/hf_models/Qwen1.5-MoE-A2.7B-Chat'
offload_dir = '/data/ziheng/offload_moe_fgo_qwen15_smoke'
os.makedirs(offload_dir, exist_ok=True)
config = {
    'offload_path': offload_dir,
    'device_memory_ratio': 0.6,
    'prefetch': True,
    'offloading_policy': 'finegrained_history_reuse',
    'historical_library_capacity': 8,
    'historical_library_metric': 'cosine',
    'historical_library_admission': 'diversity_aware',
    'prefetch_backbone_topk': 8,
}

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model = MoE(model_path, config)
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Return exactly one short word: hello'},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda:0')
for step in range(2):
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    libsize = model.engine.offloading_policy.library_size() if model.engine.offloading_policy is not None else -1
    print(f'step={step} libsize={libsize} text={text[:200]}', flush=True)
