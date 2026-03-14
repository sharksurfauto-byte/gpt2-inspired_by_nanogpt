import torch
from train_gpt2 import GPT, GPTConfig
from transformers import GPT2LMHeadModel

config_args={'n_layer':12,'n_head':12,'n_embd':768,'vocab_size':50257,'block_size':1024}
config=GPTConfig(**config_args)
model=GPT(config)
sd=model.state_dict()
print('ours keys sample:')
for k in list(sd.keys())[:20]: print(k)
print('---')
model_hf=GPT2LMHeadModel.from_pretrained('gpt2')
sd_hf=model_hf.state_dict()
print('hf keys sample:')
for k in list(sd_hf.keys())[:20]: print(k)
print('check ln names:', [k for k in sd.keys() if 'ln' in k][:20])
print('hf ln names:', [k for k in sd_hf.keys() if 'ln' in k][:20])
