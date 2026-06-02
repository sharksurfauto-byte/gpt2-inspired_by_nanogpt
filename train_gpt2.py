from torch._inductor.fx_passes.split_cat import backend
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from transformers import GPT2LMHeadModel


# ------------------------------------------------------------------------------------------------------------------------------------------------

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # make sure no of embeddings can divide no of heads perfectly

        # key, query , value projections for all heads,but in a batch
        self.c_attn = nn.Linear(config.n_embd , 3*config.n_embd) 
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # this is just a scaling factor for the initialization of the weights
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a bias but more of a mask for masked self attention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size, config.block_size))
        
    def forward(self, x):
        B,T,C = x.size() # batch size, seq length and embed dimentionality

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim = 2)

        # nh = no of heads, hs = head size
        k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2) # (B,T, nh, hs) -> (B,nh,T,hs)
        q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2) # (B,T, nh, hs) -> (B,nh,T,hs)
        v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2) # (B,T, nh, hs) -> (B,nh,T,hs)

        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # basically the formula to calculate attention : (Q . K transpose) / sqrt(d)
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float ('-inf'))
        # att = F.softmax(att, dim = -1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


        # WE ARE GONNA APPLY FLASH ATTENTION:

        y = F.scaled_dot_product_attention(q,k,v, is_causal = True) # when this is called, Flash Attention is called and this is much faster as it takes into consideration Memory hierarchy wrt GPU cores (GPU sm > GPU HBM > CPU memory)

        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head o/p(s) side by side

        


        #output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # expands project from [batch, seq, embed] to [batch, seq, 4*embed]
        self.gelu = nn.GELU(approximate='tanh')               # applies GELU but with another verison including tanh
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd) # projects it back to [batch, seq, embed]
        self.c_proj.NANOGPT_SCALE_INIT = 1 # this is just a scaling factor for the initialization of the weights

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    # we have changed the block structure a bit in order to have a better flow of residuals
    def  __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size : int = 1024 # max seq length
    vocab_size : int = 50257 # no of tokens
    n_layer : int = 12 # no of layers
    n_head : int = 12 # no of heads
    n_embd : int = 768 # embed dim

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict( # ModuleDict and dict --> so that we could index or traverse like a dictionary
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of the token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of the position embeddings
            h = nn.ModuleList([Block(config) for _ in range (config.n_layer)]), # hidden blocks --> ModuleList as we wanna traverse using integers
            ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) # we have to project embd -> vocab_size and there are no bias

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        """
        Now as to why we are actually implementing the weight sharing scheme:
        The idea is that the token embedding and the final linear layer (lm_head) are essentially doing the same job
        of mapping tokens to a vector space and then back to tokens. By sharing the weights, we reduce the number of parameters
        in the model, which makes it smaller and faster to train.
        """

        # init params:
        self.apply(self.__init__weights) # this just applies the __init__weights func across all modules

    def __init__weights(self, module):
        if isinstance(module, nn.Linear): # if the module is a Linear module, we initialize with this
            std = 0.02
            # this is just a scaling factor for the initialization of the weights
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # the 2*n_layer is because of the residual connection of the 1. Casual self attn and 2. MLP
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None: # if there is a bias, we initialize it with 0
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # for embedding modules:
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self,idx, targets = None):
        # idx is the shape of B,T
        B,T = idx.size()
        assert T<=self.config.block_size, f"Cannot forwards seq of length {T}, block size is only {self.config.block_size}"

        # forward the token and the pos embed
        pos = torch.arange(0,T, dtype = torch.long, device = idx.device) # shape T
        pos_emb = self.transformer.wpe(pos) # pos embed of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embed of shape (B,T, n_embd)
        x = tok_emb + pos_emb

        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # cross entropy cant actually take 3 dims as inputs...so what its doing is that it is flattening it into 2 dims (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head, n_embd are determined form model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer, not a param

        # init a HF/transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just a mask
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:

            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # CUSTOM CONFIGURATION FOR ADAM OPTIMIZER
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all candidate params that require grad
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] 
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # 1-D params shouldnt be decayed
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
# ------------------------------------------------------------------------------------------------------------------------------------------------

num_return_sequences = 5
max_length = 30

# ------------------------------------------------------------------------------------------------------------------------------------------------
import time

device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'

# Using Distributed Data Parallel (DPP)
from torch.distributed import init_process_group, destroy_process_group

# set up DDP
# torchrun command sets up the env variables RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), 'for now i think we need cuda for ddp'
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank ==0
else:
    # vanilla (or) non ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size=1
    master_process=True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'Using device: {device}')


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# Now we define the GRADIENT ACCUMULATION:
"""

The GPT paper has a batch size of 0.5M tokens for similar sized model as ours. But ofc we cant pass such a huge no or else our gpu will explode. So we gradient accumulate to simulate the effect of having 0.5M tokens as batch size.

"""

total_batch_size = 524288 # 2^19 ~ 0.5M in terms of tokens
B=16
T=1024
assert total_batch_size % (B*T*ddp_world_size) == 0, 'make sure total_batch_size is divisible by B*T*ddp_world_size'
grad_accum_steps = total_batch_size//(B*T*ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}.")
    print(f"=> calculated grad accumulation steps = {grad_accum_steps}.")


# ------------------------------------------------------------------------------------------------------------------------------------------------

# setting up a DataLoader
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype = torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B,T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # # at init load tokens from disk and store them in memory
        # with open ('input.txt', 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # # state
        # self.current_position = self.B * self.T * self.process_rank 

        # get the shard files:
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        assert len(shards)>0, f'no shards found in split {split}.'
        if master_process:
            print(f"found {len(shards)} shards for split {split}.")
        self.reset()

        # state, init at shard zero
        def reset():
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T+1]
        x = (buf[:-1]).view(B,T) # inputs
        y = (buf[1:]).view(B,T) # targets

        # update the pointer
        self.current_position += B*T* self.num_processes
        #if laoding the next batch would be out of bounds, reset
        if self.current_position + (B*T*self.num_processes+1)> len(self.tokens):
            self.current_shard = (self.current_shard+1)%len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position =  B*T*self.process_rank 
        
        return x,y

train_loader = DataLoaderLite(B=4, T=256, process_rank = ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=4, T=256, process_rank = ddp_rank, num_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('high') # reduces memory usage and increases speed as we are now using tf32 instead of fp32

# create model

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model) # basically increases the speed of the program a lot by compiling it first (like compilers used c, c++)

if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])

# WE DO LEARNING RATE SCHEDULING:
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
        # 1. linear warmup for warmup_iter steps
        if it<warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2. if it > lr_decay_steps return min lr
        if it > max_steps:
            return min_lr
        # 3. in btw, use cosine decay down to min lr
        decay_ratio = (it-warmup_steps)/(max_steps-warmup_steps)
        assert 0<= decay_ratio <=1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes till 0
        return min_lr + coeff * (max_lr - min_lr)

#optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas = (0.9, 0.95), eps = 1e-8)

# we are gonna use a custom optimizer function:
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate = 6e-4, device = device)

# ------------------------------------------------------------------------------------------------------------------------------------------------

""" TRAINING LOOOP  """

for step in range(50):
    t0 = time.time() 
    
    optimizer.zero_grad() # starts optmizer with 0
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        # with (torch.autocast(device_type=device, dtype = torch.bfloat16)):     # add when u have cuda access
        logits, loss = model(x,y) # NOTE: torch.autocast with bfloat16 is a GPU optimization, skipping on CPU
        loss = loss/grad_accum_steps # had a whole ass reasoning as why MSE has issues without this step
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step==grad_accum_steps-1)
        loss.backward() # adds the losses to the optimizer
    
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # we clip the global norm of the gradient at 1.0 (scr = GPT3 paper)
    # determine and set the lr for the current step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr # changing the param lr
    optimizer.step() # updates the params and dec the loss
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000 # time diff in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps*ddp_world_size)/(t1-t0)
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.4f} | lr: {lr:.4f} | norm: {norm:.4f} | dt: {dt:.4f}ms | tokens/sec: {tokens_per_sec}")
if ddp:
    destroy_process_group()

"""
We are getting a loss of 11.04. Now why is it somewhat good?
cause at model initialization we want a very uniform probability distribution or loss distribution across all the 50257 words in the vocab so it doesnt have any initial bias. 
so we want like a 1/50527 kinda thingy.
so our cross entropy loss should look like --> -ln(1/50257) which is approx 10.82
"""

import sys; sys.exit(0)

# generate! right now x is (B,T) where B=5, T=8
# set seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # B,T,vocab_size
        # taking the logits at the last pos
        logits = logits[:,-1, :] # (B, vocab_size)
        #get the probs
        probs = F.softmax(logits, dim = -1)
        # do top-k sampling of 50 ( HF default )
        # top-k probs here becomes (5,50), topk indices is (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
        # sample from the top-k probs
        ix = torch.multinomial(topk_probs,1) # (B,1)
        #gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) #(B,1)
        # append to the seq
        x = torch.cat((x, xcol), dim = 1)

# print the generated text
for i in range(num_return_sequences ):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens) # converting token back to string 
    print(">", decoded)
