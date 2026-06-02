# GPT-2 (124M) Implementation in PyTorch

This repository contains a high-performance, from-scratch implementation of the GPT-2 architecture in PyTorch. The implementation is designed to be efficient, feature-rich, and compatible with pretrained weights from Hugging Face. It includes modern training optimizations such as Flash Attention, model compilation, and distributed training support.

## 🚀 Features

- **Architectural Fidelity**: Exact replication of the GPT-2 transformer architecture (124M to 1.5B parameters).
- **Flash Attention**: Integration of `torch.nn.functional.scaled_dot_product_attention` for memory-efficient and fast attention computation.
- **Model Compilation**: Uses `torch.compile` (Inductor) for significant speedups during training.
- **Distributed Training**: Full support for Distributed Data Parallel (DDP) for multi-GPU scaling.
- **Gradient Accumulation**: Ability to simulate large batch sizes (e.g., 0.5M tokens) on consumer hardware.
- **Mixed Precision**: Support for TF32 and BF16 (autocast) training.
- **Weight Sharing**: Implements the weight tying scheme between token embeddings and the language modeling head.
- **Hugging Face Integration**: Seamlessly load pretrained weights from `gpt2`, `gpt2-medium`, `gpt2-large`, and `gpt2-xl`.
- **Evaluation & Benchmarking**: Scripts and notebooks for evaluating model performance on HellaSwag and benchmarking against OpenAI GPT-2/GPT-3 baselines.
- **Visualization**: Jupyter notebook support for visualizing positional embeddings and attention maps.
- **Custom Optimizer**: AdamW implementation with targeted weight decay (applied only to 2D weight tensors).
- **Cosine LR Decay**: Learning rate scheduler with linear warmup and cosine annealing.

## 📁 Project Structure

- `train_gpt2.py`: The main script containing the model definition, data loader, and training loop.
- `inspect_keys.py`: Utility script to compare state dict keys between this implementation and Hugging Face's GPT-2.
- `input.txt`: Sample text data (Tiny Shakespeare) used for testing and small-scale experiments.
- `play.ipynb`: Jupyter notebook for interactive testing, visualization (embeddings, attention), and benchmarking.
- `inspect_keys.py`: Helper script to verify weight mapping and architecture compatibility.

## 🏗️ Model Architecture

The model follows the GPT-2 "Pre-Norm" configuration:
1.  **Token & Positional Embeddings**: Standard learnable embeddings.
2.  **Transformer Blocks**:
    - **LayerNorm**: Applied before the attention and MLP layers.
    - **Causal Self-Attention**: Multi-head attention with a causal mask to prevent looking at future tokens.
    - **MLP**: Feed-forward network with a $4\times$ expansion factor and GELU (tanh approximation) activation.
3.  **Final LayerNorm & Head**: A final normalization layer followed by a linear projection to the vocabulary size.

## 🛠️ Installation

```bash
pip install torch transformers tiktoken numpy matplotlib tqdm
```

## 📈 Training & Evaluation

### Data Loading
The `DataLoaderLite` supports sharded datasets. By default, it is configured for the `edu_fineweb10B` dataset, but can be adapted for local text like `input.txt` (Tiny Shakespeare).

### Optimization Strategy
- **Optimizer**: AdamW with $\beta_1=0.9, \beta_2=0.95$.
- **Weight Decay**: 0.1 (applied only to weights of Linear and Embedding layers).
- **Gradient Clipping**: Global norm clipped at 1.0.
- **Learning Rate**: Max LR of 6e-4 with 10% minimum LR.

### Benchmarking
The implementation is designed to match OpenAI's GPT-2 validation losses. For example, a 124M parameter model should reach a validation loss of approximately **3.29** on the FineWeb dataset, consistent with original benchmarks.

### HellaSwag Evaluation
Support for zero-shot evaluation on the HellaSwag dataset is included, allowing for direct comparison of reasoning capabilities against standard LLM baselines.

### Distributed Execution
To run with DDP across multiple GPUs:
```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

## 🧪 Usage

### Loading Pretrained Weights
You can initialize the model with OpenAI's weights:
```python
from train_gpt2 import GPT
model = GPT.from_pretrained('gpt2') # or 'gpt2-medium', etc.
```

### Generation
The repository includes a generation loop with top-k sampling:
```python
# (Snippet from train_gpt2.py)
model.eval()
# ... tokenization and input setup ...
logits, _ = model(idx)
probs = F.softmax(logits[:, -1, :], dim=-1)
topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
# ... sampling ...
```

### Visualization
Use `play.ipynb` to visualize the learned positional embeddings:
```python
import matplotlib.pyplot as plt
plt.imshow(model.transformer.wpe.weight.detach(), cmap='gray')
```

## 📚 References

- **Attention is All You Need**: [Paper](https://arxiv.org/abs/1706.03762)
- **Language Models are Unsupervised Multitask Learners (GPT-2)**: [Paper](https://openai.com/blog/better-language-models/)
- **Andrej Karpathy's build-nanogpt**: This implementation is heavily inspired by and follows the "build-nanogpt" series.

---
*Note: This project is intended for educational purposes and as a high-performance baseline for transformer experimentation.*
