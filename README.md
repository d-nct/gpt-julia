# GPT from Scratch in Julia

A complete implementation of GPT (Generative Pre-trained Transformer) from scratch in Julia, without using large ML libraries. This implementation processes and generates text from Shakespeare's plays using character-level tokenization.

## 🏗️ Architecture

The implementation includes all key components of a transformer-based language model:

### Core Components

1. **Multi-Head Self-Attention** (`src/gpt.jl`)
   - Scaled dot-product attention
   - Causal masking for autoregressive generation
   - Multiple attention heads for different representation subspaces

2. **Feed-Forward Networks** 
   - Position-wise fully connected layers
   - GELU activation function
   - 4x expansion ratio

3. **Layer Normalization**
   - Applied before attention and feed-forward layers
   - Stabilizes training

4. **Positional Embeddings**
   - Learnable position representations
   - Added to token embeddings

5. **Transformer Blocks**
   - Combines attention + feed-forward + residual connections
   - Stacked to create deep networks

### Data Processing

- **Character-level tokenization** (`src/tokenizer.jl`)
- **Data batching and splitting** (`src/data.jl`)
- **Training data from Shakespeare's Coriolanus** (`data/input.txt`)

## 📁 File Structure

```
gpt/
├── src/
│   ├── gpt.jl          # 🧠 Main GPT implementation
│   ├── model.jl        # 📊 Bigram baseline model
│   ├── data.jl         # 🔄 Data loading and batching
│   ├── tokenizer.jl    # 🔤 Character-level tokenization
│   ├── generate.jl     # 📝 Text generation utilities
│   └── train.jl        # 🎯 Training utilities
├── data/
│   └── input.txt       # 📚 Shakespeare's Coriolanus
├── scripts/
│   └── train_model.jl  # 🚀 Training script
├── demo_gpt.jl         # 🎪 Demonstration script
├── test_gpt.jl         # 🧪 Test script
└── README.md           # 📖 This file
```

## 🚀 Usage

### Basic Usage

```julia
# Include the GPT module
include("src/gpt.jl")
using .gpt

# Create a GPT model
vocab_size = 100  # or get from Data.get_vocab_size()
config = gpt.GPTConfig(
    vocab_size,
    n_embd=256,     # embedding dimension
    n_head=8,       # number of attention heads
    n_layer=6,      # number of layers
    block_size=128  # sequence length
)

model = gpt.GPT(config)

# Generate text
text = gpt.generate_gpt_text(
    model, 
    "To be or not to be", 
    max_new_tokens=100
)
println(text)
```

### Training

```julia
# Train with default settings
trained_model = gpt.train_gpt(epochs=1000, verbose=true)

# Or train with custom configuration
config = gpt.GPTConfig(vocab_size, n_embd=128, n_head=4, n_layer=4)
trained_model = gpt.train_gpt_simple(config, epochs=500, verbose=true)
```

### Data Processing

```julia
include("src/data.jl")
using .Data

# Get vocabulary size
vocab_size = Data.get_vocab_size()

# Get training batches
x_batch, y_batch = Data.get_batch("train", batch_size=32, seq_len=64)
```

## 🧠 Implementation Details

### Multi-Head Attention

The attention mechanism implements:
- **Causal masking**: Prevents looking at future tokens
- **Scaled dot-product**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- **Multiple heads**: Different representation subspaces

```julia
# Causal mask ensures autoregressive behavior
for i in 1:T
    for j in 1:i  # only attend to past/current positions
        scores[i, j] = query[i] · key[j] / √head_size
    end
end
```

### Feed-Forward Network

```julia
# FFN with GELU activation
function feed_forward(x)
    h1 = gelu(x * W1 + b1)  # expand by 4x
    h2 = h1 * W2 + b2       # project back
    return h2
end
```

### GELU Activation

```julia
function gelu(x)
    return 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
end
```

## 🎯 Training Process

1. **Data Loading**: Character-level tokenization of Shakespeare
2. **Batching**: Create mini-batches with sequence length
3. **Forward Pass**: Through all transformer layers
4. **Loss Calculation**: Cross-entropy on next-token prediction
5. **Parameter Updates**: Gradient descent (simplified version)

## 📊 Model Configurations

### Small Model (Testing)
```julia
GPTConfig(vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=32)
# ~50K parameters
```

### Medium Model (Default)
```julia
GPTConfig(vocab_size, n_embd=128, n_head=8, n_layer=6, block_size=256)
# ~500K parameters
```

### Large Model
```julia
GPTConfig(vocab_size, n_embd=256, n_head=12, n_layer=12, block_size=512)
# ~5M parameters
```

## 🎪 Demo

Run the demonstration:

```bash
julia demo_gpt.jl
```

This will show:
- Model architecture details
- Parameter counting
- Text generation examples
- Training process explanation

## 🧪 Testing

```bash
julia test_gpt.jl
```

Tests include:
- Data loading verification
- Model creation and forward pass
- Text generation functionality
- Training loop execution

## 🔬 Technical Features

- **Pure Julia implementation**: No external ML frameworks
- **Educational focus**: Clear, readable code with comments
- **Modular design**: Separate components for easy understanding
- **Character-level**: Works directly with text characters
- **Shakespeare corpus**: Rich training data
- **Autoregressive generation**: Standard GPT text generation

## 🎓 Learning Objectives

This implementation helps understand:

1. **Transformer Architecture**: How attention, feed-forward, and normalization work together
2. **Causal Attention**: How autoregressive models prevent cheating
3. **Positional Encoding**: How transformers handle sequence order
4. **Text Generation**: How language models produce coherent text
5. **Training Dynamics**: How loss decreases and quality improves

## 🚧 Limitations

- **Simplified Training**: No automatic differentiation (for educational clarity)
- **CPU Only**: No GPU acceleration
- **Character-level**: Less efficient than subword tokenization
- **Small Scale**: Designed for learning, not production

## 🔄 Evolution from Bigram

The codebase shows the evolution from a simple bigram model to full GPT:

1. **Bigram Model** (`src/model.jl`): Simple lookup table
2. **GPT Model** (`src/gpt.jl`): Full transformer with attention

Compare the complexity:
- Bigram: O(V²) parameters (vocabulary squared)
- GPT: O(L×d²) parameters (layers × embedding dimension squared)

## 🏃‍♂️ Quick Start

1. **Clone/Navigate to project directory**
2. **Run demo**: `julia demo_gpt.jl`
3. **Train model**: `julia -e "include(\"src/gpt.jl\"); using .gpt; model = train_gpt(epochs=100)"`
4. **Generate text**: `julia -e "include(\"src/gpt.jl\"); using .gpt; println(generate_gpt_text(model, \"HAMLET:\"))"`

## 🎨 Customization

Modify hyperparameters in `GPTConfig`:
- `n_embd`: Embedding dimension (affects model capacity)
- `n_head`: Number of attention heads (affects parallelism)
- `n_layer`: Number of transformer blocks (affects depth)
- `block_size`: Maximum sequence length (affects memory)

## 📚 References

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- "The Annotated Transformer" (Rush, 2018)

---

**Built for educational purposes to understand transformer architecture from first principles! 🎓✨**