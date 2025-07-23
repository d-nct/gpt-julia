#!/usr/bin/env julia

"""
Demo script for GPT implementation

This script demonstrates how to use the GPT implementation from scratch.
Run this with: julia demo_gpt.jl

The implementation includes:
1. Multi-head self-attention with causal masking
2. Feed-forward networks with GELU activation
3. Layer normalization
4. Positional embeddings
5. Character-level tokenization
6. Text generation capabilities
"""

# Include our modules
include("src/gpt.jl")
include("src/data.jl")

using .gpt
using .Data

function demo_gpt()
    println("🤖 GPT Implementation Demo")
    println("=" ^ 50)
    
    # Load data and get vocabulary size
    vocab_size = Data.get_vocab_size()
    println("📊 Data Statistics:")
    println("   - Vocabulary size: $vocab_size characters")
    println("   - Input file: data/input.txt (Shakespeare's Coriolanus)")
    
    # Create a small GPT model for demonstration
    println("\n🧠 Creating GPT Model:")
    config = gpt.GPTConfig(
        vocab_size,
        n_embd=128,     # embedding dimension
        n_head=8,       # number of attention heads
        n_layer=6,      # number of transformer layers
        block_size=256, # maximum sequence length
        dropout=0.1f0
    )
    
    model = gpt.GPT(config)
    println("   - Embedding dimension: $(config.n_embd)")
    println("   - Number of attention heads: $(config.n_head)")
    println("   - Number of layers: $(config.n_layer)")
    println("   - Block size: $(config.block_size)")
    
    # Estimate number of parameters
    total_params = (
        length(model.wte) +           # token embeddings
        length(model.wpe) +           # position embeddings
        length(model.ln_f) +          # final layer norm
        length(model.lm_head) +       # output head
        sum([
            length(block.ln1) + length(block.ln2) +  # layer norms
            length(block.attn.query) + length(block.attn.key) + 
            length(block.attn.value) + length(block.attn.proj) +  # attention
            length(block.mlp.c_fc) + length(block.mlp.c_proj)     # feedforward
            for block in model.blocks
        ])
    )
    println("   - Approximate parameters: $(round(total_params/1000, digits=1))K")
    
    # Test text generation (before training)
    println("\n📝 Text Generation (before training):")
    prompt = "MARCIUS:"
    generated = gpt.generate_gpt_text(model, prompt, max_new_tokens=100)
    println("   Prompt: '$prompt'")
    println("   Generated: '$(generated[1:min(150, end)])...'")
    
    # Show training process
    println("\n🎯 Training Process:")
    println("   The model can be trained using:")
    println("   ```julia")
    println("   trained_model = gpt.train_gpt(epochs=1000, verbose=true)")
    println("   ```")
    println("   ")
    println("   Training features:")
    println("   - Character-level tokenization")
    println("   - Causal self-attention masking")
    println("   - Cross-entropy loss")
    println("   - Mini-batch gradient descent")
    
    # Show architecture details
    println("\n🏗️  Architecture Details:")
    println("   1. Token & Position Embeddings:")
    println("      - Character tokens -> $(config.n_embd)D vectors")
    println("      - Learnable position embeddings")
    
    println("\n   2. Transformer Blocks ($(config.n_layer)x):")
    println("      - Multi-Head Self-Attention ($(config.n_head) heads)")
    println("      - Causal masking (can't see future tokens)")
    println("      - Feed-Forward Network (4x expansion)")
    println("      - Residual connections & Layer Normalization")
    
    println("\n   3. Output:")
    println("      - Final layer normalization")
    println("      - Linear projection to vocabulary")
    println("      - Softmax for next-token probabilities")
    
    # Show usage examples
    println("\n💡 Usage Examples:")
    println("   # Create custom model")
    println("   config = gpt.GPTConfig(vocab_size, n_embd=256, n_head=8, n_layer=12)")
    println("   model = gpt.GPT(config)")
    println("   ")
    println("   # Generate text")
    println("   text = gpt.generate_gpt_text(model, \"To be or not\", max_new_tokens=200)")
    println("   ")
    println("   # Train model")
    println("   trained_model = gpt.train_gpt(epochs=5000)")
    
    println("\n✨ Key Features Implemented:")
    println("   ✓ Multi-head self-attention with causal masking")
    println("   ✓ Position-wise feed-forward networks")
    println("   ✓ Layer normalization")
    println("   ✓ Positional embeddings")
    println("   ✓ GELU activation function")
    println("   ✓ Residual connections")
    println("   ✓ Character-level tokenization")
    println("   ✓ Autoregressive text generation")
    println("   ✓ Cross-entropy loss computation")
    
    println("\n🚀 Ready to train and generate Shakespeare-like text!")
    println("=" ^ 50)
end

# Run the demo
demo_gpt()