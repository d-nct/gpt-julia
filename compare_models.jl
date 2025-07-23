#!/usr/bin/env julia

"""
Model Comparison: Bigram vs GPT

This script compares the original bigram language model with the new GPT implementation,
showing the architectural differences and capabilities.
"""

include("src/model.jl")
include("src/gpt.jl")
include("src/data.jl")

using .Model
using .gpt
using .Data
using Printf

function compare_models()
    println("🔬 Model Comparison: Bigram vs GPT")
    println("=" ^ 60)
    
    # Get vocabulary size
    vocab_size = Data.get_vocab_size()
    println("📊 Dataset: Shakespeare's Coriolanus")
    println("   - Vocabulary size: $vocab_size characters")
    
    println("\n🤖 Model Architectures:")
    
    # Bigram Model
    println("\n1️⃣  BIGRAM MODEL (Original)")
    println("   " * "-" ^ 30)
    bigram_model = Model.BigramLanguageModel(vocab_size)
    bigram_params = length(bigram_model.embedding_table)
    
    println("   Architecture:")
    println("   • Simple embedding lookup table")
    println("   • Maps current character → next character")
    println("   • No context beyond previous token")
    println("   • Parameters: $(bigram_params) ($(vocab_size) × $(vocab_size))")
    println("   • Memory: O(V²) where V = vocabulary size")
    
    # GPT Model
    println("\n2️⃣  GPT MODEL (New Implementation)")
    println("   " * "-" ^ 30)
    gpt_config = gpt.GPTConfig(vocab_size, n_embd=128, n_head=8, n_layer=6, block_size=256)
    gpt_model = gpt.GPT(gpt_config)
    
    # Calculate GPT parameters
    gpt_params = (
        length(gpt_model.wte) +           # token embeddings
        length(gpt_model.wpe) +           # position embeddings
        length(gpt_model.ln_f) +          # final layer norm
        length(gpt_model.lm_head) +       # output head
        sum([
            length(block.ln1) + length(block.ln2) +
            length(block.attn.query) + length(block.attn.key) + 
            length(block.attn.value) + length(block.attn.proj) +
            length(block.attn.bias_q) + length(block.attn.bias_k) +
            length(block.attn.bias_v) + length(block.attn.bias_proj) +
            length(block.mlp.c_fc) + length(block.mlp.c_proj) +
            length(block.mlp.bias_fc) + length(block.mlp.bias_proj)
            for block in gpt_model.blocks
        ])
    )
    
    println("   Architecture:")
    println("   • $(gpt_config.n_layer) transformer blocks")
    println("   • $(gpt_config.n_head) attention heads per block")
    println("   • $(gpt_config.n_embd)D embedding dimension")
    println("   • $(gpt_config.block_size) maximum sequence length")
    println("   • Multi-head self-attention with causal masking")
    println("   • Position-wise feed-forward networks")
    println("   • Layer normalization and residual connections")
    println("   • Parameters: $(gpt_params) (~$(round(gpt_params/1000, digits=1))K)")
    println("   • Memory: O(L×d²) where L = layers, d = embedding dim")
    
    println("\n📈 Complexity Comparison:")
    println("   " * "-" ^ 30)
    @printf("   %-20s %10s %15s\n", "Model", "Parameters", "Ratio")
    @printf("   %-20s %10d %15s\n", "Bigram", bigram_params, "1.0x")
    @printf("   %-20s %10d %15.1fx\n", "GPT", gpt_params, gpt_params/bigram_params)
    
    println("\n🧠 Capability Comparison:")
    println("   " * "-" ^ 30)
    
    println("\n   BIGRAM MODEL:")
    println("   ✓ Fast inference")
    println("   ✓ Simple architecture")
    println("   ✓ Low memory usage")
    println("   ✗ No long-range dependencies")
    println("   ✗ Limited context (1 token)")
    println("   ✗ Poor text quality")
    
    println("\n   GPT MODEL:")
    println("   ✓ Long-range dependencies")
    println("   ✓ Rich contextual understanding")
    println("   ✓ High-quality text generation")
    println("   ✓ Scalable architecture")
    println("   ✗ Higher computational cost")
    println("   ✗ More complex implementation")
    
    println("\n🔍 Technical Differences:")
    println("   " * "-" ^ 30)
    
    println("\n   Context Window:")
    println("   • Bigram: 1 token (only previous character)")
    println("   • GPT: $(gpt_config.block_size) tokens (full sequence)")
    
    println("\n   Attention Mechanism:")
    println("   • Bigram: None (simple lookup)")
    println("   • GPT: Multi-head self-attention with causal masking")
    
    println("\n   Position Information:")
    println("   • Bigram: None")
    println("   • GPT: Learnable position embeddings")
    
    println("\n   Nonlinearity:")
    println("   • Bigram: None (linear mapping)")
    println("   • GPT: GELU activation in feed-forward layers")
    
    # Test generation comparison
    println("\n📝 Generation Comparison:")
    println("   " * "-" ^ 30)
    
    test_prompt = "MARCIUS:"
    
    # Bigram generation (simulate)
    println("\n   BIGRAM OUTPUT (simulated):")
    println("   Input: '$test_prompt'")
    println("   Output: Random characters based on last token")
    println("   Quality: Incoherent, no grammar or meaning")
    
    # GPT generation
    println("\n   GPT OUTPUT:")
    println("   Input: '$test_prompt'")
    gpt_output = gpt.generate_gpt_text(gpt_model, test_prompt, max_new_tokens=100)
    println("   Output: '$(gpt_output[1:min(100, end)])...'")
    println("   Quality: Maintains character structure (before training)")
    
    println("\n🚀 Usage Examples:")
    println("   " * "-" ^ 30)
    
    println("\n   # Bigram Model Usage:")
    println("   bigram = Model.BigramLanguageModel(vocab_size)")
    println("   logits = bigram([1, 2, 3])  # context of tokens")
    println("   # Generate using Model.softmax and sampling")
    
    println("\n   # GPT Model Usage:")
    println("   config = gpt.GPTConfig(vocab_size, n_embd=256, n_head=8)")
    println("   model = gpt.GPT(config)")
    println("   text = gpt.generate_gpt_text(model, \"Hello\", max_new_tokens=50)")
    println("   trained = gpt.train_gpt(epochs=1000)")
    
    println("\n🎓 Learning Progression:")
    println("   " * "-" ^ 30)
    println("   1. Start with bigram model (understand basics)")
    println("   2. Learn attention mechanism")
    println("   3. Add position embeddings")
    println("   4. Implement feed-forward networks")
    println("   5. Stack layers with residuals")
    println("   6. Add layer normalization")
    println("   7. Complete GPT architecture")
    
    println("\n" * "=" ^ 60)
    println("🎯 Both models demonstrate different approaches to language modeling!")
    println("   • Bigram: Statistical frequency-based")
    println("   • GPT: Deep learning with attention mechanisms")
    println("\n🚀 Ready to explore both implementations!")
end

# Run the comparison
compare_models()