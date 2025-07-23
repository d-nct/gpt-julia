#!/usr/bin/env julia

# Test script for GPT implementation
using Pkg
Pkg.activate(".")

include("src/gpt.jl")
include("src/data.jl")

using .gpt
using .Data
using Printf

function test_gpt()
    println("Testing GPT Implementation")
    println("=" ^ 40)
    
    # Test data loading
    println("1. Testing data loading...")
    vocab_size = Data.get_vocab_size()
    println("   Vocabulary size: $vocab_size")
    
    xb, yb = Data.get_batch("train", batch_size=2, seq_len=10)
    println("   Batch shape: $(size(xb)) -> $(size(yb))")
    
    # Test GPT model creation
    println("\n2. Testing GPT model creation...")
    config = gpt.GPTConfig(vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=32)
    model = gpt.GPT(config)
    println("   Model created successfully!")
    println("   Config: $(config.n_embd)d embedding, $(config.n_head) heads, $(config.n_layer) layers")
    
    # Test forward pass
    println("\n3. Testing forward pass...")
    test_input = [1, 2, 3, 4, 5]  # Small test sequence
    logits = model(test_input)
    println("   Input shape: $(length(test_input))")
    println("   Output shape: $(size(logits))")
    println("   Logits range: [$(minimum(logits)), $(maximum(logits))]")
    
    # Test text generation
    println("\n4. Testing text generation...")
    try
        prompt = "To be or not to be"
        generated = gpt.generate_gpt_text(model, prompt, max_new_tokens=50)
        println("   Prompt: '$prompt'")
        println("   Generated (first 100 chars): '$(generated[1:min(100, end)])'")
        println("   Generation successful!")
    catch e
        println("   Generation failed: $e")
    end
    
    println("\n5. Testing training function...")
    try
        # Test with very small model and few epochs
        small_config = gpt.GPTConfig(vocab_size, n_embd=32, n_head=2, n_layer=2, block_size=16)
        trained_model = gpt.train_gpt_simple(small_config, epochs=5, verbose=true)
        println("   Training completed successfully!")
    catch e
        println("   Training failed: $e")
    end
    
    println("\n" * "=" ^ 40)
    println("All tests completed!")
end

# Run the test
test_gpt()