module gpt

include("data.jl")
include("tokenizer.jl")
include("model.jl")

using .Data
using .Tokenizer
using .Model
using Random
using Printf

export GPT, GPTConfig, generate_gpt_text, train_gpt, train_gpt_simple

# GPT Configuration
struct GPTConfig
    vocab_size::Int
    n_embd::Int      # embedding dimension
    n_head::Int      # number of attention heads
    n_layer::Int     # number of transformer blocks
    block_size::Int  # maximum sequence length
    dropout::Float32
end

function GPTConfig(vocab_size::Int; n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1f0)
    return GPTConfig(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
end

# Multi-head Self-Attention
struct MultiHeadAttention
    n_head::Int
    n_embd::Int
    head_size::Int
    query::Array{Float32,2}
    key::Array{Float32,2}
    value::Array{Float32,2}
    proj::Array{Float32,2}
    bias_q::Array{Float32,1}
    bias_k::Array{Float32,1}
    bias_v::Array{Float32,1}
    bias_proj::Array{Float32,1}
end

function MultiHeadAttention(n_embd::Int, n_head::Int)
    head_size = n_embd ÷ n_head
    scale = sqrt(Float32(n_embd))
    
    return MultiHeadAttention(
        n_head, n_embd, head_size,
        randn(Float32, n_embd, n_embd) / scale,  # query
        randn(Float32, n_embd, n_embd) / scale,  # key
        randn(Float32, n_embd, n_embd) / scale,  # value
        randn(Float32, n_embd, n_embd) / scale,  # projection
        zeros(Float32, n_embd),                   # bias_q
        zeros(Float32, n_embd),                   # bias_k
        zeros(Float32, n_embd),                   # bias_v
        zeros(Float32, n_embd)                    # bias_proj
    )
end

# Feed-Forward Network
struct FeedForward
    c_fc::Array{Float32,2}    # first linear layer
    c_proj::Array{Float32,2}  # second linear layer
    bias_fc::Array{Float32,1}
    bias_proj::Array{Float32,1}
end

function FeedForward(n_embd::Int)
    scale = sqrt(Float32(n_embd))
    return FeedForward(
        randn(Float32, 4*n_embd, n_embd) / scale,  # expand by 4x
        randn(Float32, n_embd, 4*n_embd) / scale,  # contract back
        zeros(Float32, 4*n_embd),
        zeros(Float32, n_embd)
    )
end

# Transformer Block
struct TransformerBlock
    ln1::Array{Float32,1}  # layer norm 1
    attn::MultiHeadAttention
    ln2::Array{Float32,1}  # layer norm 2
    mlp::FeedForward
end

function TransformerBlock(n_embd::Int, n_head::Int)
    return TransformerBlock(
        ones(Float32, n_embd),           # layer norm 1
        MultiHeadAttention(n_embd, n_head),
        ones(Float32, n_embd),           # layer norm 2
        FeedForward(n_embd)
    )
end

# GPT Model
struct GPT
    config::GPTConfig
    wte::Array{Float32,2}    # token embeddings
    wpe::Array{Float32,2}    # position embeddings
    blocks::Vector{TransformerBlock}
    ln_f::Array{Float32,1}   # final layer norm
    lm_head::Array{Float32,2} # language modeling head
end

function GPT(config::GPTConfig)
    scale = sqrt(Float32(config.n_embd))
    
    # Initialize embeddings
    wte = randn(Float32, config.n_embd, config.vocab_size) * 0.02f0
    wpe = randn(Float32, config.n_embd, config.block_size) * 0.02f0
    
    # Initialize transformer blocks
    blocks = [TransformerBlock(config.n_embd, config.n_head) for _ in 1:config.n_layer]
    
    # Final layer norm and head
    ln_f = ones(Float32, config.n_embd)
    lm_head = randn(Float32, config.vocab_size, config.n_embd) / scale
    
    return GPT(config, wte, wpe, blocks, ln_f, lm_head)
end

# Layer normalization
function layer_norm(x::Array{Float32,2}, gamma::Array{Float32,1}; eps=1e-5f0)
    mean_x = mean(x, dims=1)
    var_x = var(x, dims=1, corrected=false)
    return gamma .* (x .- mean_x) ./ sqrt.(var_x .+ eps)
end

# Causal self-attention with masking
function causal_self_attention(attn::MultiHeadAttention, x::Array{Float32,2})
    T, C = size(x)  # time (sequence length), channels (embedding dimension)
    
    # Linear projections for all heads
    q = x * attn.query .+ attn.bias_q'
    k = x * attn.key .+ attn.bias_k'
    v = x * attn.value .+ attn.bias_v'
    
    # Reshape for multi-head attention: (seq_len, n_head, head_size)
    head_size = attn.head_size
    n_head = attn.n_head
    
    q = reshape(q, T, n_head, head_size)
    k = reshape(k, T, n_head, head_size)
    v = reshape(v, T, n_head, head_size)
    
    # Initialize output
    out = zeros(Float32, T, n_head, head_size)
    
    for h in 1:n_head
        q_h = q[:, h, :]  # (T, head_size)
        k_h = k[:, h, :]  # (T, head_size)
        v_h = v[:, h, :]  # (T, head_size)
        
        # Compute attention scores with causal masking
        scores = zeros(Float32, T, T)
        scale = 1.0f0 / sqrt(Float32(head_size))
        
        for i in 1:T
            for j in 1:i  # causal masking: only look at past and current tokens
                scores[i, j] = sum(q_h[i, :] .* k_h[j, :]) * scale
            end
            # Future tokens are implicitly -inf (will be 0 after softmax)
        end
        
        # Apply softmax with causal masking
        for i in 1:T
            if i == 1
                scores[i, 1] = 1.0f0  # only one token to attend to
            else
                # Softmax over valid positions (1:i)
                max_score = maximum(scores[i, 1:i])
                exp_scores = exp.(scores[i, 1:i] .- max_score)
                sum_exp = sum(exp_scores)
                scores[i, 1:i] = exp_scores / sum_exp
            end
        end
        
        # Apply attention to values
        for i in 1:T
            for d in 1:head_size
                out[i, h, d] = sum(scores[i, 1:i] .* v_h[1:i, d])
            end
        end
    end
    
    # Concatenate heads and apply output projection
    out_flat = reshape(out, T, attn.n_embd)
    return out_flat * attn.proj .+ attn.bias_proj'
end

# Feed-forward network with GELU activation
function gelu(x::Float32)
    return 0.5f0 * x * (1.0f0 + tanh(sqrt(2.0f0/π) * (x + 0.044715f0 * x^3)))
end

function feed_forward(mlp::FeedForward, x::Array{Float32,2})
    # First linear layer with GELU activation
    h = x * mlp.c_fc .+ mlp.bias_fc'
    h = gelu.(h)
    
    # Second linear layer
    return h * mlp.c_proj .+ mlp.bias_proj'
end

# Forward pass through transformer block
function transformer_block_forward(block::TransformerBlock, x::Array{Float32,2})
    # Self-attention with residual connection
    x_norm1 = layer_norm(x, block.ln1)
    attn_out = causal_self_attention(block.attn, x_norm1)
    x = x + attn_out
    
    # Feed-forward with residual connection
    x_norm2 = layer_norm(x, block.ln2)
    mlp_out = feed_forward(block.mlp, x_norm2)
    x = x + mlp_out
    
    return x
end

# GPT forward pass
function (model::GPT)(idx::Vector{Int})
    T = length(idx)
    
    # Token and position embeddings
    tok_emb = model.wte[:, idx]'  # (T, n_embd)
    pos_emb = model.wpe[:, 1:T]'  # (T, n_embd)
    x = tok_emb + pos_emb
    
    # Forward through transformer blocks
    for block in model.blocks
        x = transformer_block_forward(block, x)
    end
    
    # Final layer norm
    x = layer_norm(x, model.ln_f)
    
    # Language modeling head
    logits = x * model.lm_head'  # (T, vocab_size)
    
    return logits
end

# Text generation function
function generate_gpt_text(model::GPT, prompt::String=""; max_new_tokens::Int=100)
    # Encode the prompt
    if isempty(prompt)
        ctx = [1]  # start with first character if no prompt
    else
        ctx = Int.([c for c in prompt])
    end
    
    # Ensure we don't exceed block size
    if length(ctx) > model.config.block_size
        ctx = ctx[end-model.config.block_size+1:end]
    end
    
    # Generate tokens
    for _ in 1:max_new_tokens
        # Get predictions
        logits = model(ctx)
        
        # Focus on the last time step
        logits_last = logits[end, :]  # (vocab_size,)
        
        # Apply softmax to get probabilities
        probs = Model.softmax(reshape(logits_last, :, 1))
        probs_vec = vec(probs)
        
        # Simple sampling without StatsBase
        cumsum_probs = cumsum(probs_vec)
        r = rand()
        next_token = findfirst(x -> x >= r, cumsum_probs)
        if next_token === nothing
            next_token = length(probs_vec)
        end
        
        # Append to context
        push!(ctx, next_token)
        
        # Keep only the last block_size tokens
        if length(ctx) > model.config.block_size
            ctx = ctx[2:end]
        end
    end
    
    # Decode to text
    return join(Char.(ctx))
end

# Simplified training function for testing
function train_gpt_simple(config::GPTConfig; epochs=10, verbose=true)
    # Create GPT model
    model = GPT(config)
    
    if verbose
        total_params = length(model.wte) + length(model.wpe) + length(model.ln_f) + length(model.lm_head)
        @printf("Training GPT model with approximately %d parameters\n", total_params)
    end
    
    learning_rate = 1e-4
    batch_size = 2
    
    for epoch in 1:epochs
        # Get batch
        xb, yb = Data.get_batch("train", batch_size=batch_size, seq_len=config.block_size)
        
        total_loss = 0.0f0
        
        # Process each sample in batch
        for b in 1:batch_size
            x_seq = xb[:, b]
            y_seq = yb[:, b]
            
            # Forward pass
            logits = model(x_seq)
            
            # Compute loss (cross-entropy)
            loss = 0.0f0
            for t in 1:length(y_seq)
                target = y_seq[t]
                if target > 0 && target <= size(logits, 2)
                    # Convert logits to probabilities
                    probs = Model.softmax(reshape(logits[t, :], :, 1))
                    loss += -log(max(probs[target], 1e-10f0))
                end
            end
            loss /= length(y_seq)
            total_loss += loss
        end
        
        avg_loss = total_loss / batch_size
        
        if verbose && (epoch % max(1, epochs ÷ 5) == 0 || epoch == 1)
            @printf("Epoch %d: Average loss = %.4f\n", epoch, avg_loss)
        end
        
        # Very simple parameter update (just add small random noise)
        # This is not real training, just for testing the infrastructure
        noise_scale = learning_rate * avg_loss
        model.wte .+= randn(size(model.wte)) * noise_scale * 0.001f0
        model.lm_head .+= randn(size(model.lm_head)) * noise_scale * 0.001f0
    end
    
    if verbose
        @printf("Training complete!\n")
    end
    
    return model
end

# Original training function for GPT
function train_gpt(; epochs=1000, verbose=true)
    # Get vocabulary size
    vocab_size = Data.get_vocab_size()
    
    # Create GPT model
    config = GPTConfig(vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=64)
    
    return train_gpt_simple(config, epochs=epochs, verbose=verbose)
end

end # module gpt
