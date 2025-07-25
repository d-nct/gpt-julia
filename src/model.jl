# Implements the bigram language model, using a simple embedding matrix and a linear layer
module Model

include("tokenizer.jl")

using Random
using .Tokenizer

"""
V: vocabulary size
D: embedding dimension
T: sequence length
B: batch size
"""
struct BigramLanguageModel
    vocab_size::Int
    embedding_dim::Int
    embedding_table::Matrix{Float32} # embedding table (V, D)
end

function BigramLanguageModel(vocab_size::Int, embedding_dim::Int)
    embedding_table = 0.01f0 * randn(Float32, vocab_size, embedding_dim) # small random values
    return BigramLanguageModel(vocab_size, embedding_dim, embedding_table)
end

"""
Forward pass. Get logits for each token in the context.

x: (T, B)
Returns: (T, B, V)
"""
function (model::BigramLanguageModel)(x::Matrix{Int})
    T, B = size(x)

    # println("size(x): $(size(x))")
    # println("size(C): $(size(model.embedding_table))")

    x_flat = reshape(x, :) # (T*B,)
    # println("size(x_flat): $(size(x_flat))")
    embeddings_flat = model.embedding_table[x_flat, :] # (T*B, D)
    # println("size(embeddings_flat): $(size(embeddings_flat))")
    logits_flat = embeddings_flat * model.embedding_table' # (T*B, D) (D, V) -> (T*B, V)
    # println("size(logits_flat): $(size(logits_flat))")
    logits = reshape(logits_flat, T, B, model.vocab_size) # (T, B, V)
    # println("size(logits): $(size(logits))")
    # exit()
    return logits
end

"""
Transform logits to a discrete probability distribution.

Y_i = exp(X_i) / sum(exp(X_i))

logits: (T, B, V)
Returns: (T, B, V)
"""
function softmax(logits::Array{Float32}; dims::Int=1)
    max_logits = maximum(logits, dims=dims) # max trick
    shifted_logits = logits .- max_logits

    exp_logits = exp.(shifted_logits) # exp(X_i)
    return exp_logits ./ sum(exp_logits, dims=dims) # normalize
end

"""
Measure the distance `H` between the predicted distribution `logits` and the target distribution `Y`.

H = -sum(Y * log(Y_p))

logits: (T, B, V)
Y: (T, B)
Returns: mean negative log likelihood.
"""
function cross_entropy_loss(logits::Array{Float32, 3}, Y::Matrix{Int64})
    println("type of logits: $(typeof(logits))")
    println("type of Y: $(typeof(Y))")

    T, B, V = size(logits)

    logits_flat = reshape(logits, T*B, V) # (T*B, V)
    Y_flat = reshape(Y, T*B) # (T*B,)

    probs_flat = softmax(logits_flat, dims=2) # (T*B, V)
    correct_probs = [probs_flat[i, Y_flat[i]] for i in 1:T*B]

    return -sum(log.(correct_probs)) / (T * B)
end

function generate_text(model::Model.BigramLanguageModel, tokenizer::Tokenizer.SimpleTokenizer, max_new_tokens::Int, context::String="")
    # set the starting context
    ctx = Tokenizer.encode(tokenizer, context)

    # generate tokens
    for _ in 1:max_new_tokens
        # get predictions for last token
        context = [ctx[end]] # markovian property
        logits = model(context) # (1, vocab_size)

        # get probabilities
        probs = Model.softmax(logits)

        # sample from distribution
        ctx_next = rand(Categorical(probs))

        # append sampled token
        push!(ctx, ctx_next)
    end

    return Tokenizer.decode(tokenizer, ctx)
end

end # module Model