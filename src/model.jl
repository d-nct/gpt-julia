# Implements the bigram language model, using a simple embedding matrix and a linear layer

module Model

using Random

struct BigramLanguageModel
    vocab_size::Int
    embedding_table::Array{Float32,2}  # weights
end

function BigramLanguageModel(vocab_size::Int)
    embedding_table = 0.01f0 * randn(Float32, vocab_size, vocab_size) # small random values
    return BigramLanguageModel(vocab_size, embedding_table)
end

function (model::BigramLanguageModel)(idx::Vector{Int})
    # idx is the context
    logits = model.embedding_table[:, idx]
    return logits
end

"""
Transform logits to a discrete probability distribution.

Y_i = exp(X_i) / sum(exp(X_i))

logits: (vocab_size, batch_size)
Returns: (vocab_size, batch_size)
"""
function softmax(logits::Matrix{Float32})
    exp_logits = exp.(logits .- maximum(logits, dims=1)) # for numerical stability
    return exp_logits ./ sum(exp_logits, dims=1)
end

"""
Measure the distance `H` between the predicted distribution `Y_p` and the target distribution `Y`.

H = -sum(Y * log(Y_p))

Y_p: (seq_len, batch_size, vocab_size)
Y: (seq_len, batch_size)
Returns: mean negative log likelihood.
"""
function cross_entropy_loss(Y_p, Y)
    seq_len, batch_size, vocab_size = size(Y_p)
    loss = 0.0f0
    for t in 1:seq_len
        for b in 1:batch_size
            target = Y[t, b]
            loss -= log(Y_p[t, b, target])
        end
    end
    return loss / (seq_len * batch_size)
end

function generate_text(model::Model.BigramLanguageModel, max_new_tokens::Int, context::String="")
    # set the starting context
    ctx = Tokenizer.encode(context)

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

    return Tokenizer.decode(ctx)
end

end # module Model