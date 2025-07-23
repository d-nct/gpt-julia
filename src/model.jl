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

function softmax(logits::Matrix{Float32})
    exp_logits = exp.(logits .- maximum(logits, dims=1)) # for numerical stability
    return exp_logits ./ sum(exp_logits, dims=1)
end

"""
    cross_entropy_loss(logits, targets)

logits: (seq_len, batch_size, vocab_size)
targets: (seq_len, batch_size)
Returns mean negative log likelihood.
"""
function cross_entropy_loss(logits, targets)
    seq_len, batch_size, vocab_size = size(logits)
    loss = 0.0f0
    for t in 1:seq_len
        for b in 1:batch_size
            target = targets[t, b]
            logit = softmax(logits[t, b, :])
            loss -= log(logit[target])
        end
    end
    return loss / (seq_len * batch_size)
end

end # module Model