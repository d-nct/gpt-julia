module Train

include("data.jl")
include("model.jl")

using .Data
using .Model
using Printf
using Random

export train

function train(; verbose::Bool=true)
    # hiperparameters
    batch_size = 32
    block_size = 8
    max_iters = 3000
    eval_interval = 300
    learning_rate = 1e-2 # alpa
    device = "cpu" # TODO: use GPU
    eval_iters = 200

    # instantiate model
    vocab_size = Data.get_vocab_size()
    model = Model.BigramLanguageModel(vocab_size)

    if verbose
        @printf("Training model with %d parameters\n", vocab_size * vocab_size)
    end

    # training loop
    for iter in 1:max_iters
        # forward pass
        xb, yb = Data.get_batch("train", batch_size=batch_size, seq_len=block_size, percent_train=0.9)
        # matrix to vector
        xb = reshape(xb, block_size * batch_size)
        yb = reshape(yb, block_size * batch_size)

        logits = model(xb)
        loss = Model.cross_entropy_loss(logits, yb)

        if verbose
            if iter % eval_interval == 0
                @printf("Passo %d: perda de treino (log) %.4f\n", iter, loss)
            end
        end

        # backward pass (gradient descent)
        grad = gradient(m -> Model.cross_entropy_loss(m(xb), yb), model)

        # backpropagate the gradient
        for p in params(model)
            p.data .-= learning_rate * grad[p]
        end

        # update
        model.embedding_table .-= learning_rate * grad.embedding_table
    end

    if verbose
        @printf("Training complete!\n")
    end

    return model, loss
end

end # module Train