module Generate

include("model.jl")
include("tokenizer.jl")

using .Model
using .Tokenizer
using Random

export generate_text

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

    return Tokenizer.decode(ctx)
end

end # module Generate