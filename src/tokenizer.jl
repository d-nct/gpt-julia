module Tokenizer

export encode, decode

function encode(text::String)
    return [c for c in text]
end

function decode(tokens::Vector{Int})
    return join(tokens)
end

end # module Tokenizer