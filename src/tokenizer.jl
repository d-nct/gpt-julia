module Tokenizer

export encode, decode, SimpleTokenizer

struct SimpleTokenizer
    chars::Vector{Char}
    str_to_int::Dict{Char, Int}
    int_to_str::Dict{Int, Char}
    vocab_size::Int

    # constructor
    function SimpleTokenizer(text::String)
        chars = sort(unique([c for c in text])) # TODO: this line is too much expensive
        vocab_size = length(chars)
        
        # mappings
        str_to_int = Dict{Char, Int}(c => i for (i, c) in enumerate(chars))
        int_to_str = Dict{Int, Char}(i => c for (i, c) in enumerate(chars))
        
        new(chars, str_to_int, int_to_str, vocab_size)
    end
end

"""
    encode(tokenizer::SimpleTokenizer, text::String)

Encode a string to a vector of integers according to the vocabulary.
"""
function encode(tokenizer::SimpleTokenizer, text::String)
    return [tokenizer.str_to_int[c] for c in text]
end

"""
    decode(tokenizer::SimpleTokenizer, tokens::Vector{Int})

Decode a vector of integers to a string according to the vocabulary.
"""
function decode(tokenizer::SimpleTokenizer, tokens::Vector{Int})
    return join([tokenizer.int_to_str[i] for i in tokens])
end

end # module Tokenizer