module Data

include("tokenizer.jl")

using .Tokenizer

export get_batch

macro INPUT_PATH()
    return "data/input.txt"
end

function prepare_data()
    path = @INPUT_PATH
    text = read(path, String)
    tokenizer = Tokenizer.SimpleTokenizer(text)
    indices = Tokenizer.encode(tokenizer, text)
    return indices, tokenizer
end

const _data_cache = Dict{Symbol, Any}()

function _get_splits(percent_train)
    if !haskey(_data_cache, :splits)
        indices, tokenizer = prepare_data()
        n = length(indices)
        n_train = Int(floor(percent_train * n))
        train_data = indices[1:n_train]
        val_data = indices[n_train+1:end]
        _data_cache[:splits] = (train_data, val_data, tokenizer)
    end
    return _data_cache[:splits]
end

function get_batch(split::String; batch_size=32, seq_len=16, percent_train=0.9)
    train_data, val_data, tokenizer = _get_splits(percent_train)
    data = split == "train" ? train_data : val_data
    n = length(data)
    x = zeros(Int, seq_len, batch_size)
    y = zeros(Int, seq_len, batch_size)
    for b in 1:batch_size
        start_idx = rand(1:n-seq_len)
        x[:, b] = data[start_idx:start_idx+seq_len-1]
        y[:, b] = data[start_idx+1:start_idx+seq_len]
    end
    return x, y
end

end # module Data