include("src/data.jl")
include("src/model.jl")
include("src/tokenizer.jl")
include("src/train.jl")
include("src/generate.jl")

using .Data
using .Model
using .Tokenizer
using .Train
using .Generate

function demo_bigram()
    println("Bigram Language Model Demo")
    println("=" ^ 50)
    
    # Load data and get vocabulary size
    indices, tokenizer = Data.prepare_data()
    println("Data Statistics:")
    println("  - Vocabulary size: $(tokenizer.vocab_size) characters")
    println("  - Input file: data/input.txt (Shakespeare's Coriolanus)")
    
    # Create and Train model
    println("\nTraining Model:")
    print("  Training...")
    trained_model = Train.train(verbose=false)
    println(" done!")
    
    # Generate text
    println("\nGenerating Text:")
    context = "To be or not"
    println("  - Context: $context")
    text = Generate.generate_text(trained_model, tokenizer, 100, context)
    println("   - Generated text: $text")
end

demo_bigram()