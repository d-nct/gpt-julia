using Pkg
Pkg.activate(".")

include("../src/train.jl")
include("../src/generate.jl")

using .Train
using .Generate

# train the model
trained_model = Train.train(verbose=true)

# generate text!
println("Generating text...")
context = "Romeo: "
output = Generate.generate_text(trained_model, 100, context)
println(output)
