# litte_anns
small scale simple artificial neural networks


## using activation functions:

```julia
using Distributions
include("Act.jl")

vals = rand(Normal(-1, 1), 1000);
relu_output = Act.positive_linear(vals);

# or if you want in place mutation
Act.positive_linear!(vals);


# alternatively 
include("activations.jl")
relu_output = positive_linear(vals);
# or
positive_linear!(vals);
```

## pytorch models
aside from the ground up software, there are networks built for toy problems in pytorch
