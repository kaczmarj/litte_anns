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
```
