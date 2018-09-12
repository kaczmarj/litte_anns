include("../Act.jl");
using Distributions

N_DATA = 10;
N_HIDDEN = 3;
inputs = rand(Normal(0, 1), N_DATA);
weight_matrix = rand(Float64, (N_HIDDEN, N_DATA));
bias = ones(N_HIDDEN);
output = Act.positive_linear(weight_matrix * inputs + bias);
