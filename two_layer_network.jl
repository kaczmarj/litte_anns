using Distributions
include("Act.jl");

N_HIDDEN = 4;
N_LAYERS = 2;
N_INPUTS = 10;

inputs = rand(Normal(0, 2), N_INPUTS);
bias_per_layer = ones(N_LAYERS, N_HIDDEN);

weights_l1 = rand(N_HIDDEN, N_INPUTS);
weights_l2 = rand(N_HIDDEN, N_HIDDEN);

output_l1 = Act.sigmoid(weights_l1 * inputs + bias_per_layer[1, :]);
output_l2 = Act.link(weights_l2 * output_l1 + bias_per_layer[2, :]);

