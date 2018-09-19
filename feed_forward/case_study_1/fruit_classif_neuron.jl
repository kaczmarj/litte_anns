using Distributions
include("../../Act.jl");


optimal_weights = [0, 1, 0];
optimal_bias = 0;

# here we investigate the classification ability of a simple 2 input 1 neuron
# artificial neural network model

inputs = [
    1 -1 -1; # should be an orange (-1 classification)
    1 1 -1; # apple, (1 classification)
    -1 -1 -1 # orange (-1 classification)
]

y_true = [-1, 1, -1];
y_predicted = Array{Int64}(undef, 3);

# predict
for stim_idx in 1:3
    y_predicted[stim_idx] = Act.sym_hardlim(
        sum(optimal_weights .* inputs[stim_idx, :]) + optimal_bias
        )
end

# measure model performance via sum squared error
sse_metric = sum((y_true .- y_predicted).^2)
