include("Act.jl");

inputs = rand(10);
weights = [.3 for i in 1:length(inputs)];
bias = 1.;
output_relu = Act.positive_linear(dot(inputs, weights) + bias);
output_sig = Act.sigmoid(dot(inputs, weights) + bias);
