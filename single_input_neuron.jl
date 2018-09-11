include("Act.jl");

input = rand();
weight = 0.5;
bias = 1.;
output_relu = Act.positive_linear(input * weight + bias);
output_sig = Act.sigmoid(input * weight + bias);
