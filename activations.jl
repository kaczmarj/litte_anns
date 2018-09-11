function sigmoid(x::Array{Float64})::Array{Float64}
    return 1 ./ (1 .+ exp.(-x))
end

function sigmoid(x::Float64)::Float64
    return 1 / (1 + exp(-x))
end

function hyperbolic_tan_sigmoid(x::Array{Float64})::Array{Float64}
    return (exp.(x) .- exp.(-x)) ./ (exp.(x) + exp.(-x))
end

function hyperbolic_tan_sigmoid(x::Float64)::Float64
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end

function positive_linear!(x::Array{Float64})
    indx = x .< 0.
    x[indx] = [0. for i in 1:sum(Int64.(indx))]
    return 
end

function positive_linear(x::Array{Float64})::Array{Float64}
    output = copy(x)
    indx = output .< 0.
    output[indx] = [0. for i in 1:sum(Int64.(indx))]
    return output
end

function positive_linear(x::Float64)::Float64
    return max(x, 0.)
end
