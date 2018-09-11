function sigmoid(x::Array{Float64})::Array{Float64}
    return 1 ./ (1 . + exp.(-x))
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

function positive_linear(x::Array{Float64})::nothing
    indx = x .< 0
    nz = sum(Int64.(indx))
    x[indx] = [0. for i in 1:nz]
    return nothing
end
