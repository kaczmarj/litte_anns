function sigmoid(x::Array{Float64})::Array{Float64}
    return 1 ./ (1 .+ exp.(-x))
end

function sigmoid{T<:Number}(x::T)::T
    return 1 / (1 + exp(-x))
end

function hyperbolic_tan_sigmoid{T<:Number, N}(x::Array{T, N})::Array{T, N}
    return (exp.(x) .- exp.(-x)) ./ (exp.(x) + exp.(-x))
end

function hyperbolic_tan_sigmoid{T<:Number}(x::T)::T
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end

function positive_linear!{T<:Number, N}(x::Array{T, N})
    indx = x .< 0.
    x[indx] = [0. for i in 1:sum(Int64.(indx))]
    return 
end

function positive_linear{T<:Number, N}(x::Array{T, N})::Array{T, N}
    output = copy(x)
    indx = output .< 0.
    output[indx] = [0. for i in 1:sum(Int64.(indx))]
    return output
end

const INNER_ZERO_RETURN = 0.;

function positive_linear{T<:Number}(x::T)::T
    return max(x, INNER_ZERO_RETURN)
end

function link{T<:Number, N}(x::Array{T, N})::Array{T, N}
    return x
end

function link{T<:Number}(x::T)::T
    return x
end
