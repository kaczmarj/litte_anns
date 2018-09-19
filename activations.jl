# various vanilla activation functions
ALN = Union{Int64, Int32, Float64, Float32};

function sigmoid(x::Array{T})::Array{T} where {T<:ALN}
    return 1 ./ (1 .+ exp.(-x))
end

function sigmoid(x::T)::T where {T<:ALN}
    return 1 / (1 + exp(-x))
end

function hyperbolic_tan_sigmoid(x::Array{T})::Array{T} where {T<:ALN}
    return (exp.(x) .- exp.(-x)) ./ (exp.(x) + exp.(-x))
end

function hyperbolic_tan_sigmoid(x::T)::T where {T<:ALN}
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end

function positive_linear!(x::Array{T}) where {T<:ALN}
    indx = x .< 0.
    x[indx] = [0. for i in 1:sum(Int64.(indx))]
    return
end

function positive_linear(x::Array{T})::Array{T} where {T<:ALN}
    output = copy(x)
    indx = output .< 0.
    output[indx] = [0. for i in 1:sum(Int64.(indx))]
    return output
end

const INNER_ZERO_RETURN = Float64(0.);

function positive_linear(x::T)::Float64 where {T<:ALN}
    return max(x, INNER_ZERO_RETURN)
end

function link(x::Array{T})::Array{T} where {T<:ALN}
    return x
end

function link(x::T)::T where {T<:ALN}
    return x
end

function sym_hardlim(x::Array{T})::Array{T} where {T<:ALN}
    output = ones(Float64, size(x, 1))
    for (idx, val) in enumerate(x)
	    if val < 0
	        output[idx] = -1.
	    end
    end
    return output
end

function sym_hardlim(x::T)::T where {T<:ALN}
    if x < 0
	    return -1.
    end
    return 1.
end

function satlin(x::Array{T})::Array{T} where {T<:ALN}
    output = ones(Float64, size(x, 1))
    for (idx, val) in enumerate(x)
	if val < 0
	    output[idx] = 0.
	elseif 0 <= val <= 1
	    output[idx] = val
	else
	    output[idx] = 1.
	end
    end
    return output
end
