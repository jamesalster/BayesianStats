
using Random 

#handle missing values for model, vector
function handle_missing_values(
        X::Union{AbstractVector, AbstractMatrix};
        ignore_missing::Bool = false
    )::Union{AbstractVector, AbstractMatrix}

    missing_rows = vec(any(ismissing, X; dims = 2))

    if X isa AbstractVector
        no_missing = X[.!missing_rows]
    elseif X isa AbstractMatrix
        no_missing = X[.!missing_rows, :]
    end

    if any(missing_rows) 
        if ignore_missing
            @warn "Dropping $(sum(missing_rows)) missing values"
        else
            error("$(sum(missing_rows)) missing values found in input. Pass ignore_missing = true")
        end
    end

    return no_missing
end

#Get bayes factor against a threshold
function bayes_factor(x::AbstractArray{Float64}, threshold)
    threshold = Float64(threshold)
    return sum(x .> threshold) / sum(x .< threshold)
end

#Booststrap sample another function
function bootstrap(x::Union{AbstractVector, AbstractMatrix}, f::Function; samples = 1000, seed = 239)

    output_type = typeof(f(x))
    output = Vector{output_type}(undef, samples)

    n = size(x, 1)
    indices = Vector{Int}(undef, n)

    #sampler funciton
    if x isa AbstractVector
        sampler = (x, indices) -> f(view(x,indices))
    elseif x isa AbstractMatrix
        sampler = (x, indices) -> f(view(x, indices,:))
    end

    for i in 1:samples
        rand!(indices, 1:n)
        output[i] = sampler(x, indices)
    end

    return output
end
