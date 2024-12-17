
#handle missing values for model, vector
function handle_missing_values(
        X::AbstractVector{<:Union{Missing, Float64}};
        ignore_missing::Bool = false
    )::Vector{Float64}

    missing_rows = ismissing.(X)

    if any(missing_rows) 
        if ignore_missing
            @warn "Dropping $(sum(missing_rows)) missing values"
        else
            error("$(sum(missing_rows)) missing values found in input. Pass ignore_missing = true")
        end
    end

    return X[.!missing_rows]
end

#Get bayes factor against a threshold
function bayes_factor(x::AbstractArray{Float64}, threshold)
    threshold = Float64(threshold)
    return sum(x .> threshold) / sum(x .< threshold)
end
