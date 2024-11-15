
#handle missing values
function handle_missing_values(
        X::AbstractArray{T,N} where {T <: Union{Missing, Real}, N};
        ignore_missing::Bool = false
    )
    missing_count = count(ismissing, X)
    
    if missing_count > 0
        ignore_missing || error("$missing_count missing values found. Set ignore_missing=true to proceed")
        @warn "Dropping $missing_count missing values"
    end
    
    if X isa Vector
        return collect(skipmissing(X))
    else  # Matrix case
        return X[.!vec(any(ismissing, X, dims=2)), :]
    end
end


#Get bayes factor against a threshold
function bayes_factor(x::AbstractArray{Float64}, threshold::Real; more_than::Bool = true)
    threshold = Float64(threshold)
    bf = sum(x .> threshold) / sum(x .< threshold)
    return more_than ? bf : (1 / bf)
end

function p_value(x::AbstractArray{Float64}, threshold::Real; more_than::Bool = true)
    threshold = Float64(threshold)
    p_value = sum(x .> threshold) / length(x)
    return more_than ? p_value : (1 - p_value)
end


