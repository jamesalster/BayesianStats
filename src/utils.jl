
function handle_missing_values(
        X::AbstractMatrix{<:Union{Missing, Float64}};
        ignore_missing::Bool = false
    )::Matrix{Float64}

    missing_rows = mapslices(ismissing, X; dims = 2)

    if any(missing_rows) 
        if ignore_missing
            @warn "Dropping $(sum(any_missing)) missing values"
        else
            error("$(sum(any_missing)) missing values found in input. Pass ignore_missing = true")
        end
    end

    return X[vec(.!missing_rows),:]
end
