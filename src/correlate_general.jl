
#### A correlation that can accept any distribution ####
using Turing
using Distributions
using LinearAlgebra: cholesky
using FillArrays: I, Diagonal

#### Output struct defined in src/correlate.jl

#### Model template

@model function correlation_model_general(
        X::Matrix{Float64}, 
        distribution::Distribution)
    
    N, D = size(X)  # D should be 2 in this case
    
    # Pre-compute statistics once
    X_means = vec(mean(X, dims=1))
    X_stds = vec(std(X, dims=1))
    
    # Vectorized priors for location and scale
    μ ~ MvNormal(X_means, X_stds)
    σ ~ filldist(Exponential(0.5), D)
    
    # LKJCholesky prior
    L_Ω ~ LKJCholesky(2, 4.0)
    
    # Construct scale matrix and transform - done once
    L = Diagonal(σ) * Matrix(L_Ω.L)
    
    # Extract correlation if needed
    Ω = Matrix(L_Ω.L) * Matrix(L_Ω.L)'
    ρ = Ω[1,2]
    
    # Non-centered parameterization - vectorized
    z ~ filldist(MvNormal(zeros(2), I), N)
    
    # Transform all z at once
    ϵ = L * z
    
    # Likelihood using the passed distribution
    for i in 1:N
        X[i,1] ~ distribution * σ[1] + μ[1] + ϵ[1,i]
        X[i,2] ~ distribution * σ[2] + μ[2] + ϵ[2,i]
    end

    # Generate predictions
    X_pred = rand(distribution, N, 2) .* σ' .+ μ' + ϵ'

    return (ρ=ρ, X_pred=X_pred)
end


#### Function to calculate correlation

function correlate_general(
        X::AbstractMatrix{Float64};
        distribution::Distribution = Normal(),
        ignore_missing::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )

    if size(X, 2) != 2
        error("Must pass correlate() a 2-column matrix")
    end

    X_model = handle_missing_values(X; ignore_missing = ignore_missing)

    model = correlation_model_general(X_model, distribution)

    @info "Sampling..."

    chain = sample(
        model,  
        NUTS(), 
        MCMCThreads(),
        round(Int, samples / chains), 
        chains)

        # Get the GQ for all chains
    gq = generated_quantities(model, chain)

    # Extract rho and X_pred
    rho = vec([g.ρ for g in gq])  # Vector of rho values
    preds = reduce(hcat, [g.X_pred for g in gq])' 

    println(typeof(rho))
    println(typeof(preds))

    return BayesianCorrelation(
        rho,
        median(rho),
        preds[:,1],
        preds[:,2],
        chain,
        string(Meta.parse(string(distribution)))
    )

end

#### Method for 2 vectors
function correlate_general(
    x::AbstractVector{<:Union{Missing, Float64}},
    y::AbstractVector{<:Union{Missing, Float64}};
    kwargs...)

    if length(x) != length(y)
        error("x and y must be the same length")
    end

    return correlate_general([x y]; kwargs...)
end