
#### WIP ####
using Turing
using Distributions
using LinearAlgebra
using FillArrays: I, Diagonal
using PDMats

#### Output Struct

struct BayesianCorrelation
    statistic::Vector{Float64}
    point_estimate::Float64
    x_est::Vector{Float64}
    y_est::Vector{Float64}
end

#### Model template

@model function correlation_model(
        X::Matrix{Float64}, 
        distribution::Distribution)

    N = size(X, 1) 

    # Priors for location parameters
    μ₁ ~ Normal(mean(X[:,1]), std(X[:,1]) * 2)
    μ₂ ~ Normal(mean(X[:,2]), std(X[:,2]) * 2)
    
    # Priors for scale parameters
    #σ₁ ~ Exponential( 1 / (std(X[:,1]) * 2) )
    #σ₂ ~ Exponential( 1 / (std(X[:,2]) * 2) )

    # Priors for scale parameters (using log-normal instead of exponential for better stability)
    logσ₁ ~ Normal(0, 1)
    logσ₂ ~ Normal(0, 1)
    σ₁ = exp(logσ₁)
    σ₂ = exp(logσ₂)
    
    # LKJCholesky prior
    L_Ω ~ LKJCholesky(2, 1.0)
    
    ## Construct scale matrix
    D = Diagonal([σ₁, σ₂])
    L = D * Matrix(L_Ω.L)
    
    # Extract correlation from Cholesky factor
    Ω = Matrix(L_Ω.L) * Matrix(L_Ω.L)'
    ρ = Ω[1,2]  # Store correlation for return

     #Non-centered parameterization
    z = Matrix{Float64}(undef, N, 2)
    for i in 1:N
        z[i,:] ~ MvNormal(zeros(2), I)
    end

    # Prior for correlation using LKJ
    #Ω ~ LKJCholesky(2, 2.0)
    
    ## Construct covariance matrix using Cholesky decomposition
    #L = LinearAlgebra.LowerTriangular(Ω.L)
    #D = Diagonal([σ₁, σ₂])
    #Σ = D * (L * L') * D
    #Σ = Symmetric(Σ)
    #ρ = (L * L')[1,2]

    ## Prior for correlation
    #ρ ~ LKJ(1, 2.0)
    
    ### Construct covariance matrix
    #Σ = [σ₁^2       ρ*σ₁*σ₂;
    #    ρ*σ₁*σ₂    σ₂^2]
    
    X_pred = Matrix(undef, N, 2)
    # Likelihood
    for i in 1:N
        # Transform z to correlated normal
        eps = L * z[i,:]
        # First component
        X[i, 1] ~ distribution * σ₁ + μ₁ + eps[1]
        # Second component
        X[i, 2] ~ distribution * σ₂ + μ₂ + eps[2]
        #X[i,:] ~ MvNormal([μ₁, μ₂], PDMat(Σ))

        X_pred[i,1] = rand(distribution) * σ₁ + μ₁ + eps[1]
        X_pred[i,2] = rand(distribution) * σ₁ + μ₁ + eps[2]
    end

    #X_pred = rand(MvNormal([μ₁, μ₂], PDMat(Σ)))

    return ρ, X_pred

end


#### Function to calculate correlation

function correlate(
        X::AbstractMatrix{<:Union{Missing, Float64}},
        distribution = Normal(),
        ignore_missing::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )

    if size(X, 2) != 2
        error("Must pass in a two column matrix.")
    end

    model_X = handle_missing_values(X; ignore_missing = ignore_missing)

    model = correlation_model(
        model_X,
        distribution
    )

    @info "Sampling...."
     
    chain = maximum_likelihood(model)
    return chain 
end
#    chain = sample(
#        model,  
#        NUTS(), 
#        MCMCThreads(),
#        samples,
#        chains,
#        progress = true)
#
#    #p = get_params(chain)
#    x_pred = reduce(hcat, generated_quantities(model, chain))'
#
#    return x_pred
#end
#    #rho = vec(1.ρ)
##
##    return BayesianCorrelation(
##        rho,
##        median(rho),
##        x_pred[:,1],
##        x_pred[:,2],
##        chain,
##        string(Meta.parse(string(distribution)))
##    )
##end
##
###### Method for x and y
##
##function correlate(
##    x::AbstractVector{<:Union{Missing, Float64}},
##    y::AbstractVector{<:Union{Missing, Float64}};
##    kwargs...)
##
##    ## Check size
##    if size(x) != size(y)
##        error("The sizes of x and y must be the same.")
##    end
##
##    mat = hcat(x, y)
##
##    return correlate(mat; kwargs...)
##end
##