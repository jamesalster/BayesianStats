
#### WIP ####
using Turing
using Distributions
using LinearAlgebra: cholesky
using FillArrays: I, Diagonal

#### Output Struct

struct BayesianCorrelation
    statistic::Vector{Float64}
    point_estimate::Float64
    x_est::Vector{Float64}
    y_est::Vector{Float64}
<<<<<<< Updated upstream
end

#### Model template

@model function correlation_model(
        x::Vector{Float64}, 
        y::Vector{Float64},
        distribution_fun = Normal)

    N = length(x) #checked to be equal to length(y)

    # Priors for location parameters
    μ₁ ~ Normal(0, 5)
    μ₂ ~ Normal(0, 5)
    
    # Priors for scale parameters
    σ₁ ~ Exponential(1)
    σ₂ ~ Exponential(1)
    
    # LKJCholesky prior
    L_Ω ~ LKJCholesky(2, 1.0)
=======
    posterior_sample::Chains
    distribution::String
end

#### Pretty print for that

function Base.show(io::IO, ::MIME"text/plain", b::BayesianCorrelation)
    println("""
    Bayesian Correlation:

    Base distribution: $(b.distribution)
    Fitted with $(prod(size(b.posterior_sample)[[1,3]])) posterior draws

    Estimated difference in means (y - x): $(round(b.point_estimate, digits = 2))
    95% Interval: $(round.(quantile(b.statistic, [0.025, 0.975]), digits = 2))

    Fields: 
        statistic: the distribution of the difference in means
        x_est and y_est: predicted population values for x and y
        posterior_sample: the Turing sample() object
    """
    )
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
    σ₁ ~ Exponential( 1 / (std(X[:,1]) * 2) )
    σ₂ ~ Exponential( 1 / (std(X[:,2]) * 2) )

    # LKJCholesky prior
    L_Ω ~ LKJCholesky(2, 2.0)
>>>>>>> Stashed changes
    
    ## Construct scale matrix
    D = Diagonal([σ₁, σ₂])
    L = D * Matrix(L_Ω.L)
    
    ## Extract correlation from Cholesky factor
<<<<<<< Updated upstream
    #Ω = Matrix(L_Ω.L) * Matrix(L_Ω.L)'
    #ρ = Ω[1,2]  # Store correlation for return

    ## Prior for correlation
    #Ρ ~ LKJ(1, 1.0)
    
    ### Construct covariance matrix
    #σ = [σ₁^2       ρ*σ₁*σ₂;
    #    ρ*σ₁*σ₂    σ₂^2]

    # Cholesky decomposition
    #L = cholesky(Σ).L

    #Println(Σ)
    
=======
    Ω = Matrix(L_Ω.L) * Matrix(L_Ω.L)'
    ρ = Ω[1,2]  # Store correlation for return

>>>>>>> Stashed changes
    # Non-centered parameterization
    z = Matrix{Float64}(undef, N, 2)
    for i in 1:N
        z[i,:] ~ MvNormal(zeros(2), I)
    end
    
    # Likelihood
    for i in 1:N
        # Transform z to correlated normal
        eps = L * z[i,:]
        
        # First component
<<<<<<< Updated upstream
        x[i] ~ Normal() * σ₁ + μ₁ + eps[1]
        
        # Second component
        y[i] ~ Normal() * σ₂ + μ₂ + eps[2]
    end

end

#@model function correlation_model(
#        x::Vector{Float64}, 
#        y::Vector{Float64},
#        distribution_fun = Normal)
#
#    N = length(x) #checked to be equal to length(y)
#
#    # Priors for location parameters
#    μ₁ ~ Normal(mean(x), std(x) * 2)
#    μ₂ ~ Normal(mean(y), std(y) * 2) 
#    
#    # Priors for scale parameters
#    σ₁ ~ Exponential( 1 / (std(x) * 2) )
#    σ₂ ~ Exponential( 1 / (std(y) * 2) )
#    
#    # LKJCholesky prior
#    L_Ω ~ LKJCholesky(2, 1.0)
#    
#    ## Construct scale matrix
#    D = Diagonal([σ₁, σ₂])
#    L = D * Matrix(L_Ω.L)
#    
#    ## Extract correlation from Cholesky factor
#    #Ω = Matrix(L_Ω.L) * Matrix(L_Ω.L)'
#    #ρ = Ω[1,2]  # Store correlation for return
#
#    ## Prior for correlation
#    #Ρ ~ LKJ(1, 1.0)
#    
#    ### Construct covariance matrix
#    #σ = [σ₁^2       ρ*σ₁*σ₂;
#    #    ρ*σ₁*σ₂    σ₂^2]
#
#    # Cholesky decomposition
#    #L = cholesky(Σ).L
#
#    #Println(Σ)
#    
#    # Non-centered parameterization
#    z = Matrix{Float64}(undef, N, 2)
#    for i in 1:N
#        z[i,:] ~ MvNormal(zeros(2), I)
#    end
#    
#    # Likelihood
#    for i in 1:N
#        # Transform z to correlated normal
#        eps = L * z[i,:]
#        
#        # First component
#        x[i] ~ distribution_fun() * σ₁ + μ₁ + eps[1]
#        
#        # Second component
#        y[i] ~ distribution_fun() * σ₂ + μ₂ + eps[2]
#    end
#
#end
=======
        X[i,1] ~ distribution * σ₁ + μ₁ + eps[1]
        
        # Second component
        X[i,2] ~ distribution * σ₂ + μ₂ + eps[2]
    end

    return ρ

end

>>>>>>> Stashed changes


#### Function to calculate correlation

function correlate(
<<<<<<< Updated upstream
        x::AbstractArray{Float64, 1},
        y::AbstractArray{Float64, 1};
        distribution = Normal,
        ignore_missing::Bool = false,
        point_estimate::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )
=======
        X::AbstractMatrix{<:Union{Missing, Float64}};
        distribution::Distribution = Normal(),
        ignore_missing::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )::BayesianCorrelation
    
    if size(X, 2) != 2
        error("Must pass in a two column matrix.")
    end

    model_X = handle_missing_values(X; ignore_missing = ignore_missing)

    model = correlation_model(
        model_X,
        distribution
    )

    @info "Sampling...."
    
    chain = sample(
        model,  
        NUTS(), 
        MCMCThreads(),
        round(Int, samples / chains), 
        chains,
        progress = true)

    #p = get_params(chain)
    # = reduce(hcat, generated_quantities(model, chain))'

    #rho = vec(q.ρ)

    #return BayesianCorrelation(
    #    rho,
    #    median(rho),
    #    q[:,1],
    #    q[:,2],
    #    chain,
    #    string(Meta.parse(string(distribution)))
    #)

    return model, chain
end

#### Method for x and y

function correlate(
    x::AbstractVector{<:Union{Missing, Float64}},
    y::AbstractVector{<:Union{Missing, Float64}};
    kwargs...)
>>>>>>> Stashed changes

    ## Check size
    if size(x) != size(y)
        error("The sizes of x and y must be the same.")
    end

<<<<<<< Updated upstream
    ## Handle missing values
    missing_x = ismissing.(x) 
    missing_y = ismissing.(y)
    any_missing = missing_x .| missing_y

    if any(any_missing) 
        if ignore_missing
            @warn "Dropping $(sum(any_missing)) missing values"
        else
            error("Missing values found in inputs. X: $(sum(missing_x)), Y: $(sum(missing_y))")
        end
    end

    x_model = x[.!any_missing]
    y_model = y[.!any_missing]

    model = correlation_model(
        x_model,
        y_model,
        distribution
    )

    #@info "Sampling...."
    
    #chain = sample(
    #    model,  
    #    NUTS(), 
    #    round(Int, samples / chains), 
    #    chains,
    #    progress = true) 

    return model
end

=======
    mat = hcat(x, y)

    return correlate(mat; kwargs...)
end
>>>>>>> Stashed changes
