
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
    
    ## Construct scale matrix
    D = Diagonal([σ₁, σ₂])
    L = D * Matrix(L_Ω.L)
    
    ## Extract correlation from Cholesky factor
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


#### Function to calculate correlation

function correlate(
        x::AbstractArray{Float64, 1},
        y::AbstractArray{Float64, 1};
        distribution = Normal,
        ignore_missing::Bool = false,
        point_estimate::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )

    ## Check size
    if size(x) != size(y)
        error("The sizes of x and y must be the same.")
    end

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

