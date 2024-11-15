
using Turing
using Distributions

#### Output Struct

struct BayesianTTest
    statistic::Vector{Float64}
    point_estimate::Float64
    bayes_factor::Float64
    p_value::Float64
    p_y_more_than_x::Float64
    posterior_sample::Chains
    distribution::String
end

#### Pretty print for that

function Base.show(io::IO, b::BayesianTTest)
     print(io, "BayesianTTest")
end

function Base.show(io::IO, ::MIME"text/plain", b::BayesianTTest)
    println("""
    Bayesian T Test:

    Distribution used: $(b.distribution)
    Fitted with $(prod(size(b.posterior_sample)[[1,3]])) posterior draws

    Estimated difference in means (y - x): $(round(b.point_estimate, digits = 3))
    95% Interval: $(round.(quantile(b.statistic, [0.025, 0.975]), digits = 3))

    Bayes Factor that difference in means (y - x) > 0: $(round(b.bayes_factor, digits = 3))
    P-value that difference in means (y - x) > 0: $(round(b.p_value, digits = 3)) 
    P-value that a an population value y > an population value x: $(round(b.p_y_more_than_x, digits = 3)) 

    Fields: 
        statistic: the distribution of the difference in means
        point_estimate: the median of that distribution
        bayes_factor: the bayes factor that difference in means > 0
        p_value: the probability that the difference in means > 0
        p_y_more_than_x: the probability that a population value y > a population value x
        posterior_sample: the Turing sample() object
        distribution: the distribution used in the test
    """
    )
end

#### Model template

@model function t_test_model(
        X::Matrix{Float64}, 
        distribution::Distribution)

    N = size(X, 1) 

    # Priors for location parameters
    μ₁ ~ Normal(mean(X[:,1]), std(X[:,1]) * 2)
    μ₂ ~ Normal(mean(X[:,2]), std(X[:,2]) * 2)
    
    # Priors for scale parameters
    σ₁ ~ Exponential( 1 / (std(X[:,1]) * 2) )
    σ₂ ~ Exponential( 1 / (std(X[:,2]) * 2) )

    # Likelihood
    for i in 1:N
        X[i,1] ~ distribution * σ₁ + μ₁
        X[i,2] ~ distribution * σ₂ + μ₂
    end

    # Predicted quantities
    X_pred = [
        rand(distribution * σ₁ + μ₁); 
        rand(distribution * σ₂ + μ₂)
    ]

    return X_pred
end

#### Function to calculate t_test - this is exported

function t_test(
        X::AbstractMatrix{<:Union{Missing, Float64}};
        distribution::Distribution = Normal(),
        ignore_missing::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )::BayesianTTest
    
    if size(X, 2) != 2
        error("Must pass in a two column matrix.")
    end

    model_X = handle_missing_values(X; ignore_missing = ignore_missing)

    model = t_test_model(
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

    p = get_params(chain)
    mu_diff = vec(p.μ₂ .- p.μ₁)

    q = reduce(hcat, generated_quantities(model, chain))'
    pred₁ = q[:,1]
    pred₂ = q[:,2]

    return BayesianTTest(
        mu_diff,
        median(mu_diff),
        bayes_factor(mu_diff, 0),
        p_value(mu_diff, 0),
        p_value(pred₂ .- pred₁, 0),
        chain,
        string(Meta.parse(string(distribution)))
    )
end

#### Method for x and y

function t_test(
    x::AbstractVector{<:Union{Missing, Float64}},
    y::AbstractVector{<:Union{Missing, Float64}};
    kwargs...)

    ## Check size
    if size(x) != size(y)
        error("The sizes of x and y must be the same.")
    end

    return t_test(hcat(x, y); kwargs...)
end
