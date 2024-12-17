
using Turing
using Distributions

#### Output Struct

struct BayesianTTest
    statistic::Vector{Float64}
    point_estimate::Float64
    bayes_factor::Float64
    x_est::Vector{Float64}
    y_est::Vector{Float64}
    posterior_sample::Chains
    distribution::String
end

#### Pretty print for that

function Base.show(io::IO, ::MIME"text/plain", b::BayesianTTest)
    println("""
    Bayesian T Test:

    Model Distribution: $(b.distribution)
    Fitted with $(prod(size(b.posterior_sample)[[1,3]])) posterior draws

    Estimated difference in means (y - x): $(round(b.point_estimate, digits = 3))
    Bayes Factor that (y - x) > 0: $(round(b.bayes_factor, digits = 3))
    95% Interval: $(round.(quantile(b.statistic, [0.025, 0.975]), digits = 3))

    Fields: 
        statistic: the distribution of the difference in means
        x_est and y_est: predicted population values for x and y
        posterior_sample: the Turing sample() object
    """
    )
end

#### Model template

@model function t_test_model(
        X::Vector{Float64}, 
        Y::Vector{Float64}, 
        distribution::Distribution)

    N_x = length(X)
    N_y = length(Y)

    # Priors for location parameters
    μ₁ ~ Normal(mean(X), std(X) * 2)
    μ₂ ~ Normal(mean(Y), std(Y) * 2)
    
    # Priors for scale parameters
    σ₁ ~ Exponential( 1 / (std(X) * 2) )
    σ₂ ~ Exponential( 1 / (std(Y) * 2) )

    # Likelihood
    for i in 1:N_x
        X[i] ~ distribution * σ₁ + μ₁
    end
    for i in 1:N_y
        Y[i] ~ distribution * σ₂ + μ₂
    end

    # Predicted quantities
    x_pred = rand(distribution * σ₁ + μ₁)
    y_pred = rand(distribution * σ₂ + μ₂)

    return [x_pred; y_pred]
end

#### Function to calculate t_test - this is exported

function t_test(
        X::AbstractVector{<:Union{Missing, Float64}},
        Y::AbstractVector{<:Union{Missing, Float64}};
        distribution::Distribution = Normal(),
        ignore_missing::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )::BayesianTTest
    
    model_X = handle_missing_values(X; ignore_missing = ignore_missing)
    model_Y = handle_missing_values(Y; ignore_missing = ignore_missing)

    model = t_test_model(model_X, model_Y, distribution; samples = samples)

    @info "Sampling...."
    
    chain = sample(
        model,  
        NUTS(), 
        MCMCThreads(),
        round(Int, samples / chains), 
        chains)

    p = get_params(chain)
    preds = reduce(hcat, generated_quantities(model, chain))'

    mu_diff = vec(p.μ₂ .- p.μ₁)

    return BayesianTTest(
        mu_diff,
        median(mu_diff),
        bayes_factor(mu_diff, 0),
        preds[:,1],
        preds[:,2],
        chain,
        string(Meta.parse(string(distribution)))
    )

end

#### Method for 2-column Matrix

function t_test(
    X::AbstractMatrix{<:Union{Missing, Float64}};
    kwargs...)

    @info "Matrix passed to t_test(): Column 1 will be taken as X and Column 2 will be taken as Y."
    return t_test(X[:,1], X[:,2]; kwargs...)
end
