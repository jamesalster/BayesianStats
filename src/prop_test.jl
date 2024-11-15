
using Turing
using Distributions

#### Output Struct

struct BayesianPropTest
    statistic::Vector{Float64}
    point_estimate::Float64
    bayes_factor::Float64
    p_value::Float64
    posterior_sample::Chains
    distribution::String
end

#### Pretty print for that

function Base.show(io::IO, ::MIME"text/plain", b::BayesianPropTest)
    println("""
    Bayesian Proportions Test:

    Prior Distribution for the probability of success: $(b.distribution)
    Fitted with $(prod(size(b.posterior_sample)[[1,3]])) posterior draws

    Estimated difference in probabilities (y - x): $(round(b.point_estimate, digits = 3))
    95% Interval: $(round.(quantile(b.statistic, [0.025, 0.975]), digits = 3))

    Bayes Factor that difference in probabilities (y - x) > 0: $(round(b.bayes_factor, digits = 3))
    P-value that difference in probabilities (y - x) > 0: $(round(b.p_value, digits = 3)) 

    Fields: 
        statistic: the distribution of the difference in probabilities
        point_estimate: the median of that distribution
        bayes_factor: the bayes factor that difference in probabilities > 0
        p_value: the probability that the difference in probabilities > 0
        posterior_sample: the Turing sample() object
        distribution: the *prior* distribution of the binomial probability p used in the test
    """
    )
end

#### Model template

@model function prop_test_model(
        successes::Matrix{Int},
        trials::Matrix{Int},
        prior_distribution::Distribution = Beta(0.5, 0.5)
    )

    N = size(successes, 1) 
    
    # Priors for probabilities 
    p₁ ~ prior_distribution
    p₂ ~ prior_distribution

    pred_successes = Matrix{Int64}(undef, N, 2)
    # Likelihood
    for i in 1:N
        successes[i,1] ~ Binomial(trials[i,1], p₁)
        successes[i,2] ~ Binomial(trials[i,2], p₂) 
    end
end

#### Function to calculate prop_test - this is exported

function prop_test(
        X::AbstractMatrix{<:Union{Missing, Real}},
        Y::AbstractMatrix{<:Union{Missing, Real}};
        prior_distribution::Distribution = Beta(0.5, 0.5),
        ignore_missing::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )::BayesianPropTest
    
    if size(X) != size(Y)
        error("X and Y must be two column matrices, containing a column of successes and trials respectively, of the same size.")
    elseif size(X, 2) != 2 || size(Y, 2) != 2
        error("Must pass in a two column matrix for X and Y.")
    end

    model_X = handle_missing_values(X; ignore_missing = ignore_missing)
    model_Y = handle_missing_values(Y; ignore_missing = ignore_missing)

    model_X = Int64.(model_X)
    model_Y = Int64.(model_Y)

    if any(model_X .< 0) || any(model_Y .< 0)
        error("All input values must be positive.")
    end

    successes = hcat(model_X[:,1], model_Y[:,1])
    trials = hcat(model_X[:,2], model_Y[:,2])

    model = prop_test_model(
        successes,
        trials,
        prior_distribution
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
    p_diff = vec(p.p₂ .- p.p₁)

    return BayesianPropTest(
        p_diff,
        median(p_diff),
        bayes_factor(p_diff, 0),
        p_value(p_diff, 0),
        chain,
        string(Meta.parse(string(prior_distribution)))
    )
end

#### Method for x and y

function prop_test(
    x_successes::AbstractVector{<:Union{Missing, Int}},
    x_trials::AbstractVector{<:Union{Missing, Int}},
    y_successes::AbstractVector{<:Union{Missing, Int}},
    y_trials::AbstractVector{<:Union{Missing, Int}};
    kwargs...)::BayesianPropTest

    ## Check size
    sizes = map(size, [x_successes, x_trials, y_successes, y_trials])
    if ! allequal(sizes)
        error("The sizes of all vectors must be the same.")
    end

    return prop_test(hcat(x_successes, x_trials), hcat(y_successes, y_trials); kwargs...)
end

### Method for integers
function prop_test(
    x_successes::Int,
    x_trials::Int,
    y_successes::Int,
    y_trials::Int;
    kwargs...)::BayesianPropTest

    return prop_test([x_successes x_trials], [y_successes y_trials]; kwargs...)
end
