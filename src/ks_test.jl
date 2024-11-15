
using Distributions

#### Output Struct

struct BayesianKSTest
    statistic::Vector{Float64}
    point_estimate::Float64
    bayes_factor::Float64
    p_value::Float64
    samples::Int
    booststrap_ks::Vector{Float64}
    observed_ks::Float64
end

#### Pretty print for that

function Base.show(io::IO, b::BayesianKSTest)
     print(io, "BayesiansKSTest")
end

function Base.show(io::IO, ::MIME"text/plain", b::BayesianKSTest)
    println("""
    Bayesian KS Test:

    Method: Booststrap with Dirichlet Weights
    Fitted with $(b.samples) bootstrap samples

    Bayes Factor that y and x come from a different distribution: $(round(b.bayes_factor, digits = 3))
    Probability that y and x come from a different distribution: $(round(b.p_value, digits = 3)) 

    Fields: 
        statistic: the distribution of the difference KS statistics (bootstrap - observed)
        point_estimate: the median of that distribution
        bayes_factor: the bayes factor that difference in KS statistics > 0
        p_value: the probability that the difference in KS statistics > 0
        samples: the number of samples used in the test
        booststrap_ks: the booststrap ks values
        observed_ks: the observed ks value
    """
    )
end

#### Test function

function dirichlet_bootstrap_ks(
        X::Vector{Float64}, 
        Y::Vector{Float64}, 
        samples::Int
    )

    n1, n2 = length(X), length(Y)
    posterior_d = zeros(samples)

    # Sort combined samples for CDF calculation
    x = sort(vcat(X, Y))

    for i in 1:samples
        # Generate Dirichlet weights
        w1 = rand(Dirichlet(ones(n1)))
        w2 = rand(Dirichlet(ones(n2)))
        
        # Calculate weighted CDFs
        cdf1 = zeros(length(x))
        cdf2 = zeros(length(x))
        
        for (j, xi) in enumerate(x)
            cdf1[j] = sum(w1[X .<= xi])
            cdf2[j] = sum(w2[Y .<= xi])
        end
        
        # Calculate KS statistic
        posterior_d[i] = maximum(abs.(cdf1 - cdf2))
    end
    
    return posterior_d
end

#### Function to calculate t_test - this is exported

function ks_test(
        X::AbstractVector{<:Union{Missing, Real}},
        Y::AbstractVector{<:Union{Missing, Real}};
        ignore_missing::Bool = false,
        samples::Int = 4000,
        )::BayesianKSTest
    
    model_X = handle_missing_values(X; ignore_missing = ignore_missing)
    model_Y = handle_missing_values(Y; ignore_missing = ignore_missing)

    model_X = Float64.(model_X)
    model_Y = Float64.(model_Y)

    #Calculate booststrap ks statistic
    booststrap_ks = dirichlet_bootstrap_ks(
        model_X,
        model_Y,
        samples
    )

    # Calculate observed KS statistic
    sorted_combined = sort(vcat(model_X, model_Y))
    n1, n2 = length(model_X), length(model_Y)
    
    cdf1 = [count(x .<= xi for x in model_X) / n1 for xi in sorted_combined]
    cdf2 = [count(x .<= xi for x in model_Y) / n2 for xi in sorted_combined]
    observed_ks = maximum(abs.(cdf1 - cdf2))

    ks_diff = booststrap_ks .- observed_ks

    return BayesianKSTest(
        ks_diff,
        median(ks_diff),
        bayes_factor(ks_diff, 0),
        p_value(ks_diff, 0),
        samples,
        booststrap_ks,
        observed_ks
    )
end

#### Method for matrix

function ks_test(X::AbstractMatrix{<:Union{Missing, Real}}; kwargs...)::BayesianKSTest
    return ks_test(X[:,1], X[:,2]; kwargs...)
end
