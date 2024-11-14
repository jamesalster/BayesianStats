
using Turing
using Distributions

#### Output Struct

struct BayesianTTest
    statistic::Vector{Float64}
    point_estimate::Float64
    x_est::Vector{Float64}
    y_est::Vector{Float64}
    posterior_sample::Chains
    distribution::String
end

#### Pretty print for that

function Base.show(io::IO, ::MIME"text/plain", b::BayesianTTest)
    println("""
    Bayesian T Test:

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

@model function t_test_model(
        x::Vector{Float64}, 
        y::Vector{Float64},
        distribution_fun::Distribution)

    N = length(x) #checked to be equal to length(y)

    # Priors for location parameters
    μ₁ ~ Normal(mean(x), std(x) * 2)
    μ₂ ~ Normal(mean(y), std(y) * 2)
    
    # Priors for scale parameters
    σ₁ ~ Exponential( 1 / (std(x) * 2) )
    σ₂ ~ Exponential( 1 / (std(y) * 2) )

    x_pred = Vector{Float64}(undef, N)
    y_pred = Vector{Float64}(undef, N)
    
    # Likelihood
    for i in 1:N
        x[i] ~ distribution_fun * σ₁ + μ₁
        y[i] ~ distribution_fun * σ₂ + μ₂

    end

    x_pred = rand(distribution_fun * σ₁ + μ₁)
    y_pred = rand(distribution_fun * σ₂ + μ₂)

    return [x_pred; y_pred]
end

#### Function to calculate t_test - this is exported

function t_test(
        x::AbstractArray{Float64, 1},
        y::AbstractArray{Float64, 1};
        distribution::Distribution = Normal(),
        ignore_missing::Bool = false,
        point_estimate::Bool = false,
        samples::Int = 4000,
        chains::Int = 4
        )::Union{Float64, BayesianTTest}

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

    model = t_test_model(
        x_model,
        y_model,
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
    q = reduce(hcat, generated_quantities(model, chain))'

    mu_diff = p.μ₂ .- p.μ₁ 

    if point_estimate
        return median(mu_diff)
    else
        output = BayesianTTest(
            vec(mu_diff),
            median(mu_diff),
            q[:,1],
            q[:,2],
            chain,
            string(Meta.parse(string(distribution)))
        )
    end

    return output
end

