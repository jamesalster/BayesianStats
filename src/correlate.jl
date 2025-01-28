
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
    posterior_sample::Chains
    distribution::String
end

#### Pretty print for that

function Base.show(io::IO, ::MIME"text/plain", b::BayesianCorrelation)
    println("""
    Bayesian Correlation:

    Model distribution: $(b.distribution)
    Fitted with $(prod(size(b.posterior_sample)[[1,3]])) posterior draws

    Estimated ρ: $(round(b.point_estimate, digits = 2))
    95% Interval: $(round.(quantile(b.statistic, [0.025, 0.975]), digits = 2))

    Fields: 
        statistic: the distribution of the correlation coefficient ρ
        x_est and y_est: predicted population values for x and y
        posterior_sample: the Turing sample() object
    """
    )
end


#### Model template

@model function correlation_model(
        X::Matrix{Float64},
        distribution::Symbol,
        df::Int64
    )

    N, D = size(X)  # D should be 2
    
    # Calculate basic statistics
    X_means = vec(mean(X, dims=1))
    X_stds = vec(std(X, dims=1))
    
    # Priors for means and standard deviations
    μ ~ MvNormal(X_means, X_stds)
    σ ~ filldist(Exponential(0.5), D)
    
    #A bit awkward this prior but the distribution does look appropriate and it's super fast
    # and claude thinks it's great because it's easy to compute the gradient
    z ~ Normal(0, 0.5)  # z-score on Fisher transform
    ρ = clamp(tanh(z), -0.999, 0.999)  # transforms back to (-1,1)

    Σ = [σ[1]^2       ρ*σ[1]*σ[2];
        ρ*σ[1]*σ[2]  σ[2]^2]
    
    # Single multivariate normal for all observations
    if distribution === :MvNormal
        model_distribution = MvNormal(μ, Σ)
    elseif distribution === :MvTDist
        model_distribution = MvTDist(df, μ, Σ)
    end

    for i in 1:N
        X[i,:] ~ model_distribution
    end

    if distribution === :MvNormal
        X_pred = rand(MvNormal(μ, Σ), N)'
    elseif distribution === :MvTDist
        X_pred = rand(MvTDist(df, μ, Σ), N)'
    end
    
    return (ρ=ρ, X_pred=X_pred)
end

#### Function to calculate correlation

function correlate(
        X::AbstractMatrix{Float64};
        distribution::Symbol = :MvTDist, #pass as string
        ignore_missing::Bool = false,
        samples::Int = 4000,
        chains::Int = 4,
        t_dist_df::Int = 2 #for Student Only
        )

    if size(X, 2) != 2
        error("Must pass correlate() a 2-column matrix")
    end

    X_model = handle_missing_values(X; ignore_missing = ignore_missing)

    if ! in(distribution, [:MvNormal, :MvTDist])
        error("Distribution must be one of :MvNormal or :MvTDist")
    end

    model = correlation_model(X_model, distribution, t_dist_df)

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
function correlate(
    x::AbstractVector{<:Union{Missing, Float64}},
    y::AbstractVector{<:Union{Missing, Float64}};
    kwargs...)

    if length(x) != length(y)
        error("x and y must be the same length")
    end

    return correlate([x y]; kwargs...)
end