
module BayesianStats

#Convenience functions
include("utils.jl")

include("correlate.jl")
include("t_test.jl")

export correlate, t_test, bayes_factor, p_value

end