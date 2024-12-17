
module BayesianStats

#Convenience functions
include("utils.jl")

#include("correlate.jl")
include("t_test.jl")

export t_test, bayes_factor

end