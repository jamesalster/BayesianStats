
module BayesianStats

#Convenience functions
include("utils.jl")

#Main methods
include("t_test.jl")
include("correlate.jl")

export bayes_factor
export t_test
export correlate

