
module BayesianStats

#Convenience functions
include("utils.jl")

#Main methods
include("t_test.jl")
include("correlate.jl")

export bayes_factor, bootstrap
export t_test
export correlate

end
