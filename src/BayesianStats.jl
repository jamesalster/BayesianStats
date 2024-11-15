
module BayesianStats

#Convenience functions
include("utils.jl")

#include("correlate.jl")
include("t_test.jl")
include("prop_test.jl")
include("ks_test.jl")

export correlate, t_test, prop_test, ks_test
export bayes_factor, p_value

end