using PyCall
# using Pkg; Pkg.add("Conda")
using Conda

# How to import functions from Python in Julia
numpy_sum = pyimport("numpy")["sum"]
# Example usage
numpy_sum(a)

# Writing a hand-written function in Python
py"""
def py_sum(A):
    s = 0.0
    for a in A:
        s += a
    return s
"""

sum_py = py"py_sum"

# How to use Scikit learn functions

using ScikitLearn

# This model requires scikit-learn. See
# http://scikitlearnjl.readthedocs.io/en/latest/models/#installation
@sk_import linear_model: LogisticRegression
model = LogisticRegression(fit_intercept=true)
fit!(model, X, y)

accuracy = sum(predict(model, X) .== y) / length(y)
println("accuracy: $accuracy")
