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

def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
#df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
#df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label    
"""

py"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

data_dir = '../input/'
df_seeds = pd.read_csv(data_dir + 'datafiles/NCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'datafiles/NCAATourneyCompactResults.csv')
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
