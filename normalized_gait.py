import pandas as pd
import plotly.graph_objs as go
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.sandbox.regression.predstd import wls_prediction_std


#Dummy DataFrames
df1 = pd.DataFrame({"Series": [2, 3, 4, 5, 4, 3, 1],})
df2 = pd.DataFrame({"Series": [1, 2, 3, 4, 5, 4, 3, 2, 1]})
df3 = pd.DataFrame({"Series": [1, 2, 3, 4, 4, 4, 3, 2]})

#Add the "Percent to Completion" column to each DataFrame to 
# accomodate varied number of rows over the same cycle
df1 = df1.assign(pct_complete = lambda x: x.index / len(x))
df2 = df2.assign(pct_complete = lambda x: x.index / len(x))
df3 = df3.assign(pct_complete = lambda x: x.index / len(x))

#Concatenate the df's and set the pct_complete as index in order 
# to align the data along the cycle
df = pd.concat([df1, df2, df3], axis=0)
df.sort_values(by='pct_complete', inplace=True)
df.set_index('pct_complete', inplace=True) 
df.index.rename(None, inplace=True)

#Trace of all data
trace = go.Scatter(
    name="Data", 
    x=df.index, 
    y=df['Series'], 
    mode='lines+markers')

'''
#Polynomial regression in order to model the general waveform of the data
#Fit a 2nd degree polynomial to the data (https://ostwalprasad.github.io/machine-learning/Polynomial-Regression-using-statsmodel.html)
'''
x=df.index.to_numpy()[:,np.newaxis] #Create a numpy array and add a new axis
index = x.ravel().argsort() #Sort and reindex the array
x = x[index].reshape(-1,1) #Reshape the array. -1 infers the first dimension of the new array based on the size of the original array

y=df['Series'].to_numpy()[:,np.newaxis]
y = y[index]#Sort y according to x sorted index
y = y.reshape(-1, 1) #(n,1)

#Generate polynomial and interaction features in a matrix
polynomial_features= PolynomialFeatures(degree=3)
#Fit to data, the transform it
xp = polynomial_features.fit_transform(x)
print(xp.shape)

'''
Use ordinary least squares for regression
Polynomial regression for 3 degrees. y=b0 + b1x + b2x^2 + b3x^3
'''
model = sm.OLS(y, xp).fit()
#run the regression
ypred = model.predict(xp)  #(n,) a flattened array
_, upper, lower = wls_prediction_std(model) # Calculate the confidence intervals
# Create scatter coordinates for best fit line
best_fit = go.Scatter(
    name='Trend', 
    x=df.index, 
    y=ypred, 
    mode='lines',
    line=dict(color='blue', width=2)
)

#Set up the confidence intervals
upper_ci = go.Scatter(
    name='Upper Confidence', 
    x=df.index, 
    y=upper, 
    mode='lines', 
    line=dict(color='red', width=1)
)
lower_ci = go.Scatter(
    name='Lower Confidence', 
    x=df.index, 
    y=lower, 
    mode='lines', 
    line=dict(color='red', width=1)
)

# Display the graph
fig = go.Figure(data=[trace, best_fit, upper_ci, lower_ci])
fig.update_layout(title="Polynomial Fit of Mixed Series w CI", xaxis_title='Percent to Completion', yaxis_title='Normalized Mean')
fig.show()

'''You can also use other curve fitting libraries like scipy.optimize for more complex and accurate curve fitting.'''
