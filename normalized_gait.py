import pandas as pd
import plotly.graph_objs as go
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.sandbox.regression.predstd import wls_prediction_std


#Constants
################
csv_name = "Dataset_1_Ethan_01062023.csv"
joint = 'RightKnee'
pln = "Sagittal Plane Right"
system = "RL - RunLab"
################
df = pd.read_csv(csv_name, index_col=0, header=[0,1])

#Import the dataframe and slice it into gait cycles
def slice_df_gait_cycles(angle_dataframe: object, plane: str, leg_and_system: str):
    #local scope
    dff = angle_dataframe
    mask = dff.loc[:, ('Phase', leg_and_system)] == 'Initial Strike'

    #Drop the other planes (dff_plane)
    dff_p = dff.loc[:, dff.columns[dff.columns.get_level_values(0).isin([plane, 'Phase'])]]

    #Drop the other phase classifications (dff_classified)
    dff_c = dff_p.loc[
        :, dff_p.columns[
            ~dff_p.columns.get_level_values(0).isin(['Phase']) #keep all of the headers that aren't Phase
            ]
        ].join(dff_p[('Phase',leg_and_system)])#Select the proper label system only and add that back to the filtered dff

    cum_sum = mask.cumsum() #Create a boolean mask --also-- LOL

    gait_cycle_list = [g for _, g in dff_c.groupby(cum_sum)]

    return gait_cycle_list

gcl = slice_df_gait_cycles(df, pln, system)

#Add the "Percent to Completion" column to each DataFrame to 
# accomodate varied number of rows over the same cycle
def reindex_to_percent_complete(df_list: list):
    df_list = [df.reset_index(drop=True) for df in df_list]
    df_list = [df.assign(pct_complete = lambda x: x.index / len(x)) for df in df_list]
    
    #Concatenate the df's and set the pct_complete as index in order 
    # to align the data along the cycle
    df_concat = pd.concat(df_list)
    df_concat.sort_values(by='pct_complete', inplace=True)
    df_concat.set_index('pct_complete', inplace=True)
    df_concat.index.rename(None, inplace=True)

    return df_concat

f_df = reindex_to_percent_complete(gcl)

#Trace of all datapoints
trace = go.Scatter(
    name="Right Knee Data", 
    x=f_df.index, 
    y=f_df.loc[:, (pln, joint)], 
    mode='lines+markers')

'''
#Polynomial regression in order to model the general waveform of the data
#Fit a 2nd degree polynomial to the data (https://ostwalprasad.github.io/machine-learning/Polynomial-Regression-using-statsmodel.html)
'''
x = f_df.index.to_numpy()[:,np.newaxis] #Create a numpy array and add a new axis
index = x.ravel().argsort() #Sort and reindex the array
x = x[index].reshape(-1,1) #Reshape the array. -1 infers the first dimension of the new array based on the size of the original array

y = f_df.loc[:, (pln, joint)].to_numpy()[:,np.newaxis]
y = y[index]#Sort y according to x sorted index
y = y.reshape(-1, 1) #(n,1)

#Generate polynomial and interaction features in a matrix
polynomial_features= PolynomialFeatures(degree=5)
#Fit to data, the transform it
xp = polynomial_features.fit_transform(x)
print(xp.shape)

'''
Use ordinary least squares for regression
Polynomial regression for n degrees. y=b0 + b1x + b2x^2 ...+ bnx^n
'''
model = sm.OLS(y, xp).fit()
#run the regression
ypred = model.predict(xp)  #(n,) a flattened array
_, upper, lower = wls_prediction_std(model) # Calculate the confidence intervals

# Create scatter coordinates for best fit line
best_fit = go.Scatter(
    name='Trend', 
    x=f_df.index, 
    y=ypred, 
    mode='lines',
    line=dict(color='blue', width=2)
)
#Set up the confidence intervals
upper_ci = go.Scatter(
    name='Upper Confidence', 
    x=f_df.index, 
    y=upper, 
    mode='lines', 
    line=dict(color='red', width=1)
)
lower_ci = go.Scatter(
    name='Lower Confidence', 
    x=f_df.index, 
    y=lower, 
    mode='lines', 
    line=dict(color='red', width=1)
)

# Display the graph
fig = go.Figure(data=[trace, best_fit, upper_ci, lower_ci])
fig.update_layout(
    title="Polynomial Fit of Right Knee w CI", 
    xaxis_title='Percent to Completion', 
    yaxis_title='Angle')
fig.show()

'''You can also use other curve fitting libraries like scipy.optimize for more complex and accurate curve fitting.'''
