import pandas as pd
import plotly.graph_objs as go
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from gait_slicer import *


def highlight_phase_median(dff, key=None):
    '''
    Algorithm:
    1. Filter the dataframe to just the phase and index
    2. Add a column with the 'Percent Complete' index in order to calculate the median
    3. Calculate the median of the for each phase.
    4. Retrieve a value for the selected phase
    '''
    indexed_phase = dff['Phase']
    indexed_phase.insert(0, 'Percent Complete', indexed_phase.index, True)
    indexed_phase = indexed_phase.reset_index(drop=True)

    #Creates a list of median values, just select the one you want, then print that
    median = indexed_phase.groupby(indexed_phase.columns[-1]).median()
    #print(median)
    
    if not key:
        pass
    else:
        phase = median[median.index == key]

        value = phase['Percent Complete'].values

    return value

if __name__ == "__main__":
    #Constants
    ################
    csv_name = "Dataset_1_Ethan_01062023.csv"
    joint = 'RightToe'
    pln = "Sagittal Plane Right"
    system = "RL - RunLab"
    ################
    df = pd.read_csv(csv_name, index_col=0, header=[0,1])

    gcl = slice_df_gait_cycles(df, pln, system)
    f_df = reindex_to_percent_complete(gcl)
    median = highlight_phase_median(f_df, key='Toe Off')
    print(median)

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
    ##Experimental
    y = np.nan_to_num(y, 0)

    #Generate polynomial and interaction features in a matrix
    polynomial_features= PolynomialFeatures(degree=5)
    #Fit to data, the transform it
    xp = polynomial_features.fit_transform(x)
    print(xp)

    '''
    #Use ordinary least squares for regression
    #Polynomial regression for n degrees. y=b0 + b1x + b2x^2 ...+ bnx^n
    '''
    #statsmodels.api.OLS(y, xp).fit().predict() returns all nan values if any nan values are present
    model = sm.OLS(y, xp).fit()
    #run the regression
    ypred = model.predict(xp)  #(n,) a flattened array
    print(ypred)
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
    #Phase highlight line
    for x in median:
        fig.add_vline(x=x, line_width=2, line_dash="dash", line_color='red')
    fig.show()


