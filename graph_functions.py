import pandas as pd
import plotly.graph_objs as go
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from gait_slicer import *


def filter_pln_n_joint(angle_dataframe: object, plane: str, leg_and_system: str):
    #local scope
    dff = angle_dataframe

    #Drop the other planes (dff_plane)
    dff_p = dff.loc[:, dff.columns[dff.columns.get_level_values(0).isin([plane, 'Phase'])]]

    #Drop the other phase classifications (dff_classified)
    dff_c = dff_p.loc[
        :, dff_p.columns[
            ~dff_p.columns.get_level_values(0).isin(['Phase']) #keep all of the headers that aren't Phase
            ]
        ].join(dff_p[('Phase',leg_and_system)])#Select the proper label system only and add that back to the filtered df

    return dff_c

def og_phase(dff, key=None):
    if not key:
        pass
    else:
        indexed_phase = dff['Phase'].values #values returns a numpy array without the headers
        
        x_val = []
        for idx, elem in enumerate(indexed_phase):
            thiselem = indexed_phase[idx]
            nextelem = indexed_phase[(idx + 1) % len(indexed_phase)]
            if thiselem == key and nextelem != key:
                x_val.append(idx)
            else:
                continue

    return x_val
 
if __name__ == "__main__":
    #Constants
    ################
    csv_name = "Dataset_1_Ethan_01062023.csv"
    joint = 'RightKnee'
    pln = "Sagittal Plane Right"
    system = "RL - RunLab"
    ################
    df = pd.read_csv(csv_name, index_col=0, header=[0,1])

    f_df = filter_pln_n_joint(df, pln, system)
    #print(f_df)
    median = og_phase(f_df, key="Toe Off")
    print(median)
    

    #Trace of all datapoints
    trace = go.Scatter(
        name="Right Knee Data", 
        x=f_df.index, 
        y=f_df.loc[:, (pln, joint)], 
        mode='lines+markers')

    
    # Display the graph
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title="Right Knee Angle", 
        xaxis_title='Index', 
        yaxis_title='Angle')
    #Phase highlight line
    for x in median:
        fig.add_vline(x=x, line_width=2, line_dash="dash", line_color='red')
    fig.show()
