import pandas as pd 
import datetime as dt

import plotly.offline as pyo 
import plotly.graph_objs as go 
import plotly.express as px

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash import Dash, dash_table

import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.sandbox.regression.predstd import wls_prediction_std


app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.LUX],
    #scale viewport for mobile devices 
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
) # https://bootswatch.com/lux/
#server = app.server

HEADER = [
    html.H1('Movement Report'), 
    html.H3('A web application for building a movement analysis report.'), 
    html.Hr()]

graph_options = dbc.Card(
    [
        html.Div(
            [
                html.H3("Options", style={"color": "#ffffff"}),
                html.Hr(),
                dbc.Label("Body Plane:", style={"color": "#ffffff"}),
                dcc.Dropdown(
                    id='plane-radio',
                    options=
                    ['Anterior Frontal', 'Sagittal Right', 'Sagittal Left', 'Posterior Frontal'],
                    value='Sagittal Right',
                    clearable=False,
                    style={'color': 'black'}
                ),
                dbc.Label("Joint Selection:", style={"color": "#ffffff"}),
                dcc.Dropdown(
                    id='joint-radio',
                    options=
                    ['RightArm', 'RightHip', 'RightKnee', 'RightAnkle', 'RightToe'],
                    value='RightKnee',
                    clearable=False,
                    style={'color': 'black'}
                ),
                dbc.Label("Phase Highlight:", style={"color": "#ffffff"}),
                dcc.Dropdown(
                    id='phase-highlight',
                    options=
                    ['None', 'Initial Strike', 'Loading Response', 'Midstance', 'Terminal Stance', 'Toe Off', 
                    'Initial Swing', 'Midswing', 'Terminal Swing'],
                    value='None',
                    clearable=False,
                    style={'color': 'primary'}
                )
            ],
        )
    ],
    color="primary",
    style={'height': '500px'},
    body=True
)

####################################
#Temp Table Info
####################################
def slice_df_gait_cycles(angle_dataframe: object, plane: str, leg_and_system: str):
    #local scope
    dff = angle_dataframe

    #Drop the other planes (dff_plane)
    dff_p = dff.loc[:, dff.columns[dff.columns.get_level_values(0).isin([plane, 'Phase'])]]

    #Drop the other phase classifications (dff_classified)
    dff_c = dff_p.loc[
        :, dff_p.columns[
            ~dff_p.columns.get_level_values(0).isin(['Phase']) #keep all of the headers that aren't Phase
            ]
        ].join(dff_p[('Phase',leg_and_system)])#Select the proper label system only and add that back to the filtered dff

    #groups = df.groupby(('Phase','RL - RunLab'))

    group_list = [g for _, g in dff_c.groupby(('Phase','RL - RunLab'))]

    return group_list

#Constants for graph function
################
csv_name = "Dataset_1_Ethan_01062023.csv"
#joint = 'RightKnee'
pln = "Sagittal Plane Right"
system = "RL - RunLab"
################
df = pd.read_csv(csv_name, index_col=0, header=[0,1])
gl = slice_df_gait_cycles(df, pln, system)

#Create a new dataframe containing mean, max, and min
def calculate_mean_min_max(df_list: list):
    #df_list = [df.reset_index(drop=True) for df in df_list]
    results_list = []

    #phases = df_list[0]["Phase"].unique()
    # Iterate through the list of DataFrames
    for i, df in enumerate(df_list):
        phase = df['Phase']
        df = df.drop(columns=['Phase'], level=0)
        df.columns = df.columns.get_level_values(1)

        # Calculate the mean, min, and max for each column
        mean_df = df.mean().round(2)
        min_val = df.min().round(2)
        max_val = df.max().round(2)
        
        # Concatenate dataframes and rename columns
        phase_header = str(phase.iloc[1,0])
        #print(phase_header)
        result = pd.concat([mean_df, min_val, max_val], axis=1)
        level_1 = ['mean','min','max']
        result.columns = level_1
        result = pd.concat([result], keys=[phase_header], axis=1)
        #print(result)
        #print(result)
        # Append the result to the list
        results_list.append(result)

# Concatenate the results into a single DataFrame
    result_df = pd.concat(results_list, names=[result], axis=1)
    #print(result_df.head())
    return result_df

f_df = calculate_mean_min_max(gl)

def datatable_settings_multiindex(df, flatten_char = '_'):
    ''' Plotly dash datatables do not natively handle multiindex dataframes. This function takes a multiindex column set
    and generates a flattend column name list for the dataframe, while also structuring the table dictionary to represent the
    columns in their original multi-level format.  
    
    Function returns the variables datatable_col_list, datatable_data for the columns and data parameters of 
    the dash_table.DataTable'''
    datatable_col_list = []
        
    levels = df.columns.nlevels
    if levels == 1:
        for i in df.columns:
            datatable_col_list.append({"name": i, "id": i})
    else:        
        columns_list = []
        for i in df.columns:
            col_id = flatten_char.join(i)
            datatable_col_list.append({"name": i, "id": col_id})
            columns_list.append(col_id)
        df.columns = columns_list

    datatable_data = df.to_dict('records')
    
    return datatable_col_list, datatable_data

columns, data = datatable_settings_multiindex(f_df)

#Limit the width of the elements in this page to the standard size of a A4 page
app.layout = html.Div(
    [
        dbc.Container(
            children=[
                #Header
                html.Div(
                    children=HEADER
                ),
                dbc.Row(
                    #Here, I need to add in the columns if I want them to be printed
                    [  #May need a card group here if I want to control the height
                        dbc.Col(graph_options, width=3, style={'height': '100%'}),
                        dbc.Col([
                            dcc.Graph(id='graph'), 
                            dash_table.DataTable(
                                id='table',
                                data=data, 
                                columns=columns
                            ),
                            
                        ], width=8)
                    ],
                    align="start"
                ),
            ], 
            id='print', 
            style={"maxWidth": "900px"} #For A4 style .pdf files, 900px seems to be the max
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Button(children=['Download'],className="mr-1",id='js',n_clicks=0),
                    ],
                    align="start"
                )
            ],
            style={"maxWidth": "130px"}
        ),
    ]
)

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input(component_id='joint-radio', component_property='value'),
)
def update_fig(joint_radio):
    joint = joint_radio
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
    #print(xp.shape)

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
    #fig.show()

    return fig

app.clientside_callback(
    """
    function(n_clicks){
        if(n_clicks > 0){
            var opt = {
                margin: 0,
                filename: 'report.pdf',
                pagebreak: { mode: ['avoid-all'] },
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 1 },
                jsPDF: { orientation: 'p', unit: 'cm', format: 'a4', precision: 8 }
            };
            //the "print" is is used to call the entire layout.
            html2pdf().from(document.getElementById("print")).set(opt).save();
        }
    }
    """,
    Output('js','n_clicks'),
    Input('js','n_clicks')
)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)

