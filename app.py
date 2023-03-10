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

from gait_slicer import *
from table_maker import *
from data_modifications import process_df
from normalized_gait import og_phase, filter_pln_n_joint

import base64
import io

from dash.exceptions import PreventUpdate


####################################
#Temp Table Info
####################################
#TODO: Initialize the page without any data
csv_name = "Dataset_1_Ethan_01062023.csv"
#TODO: The current 'slice_df_into_phases()' uses the same system despite the leg. I need to set up a dictionary to switch the system based on the leg selected.
#For the af and pf planes, I need a toggle to select which leg. For the sagittal planes, I just need to switch it based on which side is collected. 
#This is important for the tables
system = "RL - RunLab"
################


#Dash Application Starts
app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.LUX],
    #scale viewport for mobile devices 
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
) # https://bootswatch.com/lux/
#server = app.server

#User Interface
HEADER = [
    html.H1('Movement Report'), 
    html.H3('A web application for building a movement analysis report.'), 
    html.Hr()]

joint_options = {
    'Anterior Frontal Plane': ['afLeftThigh', 'afLeftKnee', 'afLeftAnkle', 'afLeftFoot', 'afRightThigh', 'afRightKnee', 'afRightAnkle', 'afRightFoot'],
    'Sagittal Plane Right': ['RightArm', 'RightHip', 'RightKnee', 'RightAnkle', 'RightToe'],
    'Sagittal Plane Left': ['LeftArm', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'],
    'Posterior Frontal Plane': ['pfWaist', 'pfLeftFemurHead', 'pfLeftKnee', 'pfLeftAnkle', 'pfRightFemurHead', 'pfRightKnee', 'pfRightAnkle'],
}

OPTIONS = dbc.Card(
    [
        html.Div(
            [
                html.H3("Options", style={"color": "#ffffff"}),
                html.Hr(),

                dcc.Upload(
                    id='upload-data',
                    children=['Upload Movement Data'],
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '0px'
                    },
                ),
                dcc.Store(id='store'),
                html.Div(id='csv-name', style={"color": "#ffffff"}),

                dbc.Label("Body Plane:", style={"color": "#ffffff"}),
                dcc.Dropdown(
                    id='plane-radio',
                    options=
                    ['Anterior Frontal Plane', 'Sagittal Plane Right', 'Sagittal Plane Left', 'Posterior Frontal Plane'],
                    value='Sagittal Plane Right',
                    clearable=False,
                    style={
                        'color': 'black',
                        'margin': '0px'
                    }
                ),

                dbc.Label("Joint Selection:", style={"color": "#ffffff"}),
                dcc.Dropdown(
                    id='joint-radio',
                    #options= ['RightArm', 'RightHip', 'RightKnee', 'RightAnkle', 'RightToe'],
                    #value='RightKnee',
                    clearable=False,
                    style={
                        'color': 'black',
                        'margin': '0px'
                    }
                ),

                dbc.Label("Phase Highlight:", style={"color": "#ffffff"}),
                dcc.Dropdown(
                    id='phase-highlight',
                    options=
                    ['None', 'Initial Strike', 'Loading Response', 'Midstance', 'Terminal Stance', 'Toe Off', 
                    'Initial Swing', 'Midswing', 'Terminal Swing'],
                    value='None',
                    clearable=False,
                    style={
                        'color': 'primary',
                        'margin': '0px'
                    }
                )
            ],
        )
    ],
    color="primary",
    style={'height': '500px'},
    body=True
)

TABLES = dbc.Container([
    html.Hr(),
    html.H3("Stance: Mean (Minimum/Maximum) for Each Joint"),
    dash_table.DataTable(
        id='stance-table',
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto', #Adds wrapping to cells
        },
    ), 
    html.H3("Swing: Mean (Minimum/Maximum) for Each Joint"),
    dash_table.DataTable(
        id='swing-table',
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto', #Adds wrapping to cells
        },
    ), 
])

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
                        dbc.Col(OPTIONS, width=3, style={'height': '100%'}),
                        dbc.Col([
                            dcc.Graph(id='graph'),  
                        ], width=9)
                    ],
                    align="start"
                ),
                dbc.Row(
                    [
                        TABLES
                    ],
                    align="start"

                ),
                dbc.Row(
                    dcc.Textarea(
                        id='textarea',
                        value='Add your notes here, Doc.',
                        style={'width': '100%', 'height': 300},
                    ),
                ),
            ], 
            
            style={"maxWidth": "1300px"} #For A4 style .pdf files, 900px seems to be the max
        ),
        #The button to download goes here if you need it
        
    ],
    id='print', 
)

#Functionality
#This successfully returns the .csv name, so it looks like it's uploaded b/c store doesnt't throw an error
@app.callback(
    Output(component_id='csv-name', component_property='children'),
    Output('store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def store_data(contents, filename):
    if contents is None:
        raise PreventUpdate

    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    return filename, df.to_json(date_format='iso', orient='split')

@app.callback(
    Output(component_id='stance-table', component_property='data'),
    Output(component_id='stance-table', component_property='columns'),
    Output(component_id='swing-table', component_property='data'),
    Output(component_id='swing-table', component_property='columns'),
    Input(component_id='plane-radio', component_property='value'),
    State('store', 'data'),
)
def update_table(plane_radio, upload):
    plane = plane_radio

    if upload is not None:
        df = process_df(upload)
    else:
        df = pd.read_csv(csv_name, index_col=0, header=[0,1])
        #print("The datatype of the old df is: ")
        #print(df.iloc[0, 0].dtype)
    #TABLE Operations
    #group list
    gl = slice_df_into_phases(df, plane, system)
    #calculated dataframe
    c_df = calculate_mean_min_max(gl)
    c_df.insert(0, 'Joint Vertex', c_df.index, True)
    c_df = c_df.reset_index(drop=True)

    stance_df = gait_section_slicer(c_df, stance=1)
    stance_columns, stance_data = datatable_settings_multiindex(stance_df)

    swing_df = gait_section_slicer(c_df, stance=0)
    swing_columns, swing_data = datatable_settings_multiindex(swing_df)

    return stance_data, stance_columns, swing_data, swing_columns

@app.callback(
    Output(component_id='joint-radio', component_property='options'),
    Input(component_id='plane-radio', component_property='value'),
)
def set_joint_options(selected_plane):
    return [{'label': i, 'value': i} for i in joint_options[selected_plane]]

@app.callback(
    Output('joint-radio', 'value'),
    Input('joint-radio', 'options'))
def set_joint_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input(component_id='joint-radio', component_property='value'),
    Input(component_id='plane-radio', component_property='value'),
    Input(component_id='phase-highlight', component_property='value'),
    State('store', 'data'),
)
def update_fig(joint_radio, plane_radio, phase_highlight, upload):
    if upload is not None:
        df = process_df(upload)
    else:
        df = pd.read_csv(csv_name, index_col=0, header=[0,1])
    
    joint = joint_radio
    pln = plane_radio
    phase = phase_highlight
    
    #Inside df
    i_df = df
    f_df = filter_pln_n_joint(i_df, pln, system)
    median = og_phase(f_df, key=phase)
    #print(median)

    #Trace of all datapoints
    trace = go.Scatter(
        name="Data", 
        x=f_df.index, 
        y=f_df.loc[:, (pln, joint)], 
        mode='lines+markers')

    # Display the graph
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title="Angle during Gait Cycle", 
        xaxis_title='Frame', 
        yaxis_title='Angle',
        showlegend=False
    )
    #Phase highlight line
    for x in median:
        fig.add_vline(x=x, line_width=2, line_dash="dash", line_color='red')
    #fig.show()

    return fig

'''
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
        
app.clientside_callback(
    """
#    function(n_clicks){
#        if(n_clicks > 0){
#            var opt = {
#                margin: 0,
#                filename: 'report.pdf',
#                pagebreak: { mode: ['avoid-all'] },
#                image: { type: 'jpeg', quality: 0.98 },
#                html2canvas: { scale: 1 },
#                jsPDF: { orientation: 'p', unit: 'cm', format: 'a4', precision: 8 }
#            };
#            //the "print" is is used to call the entire layout.
#            html2pdf().from(document.getElementById("print")).set(opt).save();
#        }
#    }
    """,
    Output('js','n_clicks'),
    Input('js','n_clicks')
)'''

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)

