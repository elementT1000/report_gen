import pandas as pd 
import datetime as dt

import plotly.offline as pyo 
import plotly.graph_objs as go 
import plotly.express as px

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State


app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.LUX],
    #scale viewport for mobile devices 
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
) # https://bootswatch.com/lux/
#server = app.server

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

#################################################################################
'''controls = [
    dbc.Select(
        id="scene",
        options=[{"label": "video"}],
        style={"maxWidth": "500px"}
    )
]

video_card = dbc.Card(
    [
        html.Div(
            id='video display',
            children=[
                dbc.CardBody(
                    dash_player.DashPlayer(
                        # width indicates the percentage of the card taken
                        id="video", width="100%", height="auto", controls=True
                    )

                )
            ]
        )
    ],
    color=colors['background'],
    style={"maxWidth": "1280px"},
    outline=True
)

video_options = dbc.Card(
    [
        html.Div(
            [
                html.H3('Video Upload'),
                html.Hr(),
                dcc.Store(id='video_name_store'),
                du.Upload(
                    id='upload-video', max_file_size=4000  # 4000MB
                ),
                dbc.Label("Set AI Settings:", style={'fontSize': '24px', 'textAlign': 'center'}),
                # We probably need a dropdown for walking or running as well
                dcc.Dropdown(
                    id="plane-setting",
                    options=['Sagittal Plane', 'Frontal Plane'],
                    value='Sagittal Plane',
                    clearable=False,
                    style={
                        'color': 'black',
                    }
                ),
                dcc.Checklist(
                    id="checklist",
                    options=[
                        {"label": "Find Angles", "value": "FA"},
                        {"label": "Find Phase", "value": "FP"}
                    ],
                    value=[],
                    inline=False,
                    inputStyle={"margin-right": "20px", "margin-left": "10px"},
                    style={
                        "fontSize": "18px",
                        'color': colors['text'],

                    }

                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button("process-button", outline=True, color='warning', disabled=True),
                        dbc.Button("download-button", outline=True, color='warning', disabled=True)
                    ],
                    class_name="d-flex justify-content-center"
                )
            ],

        )
    ],
    # This pushes the words out of the box when the screen is small
    # body=True
)


app.layout = dbc.Container(
    id="app-container",
    style={'backgroundColor': colors['background']},
    children=[
        # Banner Display
        html.Div(
            className="header",
            children=[
                html.H2("T4 Movement Analyses", id="title", style={'textAlign': 'left', 'color': colors['text']}),
                html.Hr(),
            ],
        ),
        # row 1
        dbc.Row(
            [
                dbc.Col(video_card, width=9),
                dbc.Col(video_options, width=3)
            ],
            align="start",
        ),
        # row 2
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='graph'), width=9),
                dbc.Col(graph_options, width=3, style={'height': '100%'})
            ],
            align="start"
        ),
        # extra row
        dbc.Row(
            [
                dbc.Col(dbc.Card(dbc.Row([dbc.Col(c) for c in controls]), body=True), width=3)
            ]
        )
    ],
    fluid=True
)
'''
#################################################################################
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
                dbc.Label("Joint Selection:", style={"color": "#ffffff"}),
                dcc.Dropdown(
                    id='joint-radio',
                    options=
                    ['All', 'Near Hip Angle', 'Near Knee Flexion', 'Near Ankle Angle'],
                    value='Near Hip Angle',
                    clearable=False,
                    style={'color': 'black'}
                ),
                dbc.Label("Phase Highlight:", style={"color": "#ffffff"}),
                dcc.Dropdown(
                    id='phase-highlight',
                    options=
                    ['None', 'Initial Strike', 'Loading Response', 'Midstance',
                     'Terminal Stance', 'Initial Swing', 'Midswing', 'Terminal Swing'],
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
                        dbc.Col(dcc.Graph(id='example-graph',figure=fig), width=8)
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