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

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

#Limit the width of the elements in this page to the standard size of a A4 page
app.layout = html.Div([
    html.Div([
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),
        
        dcc.Graph(
            id='example-graph',
            figure=fig
        )
        ],
        id='print'),

dbc.Button(children=['Download'],className="mr-1",id='js',n_clicks=0),
],id='main',)

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
    app.run_server(port='8050',debug=True)