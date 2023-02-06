import pandas as pd
from dash import Dash, dash_table
from dash import dcc, html
import dash_bootstrap_components as dbc


#Import the dataframe and slice it into groups by phase
def slice_df_into_phases(angle_dataframe: object, plane: str, leg_and_system: str):
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

#Create a new dataframe containing mean, max, and min
def calculate_mean_min_max(df_list: list):
    results_list = []

    # Iterate through the list of DataFrames
    for i, df in enumerate(df_list):
        phase = df['Phase']
        df = df.drop(columns=['Phase'], level=0)
        df.columns = df.columns.get_level_values(1)

        # Calculate the mean, min, and max for each column
        mean_df = df.mean().astype(int).astype(str)
        min_val = df.min().astype(int).astype(str)
        max_val = df.max().astype(int).astype(str)
        
        # Concatenate dataframes and rename columns
        phase_header = str(phase.iloc[1,0])
        
        result = pd.concat([mean_df, min_val, max_val], axis=1)
        level_1 = ['mean','min','max']
        result.columns = level_1

        #combine the values into a single column 
        combine_values = lambda x, y, z: '{0} ({1}/{2})'.format(x, y, z)
        #result[Mean (Max/Min)]
        result[phase_header] = result.apply(lambda row: combine_values(row['mean'], row['min'], row['max']), axis=1)
        result.drop(columns=['mean', 'min', 'max'], inplace=True)

        result = pd.concat([result], axis=1) #keys=[phase_header],
        
        # Append the result to the list
        results_list.append(result)

# Concatenate the results into a single DataFrame
    result_df = pd.concat(results_list, names=[result], axis=1) #keys=[phase_header]
    return result_df

def gait_section_slicer(fc_df, stance=1):
    stance_section = ['Joint Vertex', 'Initial Strike', 'Loading Response', 'Midstance', 'Terminal Stance', 'Toe Off']
    swing_section = ['Joint Vertex', 'Initial Swing', 'Midswing', 'Terminal Swing']

    if stance:
        fc_df = fc_df[stance_section].reindex(columns=stance_section)
    else:
        fc_df = fc_df[swing_section].reindex(columns=swing_section)

    return fc_df

def datatable_settings_multiindex(mi_df, flatten_char = '_'):

    datatable_col_list = []
    
    levels = mi_df.columns.nlevels
    if levels == 1:
        for i in mi_df.columns:
            datatable_col_list.append({"name": i, "id": i})
    else:        
        columns_list = []
        for i in mi_df.columns:
            col_id = flatten_char.join(i)
            datatable_col_list.append({"name": i, "id": col_id})
            columns_list.append(col_id)
        mi_df.columns = columns_list

    datatable_data = mi_df.to_dict('records')
    
    return datatable_col_list, datatable_data

if __name__ == "__main__":
    #Constants
    ################
    csv_name = "Dataset_1_Ethan_01062023.csv"
    joint = 'RightKnee'
    pln = "Sagittal Plane Right"
    system = "RL - RunLab"
    ################
    df = pd.read_csv(csv_name, index_col=0, header=[0,1])
    #group list
    gl = slice_df_into_phases(df, pln, system)
    #calculated dataframe
    c_df = calculate_mean_min_max(gl)
    c_df.insert(0, 'Joint Vertex', c_df.index, True)
    c_df = c_df.reset_index(drop=True)
    #print(c_df)


    app = Dash(__name__)

    stance_df = gait_section_slicer(c_df, stance=1)
    stance_columns, stance_data = datatable_settings_multiindex(stance_df)

    swing_df = gait_section_slicer(c_df, stance=0)
    swing_columns, swing_data = datatable_settings_multiindex(swing_df)

    app.layout = dbc.Container([
    html.H3("Stance: Mean (Minimum/Maximum) for Each Joint"),
    html.Hr(),
    dash_table.DataTable(
        id='stance_table',
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto', #Adds wrapping to cells
        },
        data=stance_data,      
        columns=stance_columns
    ), 
    html.H3("Swing: Mean (Minimum/Maximum) for Each Joint"),
    html.Hr(),
    dash_table.DataTable(
        id='swing_table',
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto', #Adds wrapping to cells
        },
        data=swing_data,      
        columns=swing_columns
    ), 
])

    app.run_server(debug=True)
