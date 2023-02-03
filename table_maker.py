import pandas as pd
from dash import Dash, dash_table


#Constants
################
csv_name = "Dataset_1_Ethan_01062023.csv"
joint = 'RightKnee'
pln = "Sagittal Plane Right"
system = "RL - RunLab"
################
df = pd.read_csv(csv_name, index_col=0, header=[0,1])

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
        mean_df = df.mean().round(2).astype(str)
        min_val = df.min().round(2).astype(str)
        max_val = df.max().round(2).astype(str)
        
        # Concatenate dataframes and rename columns
        phase_header = str(phase.iloc[1,0])
        
        result = pd.concat([mean_df, min_val, max_val], axis=1)
        level_1 = ['mean','min','max']
        result.columns = level_1

        #combine the values into a single column 
        combine_values = lambda x, y, z: '{0} ({1}/{2})'.format(x, y, z)
        result['Mean (Max/Min)'] = result.apply(lambda row: combine_values(row['mean'], row['min'], row['max']), axis=1)
        result.drop(columns=['mean', 'min', 'max'], inplace=True)

        result = pd.concat([result], keys=[phase_header], axis=1)
        
        # Append the result to the list
        results_list.append(result)

# Concatenate the results into a single DataFrame
    result_df = pd.concat(results_list, names=[result], axis=1)
    return result_df


#group list
gl = slice_df_into_phases(df, pln, system)
#calculated dataframe
c_df = calculate_mean_min_max(gl)
print(c_df)


app = Dash(__name__)

def datatable_settings_multiindex(df, flatten_char = '_'):
    
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

columns, data = datatable_settings_multiindex(c_df)
#print(columns)
'''value = data[0]
print(type(value))
print(value)'''


app.layout = dash_table.DataTable(
    id='table',
    data=data, 
    columns=columns
)

if __name__ == '__main__':
    app.run_server(debug=True)
