import pandas as pd


csv_name = "Dataset_1_Ethan_01062023.csv"
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

gcl = slice_df_gait_cycles(df, "Sagittal Plane Right", "RL - RunLab")

'''for index, cycle in enumerate(gcl):
    print(f"DataFrame {index+1}:")
    print(cycle)'''

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

final_df = reindex_to_percent_complete(gcl)

print(final_df.to_string())
