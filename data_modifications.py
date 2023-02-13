import pandas as pd

def process_df(upload):
    #Necessary to rebuild the dataframe after importing from df
    df_from_json = pd.read_json(upload, orient='split')

    #Rebuild the row and column headers
    df_from_json.set_index(df_from_json.columns[0], inplace=True)
    df_from_json.index.rename(None, inplace=True)
    level_0 = df_from_json.columns.tolist()
    clean_header = [item.split('.')[0] for item in level_0]
    level_1 = df_from_json.iloc[0].tolist()
    df_from_json.columns = pd.MultiIndex.from_tuples(list(zip(clean_header, level_1)))
    df_from_json = df_from_json.iloc[1:]

    #Split numbers and phases, convert numbers to float, then reattach
    phases = df_from_json.loc[
        :, df_from_json.columns[
            df_from_json.columns.get_level_values(0).isin(['Phase'])
        ]
    ]
    angles = df_from_json.drop(columns=[('Phase')], level=0)
    float_angles = angles.apply(lambda x: x.astype('float64'))

    df = pd.concat([float_angles, phases], axis=1, sort=False)
    
    return df