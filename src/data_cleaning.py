import pandas as pd

def clean_tract_fips(df, col, new_col):
    fips = []
    for i in df.index:
        val = str(df[col][i])
        if len(val) ==10:
            val = "0" + val
        fips.append(val)
        
    new_df = df.copy()
    new_df[new_col] = fips
    new_df = new_df.drop_duplicates(subset=[new_col])
        
    return new_df

def merge_dfs(df1, df2, cols, on='new_fips', how='inner'):
    df1 = df1.merge(df2, on='new_fips', how='inner')
    for col in cols:
        df1.drop(col, axis=1, inplace=True)
    return df1

def output_new_chem_df(df, org_col_title, col_val):
    new_df = df[df[org_col_title] == col_val]
    return new_df

def merge_chem_dfs(df_to_merge, chem_df, pollutant, col_to_merge_on):
    cancer_risk_col = pollutant + "_cancer_risk_per_million"
    respiratory_HI_col = pollutant + "_repiratory_HI"
    chem_df[cancer_risk_col] = chem_df['Total.Cancer.Risk..per.million.']
    chem_df[respiratory_HI_col] = chem_df['Total.Respiratory.HI']
    df_to_merge = df_to_merge.merge(chem_df, on='new_fips', how = 'inner')
    df_to_merge.drop(['Total.Respiratory.HI', 'Total.Cancer.Risk..per.million.', 'Tract', 'Pollutant.Name'], axis=1, inplace=True)
    return df_to_merge

def remove_rows_with_val(df, col_name, val):
    df.drop(df.loc[df[col_name]==val].index, inplace=True)

if __name__ == '__main__':
    #read in csv files to combine into single dataframe for analysis
    cities_df = pd.read_csv('./tract_data/500_cities_data.csv')
    
    superfunds = pd.read_csv('./tract_data/superfunds.csv')
    superfunds = superfunds.loc[:, ['FIPS_Tract', 'has_superfund']]

    chems = pd.read_csv('./tract_data/air_pol.csv')
    chems = chems.loc[:, ['Tract', 'Pollutant.Name', 'Total.Cancer.Risk..per.million.', 'Total.Respiratory.HI']]

    svi = pd.read_csv('./tract_data/SVI.csv')
    svi = svi.loc[:, ['FIPS','EP_POV', 'EP_MINRTY', 'EP_NOHSDP', 'EP_LIMENG', 'EP_AGE65']].reset_index(drop=True)

    pm = pd.read_csv('./tract_data/daily_PM25.csv')
    pm = pm.loc[:, ['ctfips', 'DS_PM_pred', 'DS_PM_stdd']]
    pm = pm.groupby(['ctfips']).mean().reset_index()



    cities = cities_df.loc[:, ['StateAbbr', 'PlaceName', 'PlaceFIPS', 'TractFIPS', 
     'Place_TractID', 'Population2010', 'CANCER_CrudePrev', 'CANCER_Crude95CI',
     'CASTHMA_CrudePrev', 'CASTHMA_Crude95CI', 'COPD_CrudePrev', 'COPD_Crude95CI']]
    
    final = clean_tract_fips(cities, 'TractFIPS', 'new_fips')
    final.drop(['PlaceFIPS', 'TractFIPS', 'Place_TractID'], axis=1, inplace=True)
    
    sf_cleaned = clean_tract_fips(superfunds, 'FIPS_Tract', 'new_fips')
    final = merge_dfs(final, sf_cleaned, cols=['FIPS_Tract'], on='new_fips', how = 'inner')

    chem_cols = ['ACETALDEHYDE', 'BENZENE', '1,3-BUTADIENE', 'CYANIDE COMPOUNDS',
       'DIESEL PM', 'TOLUENE']

    #list of dataframes, broken out one for each poolutant
    chem_df_list = [output_new_chem_df(chems, 'Pollutant.Name', col_val)
                    for col_val in chem_cols]

    #another list of data frames, after they have been cleaned up a bit
    cleaned_chem_df_list = [clean_tract_fips(chem_df, 'Tract', 'new_fips') for chem_df in chem_df_list]

    #tuple of pollutant name & chem df for which column names will be adjusted 
    chem_tups =  list(zip(chem_cols, cleaned_chem_df_list))
    
    for tup in chem_tups:
        final = merge_chem_dfs(final, tup[1], tup[0], 'new_fips')
    
    svi_cols = ['EP_POV', 'EP_MINRTY', 'EP_NOHSDP', 'EP_LIMENG', 'EP_AGE65']

    for col in svi_cols:
        remove_rows_with_val(svi, col, -999.0)
    cleaned_svi = clean_tract_fips(svi, 'FIPS', 'new_fips')

    final = merge_dfs(final, cleaned_svi, cols=['FIPS'], on='new_fips', how='inner')

    pm_cleaned = clean_tract_fips(pm, 'ctfips', 'new_fips')
    final = merge_dfs(final, pm_cleaned, cols=['ctfips'], on='new_fips', how='inner')

    #final.to_csv (r'combined_df.csv', index=False, header=True)

    #creating various data frames for initial eda/additional testing if desired
    demographic_df = final.drop(['StateAbbr', 'PlaceName', 'Population2010', 'CANCER_Crude95CI', 'CASTHMA_CrudePrev', 'CASTHMA_Crude95CI', 'COPD_CrudePrev',
                                'COPD_Crude95CI', 'has_superfund', 'ACETALDEHYDE_cancer_risk_per_million', 'ACETALDEHYDE_repiratory_HI', 'BENZENE_cancer_risk_per_million',
                                'BENZENE_repiratory_HI', '1,3-BUTADIENE_cancer_risk_per_million', '1,3-BUTADIENE_repiratory_HI', 'CYANIDE COMPOUNDS_cancer_risk_per_million', 
                                'CYANIDE COMPOUNDS_repiratory_HI', 'DIESEL PM_cancer_risk_per_million', 'DIESEL PM_repiratory_HI', 'TOLUENE_cancer_risk_per_million', 
                                'TOLUENE_repiratory_HI', 'DS_PM_pred', 'DS_PM_stdd'], axis=1)

    enviro_df = final.drop(['StateAbbr', 'PlaceName', 'Population2010', 'CANCER_Crude95CI', 'CASTHMA_CrudePrev', 'CASTHMA_Crude95CI', 'COPD_CrudePrev',
                                'COPD_Crude95CI','ACETALDEHYDE_cancer_risk_per_million', 'BENZENE_cancer_risk_per_million', '1,3-BUTADIENE_cancer_risk_per_million', 'CYANIDE COMPOUNDS_cancer_risk_per_million', 
                                'DIESEL PM_cancer_risk_per_million', 'TOLUENE_cancer_risk_per_million', 'DS_PM_stdd', 'EP_POV', 'EP_MINRTY', 'EP_NOHSDP', 'EP_LIMENG', 'EP_AGE65'], axis=1)

    env_avg_df = enviro_df.copy()
    col = enviro_df.loc[:, 'ACETALDEHYDE_repiratory_HI':'TOLUENE_repiratory_HI']

    env_avg_df.drop(['ACETALDEHYDE_repiratory_HI', 'BENZENE_repiratory_HI', '1,3-BUTADIENE_repiratory_HI', 'CYANIDE COMPOUNDS_repiratory_HI', 
                                    'DIESEL PM_repiratory_HI', 'TOLUENE_repiratory_HI'], axis=1, inplace=True)
    env_avg_df['avg_chem_HI'] = col.mean(axis=1)
    #enviro_df['avg_HI'] = sum(['TOLUENE_repiratory_HI', 'ACETALDEHYDE'])

    #only run if need to create another csv file from these datasets
    # demographic_df.to_csv (r'demographic_df.csv', index=False, header=True)
    # enviro_df.to_csv (r'enviro_df.csv', index=False, header=True)
    
    #env_avg_df.to_csv (r'env_avg_df.csv', index=False, header=True)

