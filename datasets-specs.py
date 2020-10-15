#%%
import numpy as np
import pandas as pd


#%%
datasets_names = [
    'codling', 
    'nassCDS', 
    'races2000', 
    'terrorism', 
    'midwest', 
    'mpg', 
    'msleep', 
    'txhousing', 
    'gtcars', 
    'water', 
    'ca2006', 
    'UKHouseOfCommons', 
    'myeloid', 
    'us_rent_income', 
    'Baseball',
    #'spotify',
]
columns_targets = [
    'dead', # codling
    'injSeverity', # nassCDS
    'time', # races2000
    'nkill.us', # terrorism
    'percollege', # midwest
    'cty', # mpg
    'sleep_total', # msleep
    'sales', # txhousing
    'mpg_c', # gtcars
    'mortality', # water
    'Bush2004', # ca2006
    'y1', # UKHouseOfCommons
    'futime', # myeloid
    'estimate', # us_rent_income
    'years', # Baseball
    #'danceability', # spotify
]
columns_nominals = [
    ['Cultivar'], # codling
    ['dvcat','dead','airbag','seatbelt','frontal','sex','abcat','occRole','deploy'], # nassCDS
    ['type'], # races2000
    ['methodology','method'], # terrorism
    ['county','state','inmetro','category'], # midwest
    ['manufacturer','model','trans','drv','fl','class'], # mpg
    ['genus','vore','order'], # msleep
    ['city'], # txhousing
    ['mfr','trim','bdy_style','drivetrain','trsmn','ctry_origin'], # gtcars
    ['location'], # water
    ['IncParty','open'], # ca2006
    ['constituency','county',], # UKHouseOfCommons
    ['trt','sex'], # myeloid
    ['NAME','variable'], # us_rent_income
    ['league86', 'div86', 'team86', 'posit86', 'league87', 'team87'], # Baseball
    #['artists','explicit','mode'], # spotify
]
columns_drop = [
    [], # codling
    ['caseid'], # nassCDS
    ['timef'], # races2000
    ['pNA.nkill','pNA.nkill.us','pNA.nwound','pNA.nwound.us','worldPopulation','USpopulation','worldDeathRate','USdeaths','kill.pmp','kill.pmp.us','pkill','pkill.us'], # terrorism
    ['PID',], # midwest
    [], # mpg
    ['name','conservation'], # msleep
    [], # txhousing
    ['model'], # gtcars
    ['town'], # water
    ['district','IncName','contested'], # ca2006
    [], # UKHouseOfCommons
    ['id'], # myeloid
    ['GEOID'], # us_rent_income
    ['name1','name2'], # Baseball   
    #['name'], # spotify
]
datasets = [pd.read_csv('data/'+path+'.csv', index_col=0) for path in datasets_names]
datasets = [dataset.drop(columns=drop_cols).dropna() for dataset,drop_cols in zip(datasets, columns_drop)]
columns_numericals = [list(set(dataset.select_dtypes('number').columns) - set(nominals))
                      for dataset, nominals in zip(datasets, columns_nominals)]


#%%
datasets_specs = pd.DataFrame(index=datasets_names)
datasets_specs['Target Var'] = columns_targets
datasets_specs['Drop Vars'] = columns_drop
datasets_specs['Instances'] = [d.shape[0] for d in datasets]
datasets_specs['Variables'] = [d.shape[1] for d in datasets]
datasets_specs['Numericals'] = [len(n) for n in columns_numericals]
datasets_specs['Nominals'] = [len(n) for n in columns_nominals]
datasets_specs['Cardinality'] = [d.loc[:, n].nunique().values for d,n in zip(datasets, columns_nominals)]
datasets_specs.to_csv('results/datasets-specs.csv')
datasets_specs


# %%
