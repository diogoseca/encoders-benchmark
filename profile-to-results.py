#%% imports
import pandas as pd


#%% encoders 
encoders_names = [
    'DropEncoder', 'BackwardDifferenceEncoder', 'BaseNEncoder', 
    'BinaryEncoder', 'CountEncoder', 'HashingEncoder', 'HelmertEncoder', 
    'OrdinalEncoder', 'OneHotEncoder', 'SumEncoder', 'CatBoostEncoder', 
    'GLMMEncoder', 'JamesSteinEncoder', 'LeaveOneOutEncoder', 
    'MEstimateEncoder', 'TargetEncoder', 'PolynomialEncoder',
]


#%% read profile csvs
profile = []
for name in encoders_names:
    columns = ['Dataset', 'MaxRAM', 'CPUTime']
    encoder_profile = pd.read_csv('profile/'+name+'.csv', names=columns, skiprows=1)
    encoder_profile['Encoder'] = name.replace('Encoder','')
    profile.append(encoder_profile)
profile = pd.concat(profile).set_index(['Dataset', 'Encoder'])
profile = profile.replace('\s.+', '', regex=True).astype(float)
profile


#%% RAM 
max_ram = profile['MaxRAM'].unstack(level='Dataset')
max_ram = max_ram.round(1)
max_ram.to_csv('results/profile-ram.csv')
max_ram.to_latex('results/profile-ram.tex')
max_ram


#%% CPU 
cpu_time = profile['CPUTime'].unstack(level='Dataset')
cpu_time = cpu_time.round(3)
cpu_time.to_csv('results/profile-cpu.csv')
cpu_time.to_latex('results/profile-cpu.tex')
cpu_time


# %%
