
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from time import process_time
from memory_profiler import memory_usage
from pathlib import Path


##################################################################
########################### DATASETS #############################
##################################################################
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
    ['IncParty','open','contested'], # ca2006
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
    ['district','IncName'], # ca2006
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


##################################################################
##################### LEARNING ALGORITHMS ########################
##################################################################

class MLPRegressor2(MLPRegressor):
    def __init__(self): super().__init__(hidden_layer_sizes=(100,100))
        
models = [
    ElasticNet,
    KNeighborsRegressor,
    SVR,
    #GaussianProcessRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    MLPRegressor,
    MLPRegressor2,
]


##################################################################
############### TRAINING TESTING AND MEASURING  ##################
##################################################################

def measure_encoder(encoder_class, encoder_name=None, save_results=False, 
                    profile_dir='profile/', metrics_dir='metrics/'):
    encoder_name = encoder_name or encoder_class.__name__
    print('Measuring '+encoder_name+'...')

    # things to measure
    metrics = defaultdict(list)
    max_ram = defaultdict(list)
    cpu_time = defaultdict(list)
    
    # measure the encoder on different datasets
    ds_bar = tqdm(list(zip(datasets, datasets_names, columns_drop, columns_targets, columns_nominals, columns_numericals)))
    for dataset, name, drop_cols, target_col, nominal_cols, numerical_cols in ds_bar:
        ds_bar.set_postfix_str('(dataset='+name+')')
        
        # split dataset into X and y
        X = dataset.drop(columns=target_col)
        y = dataset.loc[:, target_col]
        
        # measure the encoder on different dataset splits
        folds_bar = tqdm(RepeatedKFold(n_splits=2, n_repeats=15, random_state=2).split(X,y), total=30, leave=False)
        for train_index, test_index in folds_bar:
            folds_bar.set_postfix_str('(folds loop)')
            
            # split dataset into train + test
            Xy_train = dataset.iloc[train_index].copy()
            Xy_test = dataset.iloc[test_index].copy()

            # standardize using training data's mean and std
            numerical_cols = list(set(numerical_cols) & set(Xy_train.select_dtypes('number').columns))
            mean = Xy_train.loc[:, numerical_cols].mean()
            std = Xy_train.loc[:, numerical_cols].std()
            Xy_train.loc[:, numerical_cols] = (Xy_train.loc[:, numerical_cols] - mean)/std
            Xy_test.loc[:, numerical_cols] = (Xy_test.loc[:, numerical_cols] - mean)/std

            # use sklearn standard
            X_train = Xy_train.drop(columns=[target_col])
            X_test = Xy_test.drop(columns=[target_col])
            y_train = Xy_train.loc[:, target_col]
            y_test = Xy_test.loc[:, target_col]
            
            # define the encoder application
            def apply_encoder(nominal_cols, X_train, X_test, y_train):
                encoder = encoder_class(cols=nominal_cols).fit(X=X_train, y=y_train)
                X_train = encoder.transform(X=X_train)
                X_test = encoder.transform(X=X_test)
                return X_train, X_test

            # apply the encoder with profiling of time and memory
            start_time = process_time()
            memory, (X_train, X_test) = memory_usage(
                (apply_encoder, (nominal_cols, X_train, X_test, y_train), {}), include_children=True, retval=True
            )

            # save profiling results of this validation fold
            max_ram[name] += [max(memory) - min(memory)]
            cpu_time[name] += [process_time() - start_time]

            # measure the encoder on different models
            models_bar = tqdm(models, leave=False)
            for model in models_bar:
                model_name = model.__name__
                models_bar.set_postfix_str('(model='+model_name+')')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = model().fit(X=X_train, y=y_train)
                pred_test = model.predict(X=X_test)
                metrics[name, model_name, 'RMSE'] += [mean_squared_error(y_test, pred_test, squared=False)]
                metrics[name, model_name, 'MAE'] += [median_absolute_error(y_test, pred_test)]
                metrics[name, model_name, 'R2'] += [r2_score(y_test, pred_test)]
    
    # aggregate metrics
    metrics = {k: "{:.3f} ±{:.3f}".format(np.mean(metrics[k]), np.std(metrics[k])) for k in metrics}
    metrics = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=[encoder_name])
    
    # aggregate profile
    max_ram = {k: max(max_ram[k]) for k in max_ram}
    cpu_time = {k: max(cpu_time[k]) for k in cpu_time}
    profile = pd.DataFrame([[max_ram[d], cpu_time[d]] for d in datasets_names], 
                           columns=['Max RAM ()', 'CPU time (s)'], index=datasets_names)

    # save results
    if save_results:
        # metrics
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_dir+encoder_name+'.csv')
        # profile
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        profile.to_csv(profile_dir+encoder_name+'.csv')
    # return results
    else:
        return metrics, profile