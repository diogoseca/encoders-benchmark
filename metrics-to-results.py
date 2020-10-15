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


#%% transform metrics to dataframe of scores
metrics = []
columns = ['Dataset', 'LearningAlgorithm', 'Metric', 'Mean']
for name in encoders_names:
    encoder_metrics = pd.read_csv('metrics/'+name+'.csv', names=columns, skiprows=1)
    encoder_metrics['Encoder'] = name
    metrics.append(encoder_metrics)
metrics = pd.concat(metrics)
metrics.Encoder = metrics.Encoder.str.replace('Encoder','')
metrics.LearningAlgorithm.replace({
    'ElasticNet':'EN',
    'GradientBoostingRegressor':'GBT',
    'KNeighborsRegressor':'KNN',
    'MLPRegressor':'NN-1L',
    'MLPRegressor2':'NN-2L',
    'RandomForestRegressor':'RF',
    'SVR':'SVM',
}, inplace=True)
metrics = metrics.set_index(['Encoder', 'Dataset', 'LearningAlgorithm', 'Metric']).iloc[:,0]
metrics = metrics.str.replace('\s.+', '', regex=True).astype(float)
metrics = metrics.unstack(level='Dataset')
rmse = metrics.xs('RMSE', level='Metric')
interquartile_range = lambda df: df.quantile(.75) - df.quantile(.25)
dist_to_best_scaled = lambda df: (df - df.min()) / interquartile_range(df)
scores = dist_to_best_scaled(rmse)
scores.to_csv('results/scores.csv')
scores

    

#%% BEST ENCODER PER ALGORITHM
# E.g.:
#						algorithm1	algorithmN 	algorithmMean*
#			encoder1	mean		mean 		mean
#			encoder2	mean		mean 		mean
#
encoders_algorithms = scores.mean(axis=1).unstack('LearningAlgorithm').rename_axis(None, axis=0)
encoders_algorithms['MeanScore'] = encoders_algorithms.mean(axis=1)
encoders_algorithms.sort_values('MeanScore', inplace=True)
encoders_algorithms = encoders_algorithms.round(4)
encoders_algorithms.to_csv('results/encoders-algorithms.csv')
encoders_algorithms.to_latex('results/encoders-algorithms.tex')
encoders_algorithms
    

#%% BEST ENCODER PER DATASET
# E.g.:
#						dataset1	datasetN 	datasetMean*
#			encoder1	mean		mean 		mean
#			encoder2	mean		mean 		mean
#
encoders_datasets = scores.stack().unstack('LearningAlgorithm').mean(axis=1)
encoders_datasets = encoders_datasets.unstack('Dataset').rename_axis(None, axis=0)
encoders_datasets['MeanScore'] = encoders_datasets.mean(axis=1)
encoders_datasets.sort_values('MeanScore', inplace=True)
encoders_datasets = encoders_datasets.round(4)
encoders_datasets.to_csv('results/encoders-datasets.csv')
encoders_datasets.to_latex('results/encoders-datasets.tex')
encoders_datasets


# %%
