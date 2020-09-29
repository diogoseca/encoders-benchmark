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
metrics = metrics.set_index(['Encoder', 'Dataset', 'LearningAlgorithm', 'Metric']).iloc[:,0]
metrics = metrics.str.replace('\s.+', '', regex=True).astype(float)
metrics = metrics.unstack(level='Dataset')
metrics.columns.name = 'DATASETS:'
rmse = metrics.xs('RMSE', level='Metric')
interquartile_range = lambda df: df.quantile(.75) - df.quantile(.25)
dist_to_best_scaled = lambda df: (df - df.min()) / interquartile_range(df)
scores = dist_to_best_scaled(rmse)


#%% BEST GENERAL-PURPOSE {MODEL+ENCODER}
# E.g.:						 	dataset1	datasetN 	Score*
#			model 1 encoder 1	mean		mean 		score
#					encoder N	mean		mean 		score
#			model N encoder 1	mean		mean 		score
#					encoder N	mean		mean 		score
# 
best_models_encoders = scores.copy()
best_models_encoders['MeanScore'] = scores.mean(axis=1)
best_models_encoders.sort_values('MeanScore', inplace=True)
best_models_encoders.to_csv('results/best-models-encoders.csv')
best_models_encoders
    

#%% BEST GENERAL-PURPOSE ENCODER
# E.g.:
#						dataset1	datasetN 	datasetMean*
#			encoder1	mean		mean 		mean
#			encoder2	mean		mean 		mean
#
best_encoders = best_models_encoders['MeanScore'].unstack('LearningAlgorithm').rename_axis(None, axis=0)
best_encoders['MeanScore'] = best_encoders.mean(axis=1)
best_encoders.sort_values('MeanScore', inplace=True)
best_encoders.to_csv('results/best-encoders.csv')
best_encoders


# %%
