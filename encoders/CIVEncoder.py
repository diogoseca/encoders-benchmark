from measure import measure_encoder
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_kernels
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from openTSNE import TSNE


class CIVEncoder:
    def __init__(self, cols=None, default_term=1):
        self.nominal_cols = cols
        self.default_term = default_term

    def mmd2u(self, X, Y):
        # code from functon above
        m = len(X)
        n = len(Y)

        # stack all instances together
        XY = np.vstack([X, Y])
        # compute the kernel matrix using RBF kernel function
        K = pairwise_kernels(XY, metric='rbf', n_jobs=-1)
        # term 1
        if m==1:
            term1 = self.default_term
        else:
            Kx = K[:m, :m]
            term1 = (Kx.sum() - Kx.diagonal().sum()) / (m * (m - 1))
        # term 2
        if n==1:
            term2 = self.default_term
        else:
            Ky = K[m:, m:]
            term2 = (Ky.sum() - Ky.diagonal().sum()) / (n * (n - 1))
        # term 3
        Kxy = K[:m, m:]
        term3 = Kxy.sum() * (2.0 / (m * n))
        # compute the MMD² unbiased statistic
        return term1 + term2 - term3

    def mmd_matrix(self, data, categories, max_sample=1000):
        # standardize numeric data
        data = (data - data.mean()) / data.std()

        # init the dissimilarity matrix
        categories_uniques = categories.unique()
        dmatrix = pd.DataFrame(index=categories_uniques, columns=['distance to '+str(a) for a in categories_uniques])

        # calculate the dissimilarity matrix
        for i in range(dmatrix.shape[0]):
            # get sample_i
            sample_i = data[categories==dmatrix.index[i]]
            if len(sample_i) > max_sample: sample_i = sample_i.sample(max_sample)
            
            # the diagonal should be 0. let's see what the estimated value is 
            dmatrix.iloc[i,i] = self.mmd2u(sample_i, sample_i)
        
            for j in range(i+1, dmatrix.shape[1]):
                # get sample_j
                sample_j = data[categories==dmatrix.index[j]]
                if len(sample_j) > max_sample: sample_j = sample_j.sample(max_sample)
                
                # calculate the dissimilarity between sample_i and sample_j
                dissimilarity = self.mmd2u(sample_i, sample_j)
                
                # save the dissimilarity to the matrix
                dmatrix.iloc[j,i] = dissimilarity
                dmatrix.iloc[i,j] = dissimilarity
        
        # adjust the dissimilarities so that the diagonal becomes 0s
        for i in range(dmatrix.shape[1]):
            dmatrix.iloc[:, i] -= dmatrix.iloc[i,i]

        # root the squared MMD
        dmatrix = dmatrix.pow(0.5)

        return dmatrix
            
    def fit(self, X, y):
        numeric_data = X.join(y).drop(columns=self.nominal_cols)
        nominal_data = X[self.nominal_cols]
        
        # calculate cat->mmd mappings
        self.mmds = {}
        for col in self.nominal_cols:
            mmds = self.mmd_matrix(numeric_data, nominal_data[col])
            mmds.columns = [col+': '+c for c in mmds.columns]
            self.mmds[col] = mmds

        # encode categoricals
        for col in self.nominal_cols:
            nominal_data = nominal_data.join(self.mmds[col], on=col).drop(columns=col)
            new_cols = self.mmds[col].columns
            means = self.mmds[col].mean()
            nominal_data.loc[:, new_cols] = nominal_data.loc[:, new_cols].fillna(means)

        # train dimensionality reductor (tsne)
        n_components = min(2, nominal_data.shape[0], nominal_data.shape[1])
        self.dim_reducer = TSNE(n_components=n_components, perplexity=3, n_jobs=-1).fit(nominal_data.values)
        #self.dim_reducer = MDS(metric=True).fit(nominal_data.values)
        #self.dim_reducer = MDS(metric=False).fit(nominal_data.values)
        # TODO: test cMDS and NMDS - sklearn.manifold.MDS
        # TODO: test DiSTATIS - DistatisR:statis
        # TODO: test NCA ??

        return self

    def transform(self, X):
        numeric_data = X.drop(columns=self.nominal_cols)
        nominal_data = X[self.nominal_cols]

        # encode categoricals
        for col in self.nominal_cols:
            nominal_data = nominal_data.join(self.mmds[col], on=col).drop(columns=col)
            new_cols = self.mmds[col].columns
            # fill nans with means - there might be other ways
            means = self.mmds[col].mean()
            nominal_data.loc[:, new_cols] = nominal_data.loc[:, new_cols].fillna(means)

        # reduce cat dimensions
        nominal_data = self.dim_reducer.transform(nominal_data.values)
        nominal_data = pd.DataFrame(nominal_data, index=X.index, columns=['CIV'+str(i) for i in range(nominal_data.shape[1])])
        
        # concat numeric with categorical
        X = pd.concat([numeric_data, nominal_data], axis=1)
        return X
    

### test time!!
from functools import partial
measure_encoder(partial(CIVEncoder, default_term=1), save_results=True)