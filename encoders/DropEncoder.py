from measure import measure_encoder

class DropEncoder:
    def __init__(self, cols=None):
        self.drop_cols=cols
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols)
    
measure_encoder(DropEncoder, save_results=True)