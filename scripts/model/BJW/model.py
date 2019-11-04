class model(object):
    def __init__(self, train, test, folds=FOLDS, seed=SEED, log=True):
        self.train = train
        self.test = test.drop('SELNG_PRER_STOR', axis=1)
        self.kf = KFold(n_splits=folds, random_state=seed)
        self.log = log
    
    def ridge_model(self):
        oof = np.zeros(len(self.train))
        pred = np.zeros(len(self.test))
        for trn_idx, val_idx in self.kf.split(self.train):
            train_df = self.train.loc[trn_idx]
            valid_df = self.train.loc[val_idx].drop('SELNG_PRER_STOR', axis=1)
            if self.log:
                ridge_model = Ridge().fit(train_df.drop('SELNG_PRER_STOR', axis=1), np.log1p(train_df['SELNG_PRER_STOR']))
            else:
                ridge_model = Ridge().fit(train_df.drop('SELNG_PRER_STOR', axis=1), train_df['SELNG_PRER_STOR'])
            oof[val_idx] = ridge_model.predict(valid_df)
            pred += ridge_model.predict(self.test)/self.kf.n_splits
        return oof, pred
    
    
    def rf_model(self):
        
    
        
        return oof, pred
    
    
    def xgb_model(self):
        warnings.filterwarnings(action='ignore')
        
        params={
            'eta': 0.001,
            'max_depth': 16,
            'min_child_weight': 16,
            'n_estimators' : 10000,
            'subsample': 0.9
        }
    
    
        oof = np.zeros(len(self.train))
        pred = np.zeros(len(self.test))
        for trn_idx, val_idx in self.kf.split(self.train):
            if self.log:
                train_df = xgb.DMatrix(self.train.loc[trn_idx].drop('SELNG_PRER_STOR', axis=1), label=np.log1p(self.train.loc[trn_idx, 'SELNG_PRER_STOR']))
                valid_df = xgb.DMatrix(self.train.loc[val_idx].drop('SELNG_PRER_STOR', axis=1), label=np.log1p(self.train.loc[val_idx, 'SELNG_PRER_STOR']))
            else:
                train_df = xgb.DMatrix(self.train.loc[trn_idx].drop('SELNG_PRER_STOR', axis=1), label=self.train.loc[trn_idx, 'SELNG_PRER_STOR'])
                valid_df = xgb.DMatrix(self.train.loc[val_idx].drop('SELNG_PRER_STOR', axis=1), label=self.train.loc[val_idx, 'SELNG_PRER_STOR'])
            xgb_model = xgb.train(params, train_df, num_boost_round=30000, evals=[(train_df, 'train'), (valid_df, 'val')], verbose_eval=5000, early_stopping_rounds=500)
            oof[val_idx] = xgb_model.predict(xgb.DMatrix(self.train.loc[val_idx].drop('SELNG_PRER_STOR', axis=1)))
            pred += xgb_model.predict(xgb.DMatrix(self.test))/self.kf.n_splits
        return oof, pred
    
    def rmse(self, true, pred):
        
        if self.log:
            true = np.expm1(pred)
            mse = mean_squared_error(true, pred)
            rmse = np.round(np.sqrt(mse), 2)
        else:
            mse = mean_squared_error(true, pred)
            rmse = np.round(np.sqrt(mse), 2)
    
        return rmse
