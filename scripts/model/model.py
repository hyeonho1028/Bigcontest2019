import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    # y_hat = np.argmax(y_hat.reshape(y_true.shape[0], 64), axis=1)
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat, average='weighted'), True

def lgb_model(train, valid, train_label, seed=42, label=None):
    if label=='adjust_survival_time':
        params = {
            'objective':'binary',
            "boosting": "gbdt",
            'learning_rate': 0.001,
            # 'subsample' : 0.6,
            # 'sumsample_freq':1,
            # 'colsample_bytree':0.221856,
            'max_depth': 8,
            # 'max_bin':255,
            # "lambda_l1": 0.25,
            # "lambda_l2": 1,
            # 'min_child_weight': 0.2,
            # 'min_child_samples': 20,
            # 'min_gain_to_split':0.02,
            # 'min_data_in_bin':3,
            # 'bin_construct_sample_cnt':5000,
            # 'cat_l2':10,
            'metrics':'binary_error',
            'verbose':-1,
            'nthread':-1,
            'seed':seed
        }
        trn_label = train_label.loc[train_label['char_id'].isin(train['char_id']), label]
        val_label = train_label.loc[train_label['char_id'].isin(valid['char_id']), label]

        train_df = lgb.Dataset(train.drop(columns='char_id'), label=trn_label)
        valid_df = lgb.Dataset(valid.drop(columns='char_id'), label=val_label)
        
        lgb_model = lgb.train(params, train_df, 5000, valid_sets = [train_df, valid_df], early_stopping_rounds = 25, verbose_eval=50)
        preds = lgb_model.predict(valid.drop(columns='char_id'))

    else:
    # elif label=='survival_time':
        params = {
            'objective':'multiclass',
            'num_class':64,
            "boosting": "gbdt",
            'learning_rate': 0.001,
            'subsample' : 0.6,
            'sumsample_freq':1,
            'colsample_bytree':0.221856,
            'max_depth': 16,
            'max_bin':255,
            "lambda_l1": 0.25,
            "lambda_l2": 1,
            'min_child_weight': 0.2,
            'min_child_samples': 20,
            'min_gain_to_split':0.02,
            'min_data_in_bin':3,
            'bin_construct_sample_cnt':5000,
            'cat_l2':10,
            'verbose':-1,
            'nthread':-1,
            'seed':seed
        }
        trn_label = train_label.loc[train_label['acc_id'].isin(train['acc_id']), label]
        val_label = train_label.loc[train_label['acc_id'].isin(valid['acc_id']), label]
        trn_label-=1;val_label-=1

        train_df = lgb.Dataset(train.drop(columns='acc_id'), label=trn_label)
        valid_df = lgb.Dataset(valid.drop(columns='acc_id'), label=val_label)
        
        evals_result = {}
        lgb_model = lgb.train(params, train_df, 5000, valid_sets = [train_df, valid_df], early_stopping_rounds = 25, verbose_eval=50, feval=lgb_f1_score, evals_result=evals_result)
        preds = lgb_model.predict(valid.drop(columns='acc_id'))
        preds = np.argmax(preds, axis=1)+1
#     else:
#         params = {
#             'objective':'regression',
#             "boosting": "gbdt",
#             'learning_rate': 0.1,
# #             'subsample' : 0.6,
# #             'sumsample_freq':1,
# #             'colsample_bytree':0.221856,
# #             'max_depth': 16,
# #             'max_bin':255,
# #             'lambda_l1': 0.25,
# #             "lambda_l2": 1,
# #             'min_child_weight': 0.2,
# #             'min_child_samples': 20,
# #             'min_gain_to_split':0.02,
# #             'min_data_in_bin':3,
# #             'bin_construct_sample_cnt':5000,
# #             'cat_l2':10,
#             'verbose':-1,
#             'nthread':-1,
#             'metrics':'mse',
#             'seed':seed
#         }
#         trn_label = train_label.loc[train_label['acc_id'].isin(train['acc_id']), label]
#         val_label = train_label.loc[train_label['acc_id'].isin(valid['acc_id']), label]

#         train_df = lgb.Dataset(train.drop(columns='acc_id'), label=trn_label)
#         valid_df = lgb.Dataset(valid.drop(columns='acc_id'), label=val_label)

#         lgb_model = lgb.train(params, train_df, 5000, valid_sets = [train_df, valid_df], early_stopping_rounds = 25, verbose_eval=50)
#         preds = lgb_model.predict(valid.drop(columns='acc_id'))

# #         model = Ridge().fit(train.drop(columns='acc_id'), trn_label)
# #         preds = model.predict(valid.drop(columns='acc_id'))
    return preds



