# Essentials
import os
import sys
import numpy as np
import pandas as pd
import datetime
import random
import argparse
from math import sqrt
from time import time
from collections import Counter
# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Stats
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p, inv_boxcox
from scipy.stats import boxcox_normmax

# Misc
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
# from sklearn.utils.fixes import loguniform
from sklearn.model_selection import train_test_split

import torch
import warnings
warnings.filterwarnings(action="ignore")

# Get the directory two levels up
two_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Append to sys.path
sys.path.append(two_up)
from data import MetaLearningSystemDataLoader
from utils.parser_utils import get_args


def mae(y, y_pred):
    y_c = y[(y_pred < 100) & (y_pred > 0)]
    y_pred_c = y_pred[(y_pred < 100) & (y_pred > 0)]
    return np.mean(np.abs(y_c-y_pred_c))


def smape(y, y_pred):
    y_c = y[(y < 100) & (y > 0)]
    y_pred_c = y_pred[(y < 100) & (y > 0)]

    numerator = np.abs(y_c - y_pred_c)
    denominator = 0.5*(y_c + y_pred_c)
    return np.mean(numerator / denominator)


def r_square(y, y_pred):
    y_c = y[(y < 100) & (y > 0)]
    y_pred_c = y_pred[(y < 100) & (y > 0)]
    mean_y = np.mean(y_c)
    ss_tot = np.sum((y_c - mean_y) ** 2)
    ss_res = np.sum((y_c - y_pred_c) ** 2)
    return 1 - (ss_res / ss_tot)


# Hyperparameter to be set manually
n_iter = 100
meta_cv = False
os.environ['DATASET_DIR'] = '../../'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

args, device = get_args()

train_rng = np.random.RandomState(seed=args.train_seed)
train_seed = train_rng.randint(1, 999999)
test_rng = np.random.RandomState(seed=args.val_seed)
test_seed = test_rng.randint(1, 999999)

rng = np.random.RandomState(seed=test_seed)
dataset_df = pd.read_csv(args.work_path + 'datasets/ICC2018_cleaned_1.csv')
vc = dataset_df['node_name'].value_counts().to_frame()
total_samples = args.num_samples_per_class + args.num_target_samples

total_label_types = sum(vc['node_name'] >= total_samples)
label_list = (vc['node_name'] >= total_samples).index.to_list()
rng.shuffle(label_list)

x_train_id = int(args.train_val_test_split[0] * total_label_types)
x_train_classes = label_list[:x_train_id]
x_train_df = dataset_df[dataset_df['node_name'].isin(x_train_classes)].copy()
if args.label_name != "buffer_rate":  # 只能是startup_delay 和 buffer_rate；默认都放到最后一列
    # x_train_df["buffer_rate"] = 4 * x_train_df["tcp_conntime"] + x_train_df["avg_fbt_time"]
    x_train_df['buffer_rate'], x_train_df[args.label_name] = x_train_df[args.label_name], x_train_df['buffer_rate']
    if args.label_name == "AvgQualityIndex": 
        x_train_df["buffer_rate"] = x_train_df["buffer_rate"] * 100
else:
    x_train_df[args.label_name] = x_train_df[args.label_name] * 100

train_features, train_labels = [], []
for label in x_train_classes:
    tmp = x_train_df.sample(n=total_samples, random_state=args.train_seed).values[:, args.num_of_classes:]
    train_features.append(tmp[:, :-1])
    train_labels.append(tmp[:, -1:])
train_features = np.array(train_features).reshape((-1, args.image_channels))
train_labels = np.array(train_labels).reshape((-1, 1))

opt_catboost_params = {'bootstrap_type': 'Bernoulli',
                       'random_strength': 1,
                       # 'min_data_in_leaf': 10,
                       'l2_leaf_reg': 3,
                       'learning_rate': 0.01,
                       'loss_function': 'MAE',
                       'eval_metric': 'MAE',
                       # 'grow_policy': 'Depthwise',
                       # 'leaf_estimation_method': 'Exact',
                       # 'task_type': 'GPU',
                       # 'devices': '1',
                       'od_type': 'IncToDec',
                       # 'od_wait': 500,
                       # 'metric_period': 500,
                       # 'od_pval': 1e-10,
                       # 'max_ctr_complexity': 8,
                       'depth': 10,
                       'iterations': 10,
                       'verbose': False,
                       'random_seed': train_seed}
cb = CatBoostRegressor(**opt_catboost_params)
cv = KFold(n_splits=5, shuffle=False)

if meta_cv:
    param_dist = {
        "depth": [5, 10, 25, 50, 100],
        "iterations": [10, 50, 100, 500, 1000, 2500, 3000],
        # "min_child_samples": [18, 19, 20, 21, 22],
        # "min_child_weight": [0.001, 0.002],
        # "subsample": stats.uniform(0, 1),
    }
    random_search = RandomizedSearchCV(
        cb, param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv,
        refit=False,
        random_state=train_seed
    )

    start = time()
    random_search.fit(train_features, train_labels)
    print(
        "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
        % ((time() - start), n_iter)
    )
    print("Best parameters: ", random_search.best_params_)
    # CV on original data
    opt_catboost_params.update(random_search.best_params_)
    # opt_catboost_params.pop('task_type')
    # opt_catboost_params.pop('devices')
    cb_best = CatBoostRegressor(**opt_catboost_params)
else:
    print("Skip meta training for searching best parameters: ")
    cb_best = CatBoostRegressor(**opt_catboost_params)

del train_features, train_labels

MLDataLoader = MetaLearningSystemDataLoader(args=args)
train_time, infer_time = [], []
mae_lst, smape_lst, r2_lst = [], [], []
for test_sample_idx, test_sample in enumerate(MLDataLoader.get_test_batches(total_batches=int(args.num_evaluation_tasks / args.batch_size),
                                              augment_images=False)):
    x_support_set, x_target_set, y_support_set, y_target_set, _ = test_sample # x_support_set in shape [8, 5, 64, 16, 1, 1]; y_support_set in shape [8, 5, 64]; target_set is same.
    tb, cn, sn, c = x_support_set.size(0), x_support_set.size(1), x_support_set.size(2), x_support_set.size(3)
    x_support_set, x_target_set = torch.squeeze(x_support_set).view(-1, sn, c), torch.squeeze(x_target_set).view(-1, sn, c) # x_support_set in shape [40, 64, 16];
    y_support_set, y_target_set = y_support_set.view(-1, sn, 1), y_target_set.view(-1, sn) # y_support_set in shape [40, 64, 1];

    train_time_task, infer_time_task = [], []
    mae_lst_task, smape_lst_task, r2_lst_task = [], [], []

    for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(
            zip(x_support_set, y_support_set, x_target_set, y_target_set)):
        x_support, x_target = x_support_set_task.numpy(), x_target_set_task.numpy() # x_support and x_target in shape [320, 16]
        y_support, y_target = y_support_set_task.numpy(), y_target_set_task.view(-1, ).numpy() # y_support in shape [320, 1]; y_target in shape [320, ];

        start1 = time()
        cb_best.fit(x_support, y_support)
        end1 = time()
        test_pred = cb_best.predict(x_target).reshape(-1)
        end2 = time()
        pred_mae, pred_smape, pred_r2 = mae(y_target, test_pred), smape(y_target, test_pred), r_square(y_target, test_pred)
        train_time_task.append(end1 - start1)
        infer_time_task.append(end2 - end1)
        mae_lst_task.append(pred_mae)
        smape_lst_task.append(pred_smape)
        r2_lst_task.append(pred_r2)

    # print(f"Batch {test_sample_idx}: training time is {end1 - start1:.2f} s")
    # print(f"Batch {test_sample_idx}: inference time is {end2 - end1:.2f} s")
    train_time.append(np.mean(train_time_task))
    infer_time.append(np.mean(infer_time_task))
    mae_lst.append(np.mean(mae_lst_task))
    smape_lst.append(np.mean(smape_lst_task))
    r2_lst.append(np.mean(r2_lst_task))
    print(f"Batch {test_sample_idx}: MAE {np.mean(mae_lst_task):.4f}, SMAPE: {np.mean(smape_lst_task):.4f} and "
          f"R^2 : {np.mean(r2_lst_task):.4f}")


print(f"Training time is {np.mean(train_time):.2f} s")
print(f"Inference time is {np.mean(infer_time):.2f} s")
print(f"Meta-Test target set MAE: {np.mean(mae_lst):.4f}, SMAPE: {np.mean(smape_lst):.4f} and R^2: {np.mean(r2_lst):.4f}")
