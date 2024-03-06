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
from lightgbm import LGBMRegressor
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
dataset_df = pd.read_csv(args.work_path + 'datasets/ICC2018_cleaned.csv')
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

lg = LGBMRegressor(objective='mean_absolute_error',
                   boosting_type='dart',
                   # feature_name=df_features.columns.to_list(),
                   # categorical_feature='2,3,4,5,143',
                   learning_rate=0.01,
                   bagging_seed=train_seed,
                   feature_fraction_seed=train_seed,
                   data_random_seed=train_seed,
                   n_jobs=5,
                   verbose=-1,
                   # device="gpu",
                   # gpu_platform_id=0,
                   # gpu_device_id=0,
                   # gpu_use_dp=False,
                   random_state=train_seed)
cv = KFold(n_splits=5, shuffle=False)

if meta_cv:
    param_dist = {
        "max_depth": [5, 10, 25, 50, 100],
        "max_bin": [255],
        "n_estimators": [10, 50, 100, 500, 1000, 2500, 3000],
    }
    random_search = RandomizedSearchCV(
        lg, param_distributions=param_dist,
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
    lg_best = LGBMRegressor(objective='mean_absolute_error',
                            boosting_type='dart',
                            max_depth=random_search.best_params_["max_depth"],
                            max_bin=random_search.best_params_["max_bin"],
                            n_estimators=random_search.best_params_["n_estimators"],
                            learning_rate=0.01,
                            bagging_seed=test_seed,
                            feature_fraction_seed=test_seed,
                            data_random_seed=test_seed,
                            verbose=-1,
                            random_state=test_seed)
else:
    print("Skip meta training for searching best parameters: ")
    lg_best = LGBMRegressor(objective='mean_absolute_error',
                            boosting_type='dart',
                            # feature_name=df_features.columns.to_list(),
                            # categorical_feature='2,3,4,5,143',
                            max_depth=5,
                            max_bin=255,
                            n_estimators=10,
                            learning_rate=0.01,
                            bagging_seed=test_seed,
                            feature_fraction_seed=test_seed,
                            data_random_seed=test_seed,
                            n_jobs=5,
                            verbose=-1,
                            # device="gpu",
                            # gpu_platform_id=0,
                            # gpu_device_id=0,
                            # gpu_use_dp=False,
                            random_state=test_seed)

del train_features, train_labels

MLDataLoader = MetaLearningSystemDataLoader(args=args)
train_time, infer_time = [], []
mae_lst, smape_lst, r2_lst = [], [], []
for test_sample_idx, test_sample in enumerate(MLDataLoader.get_test_batches(total_batches=int(args.num_evaluation_tasks / args.batch_size),
                                              augment_images=False)):
    x_support_set, x_target_set, y_support_set, y_target_set, _ = test_sample # x_support_set in shape [8, 5, 64, 16, 1, 1]; y_support_set in shape [8, 5, 64]; target_set is same.
    train_time_task, infer_time_task = [], []
    mae_lst_task, smape_lst_task, r2_lst_task = [], [], []
    for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(
            zip(x_support_set, y_support_set, x_target_set, y_target_set)):
        x_support, x_target = torch.squeeze(x_support_set_task).view(-1, args.image_channels).numpy(), torch.squeeze(x_target_set_task).view(-1, args.image_channels).numpy() # x_support and x_target in shape [320, 16]
        y_support, y_target = y_support_set_task.view(-1, 1).numpy(), y_target_set_task.view(-1, ).numpy() # y_support in shape [320, 1]; y_target in shape [320, ];

        start1 = time()
        lg_best.fit(x_support, y_support)
        end1 = time()
        test_pred = lg_best.predict(x_target).reshape(-1)
        end2 = time()
        pred_mae, pred_smape, pred_r2 = mae(y_target, test_pred), smape(y_target, test_pred), r_square(y_target, test_pred)
        train_time_task.append(end1 - start1)
        infer_time_task.append(end2 - end1)
        mae_lst_task.append(pred_mae)
        smape_lst_task.append(pred_smape)
        r2_lst_task.append(pred_r2)
    train_time.append(np.mean(train_time_task))
    infer_time.append(np.mean(infer_time_task))
    mae_lst.append(np.mean(mae_lst_task))
    smape_lst.append(np.mean(smape_lst_task))
    r2_lst.append(np.mean(r2_lst_task))

    # print(f"Batch {test_sample_idx}: training time is {end1 - start1:.2f} s")
    # print(f"Batch {test_sample_idx}: inference time is {end2 - end1:.2f} s")
    print(f"Batch {test_sample_idx}: MAE {np.mean(mae_lst_task):.4f}, SMAPE: {np.mean(smape_lst_task):.4f} and "
          f"R^2 : {np.mean(r2_lst_task):.4f}")


print(f"Training time is {np.mean(train_time):.2f} s")
print(f"Inference time is {np.mean(infer_time):.2f} s")
print(f"Meta-Test target set MAE: {np.mean(mae_lst):.4f}, SMAPE: {np.mean(smape_lst):.4f} and R^2: {np.mean(r2_lst):.4f}")
