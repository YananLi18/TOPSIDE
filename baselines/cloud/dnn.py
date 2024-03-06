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
from utils.nn_baseline import *

# Hyperparameter to be set manually
os.environ['DATASET_DIR'] = '../../'
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

args, device = get_args()
seed = args.train_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

args.bnn_batch_size = 64
args.bnn_num_epochs = 50
args.bnn_learning_rate = 0.01
args.bnn_standard_scaler = True

train_rng = np.random.RandomState(seed=args.train_seed)
train_seed = train_rng.randint(1, 999999)
test_rng = np.random.RandomState(seed=args.val_seed)
test_seed = test_rng.randint(1, 999999)
rng = np.random.RandomState(seed=test_seed)
dataset_df = pd.read_csv(args.work_path + 'datasets/ICC2018_cleaned.csv')
idx_lst = dataset_df.columns.to_list()
dic_num, dic_idx = {}, {}
for f in ['node_name', 'NbClients','DASHPolicy','StallLabel', 'ClientResolution']:
    le = LabelEncoder()
    le.fit(dataset_df[f])
    dic1 = {x: le.transform([x])[0] for x in dataset_df[f].unique()}
    dataset_df[f] = dataset_df[f].map(dic1)
    dic_num[f] = len(dataset_df[f].unique())
    dic_idx[f] = idx_lst.index(f)
del dataset_df


MLDataLoader = MetaLearningSystemDataLoader(args=args)
train_time_lst, infer_time_lst = [], []
mae_lst, smape_lst, r2_lst = [], [], []
for test_sample_idx, test_sample in enumerate(MLDataLoader.get_test_batches(total_batches=int(args.num_evaluation_tasks / args.batch_size),
                                              augment_images=False)):
    x_support_set, x_target_set, y_support_set, y_target_set, _ = test_sample # x_support_set in shape [8, 5, 64, 16, 1, 1]; y_support_set in shape [8, 5, 64]; target_set is same.
    train_time_task, infer_time_task = [], []
    mae_lst_task, smape_lst_task, r2_lst_task = [], [], []
    for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(
            zip(x_support_set, y_support_set, x_target_set, y_target_set)):
        feature_idx = args.image_channels + args.num_of_classes - args.start_idx
        x_support, x_target = torch.squeeze(x_support_set_task).view(-1, feature_idx).numpy(), torch.squeeze(x_target_set_task).view(-1, feature_idx).numpy() # x_support and x_target in shape [320, 16]
        y_support, y_target = y_support_set_task.view(-1, 1).numpy(), y_target_set_task.view(-1, 1).numpy() # y_support in shape [320, 1]; y_target in shape [320, ];

        # 输入NN的训练集和测试集，返回训练时间，推理时间，MAE，SMAPE，R^2
        # --contain_categories
        train_time, infer_time, pred_mae, pred_smape, pred_r2 = nn_pipeline(args, x_support, y_support, x_target, y_target, dic_num, dic_idx)

        train_time_task.append(train_time)
        infer_time_task.append(infer_time)
        mae_lst_task.append(pred_mae)
        smape_lst_task.append(pred_smape)
        r2_lst_task.append(pred_r2)
    train_time_lst.append(np.mean(train_time_task))
    infer_time_lst.append(np.mean(infer_time_task))
    mae_lst.append(np.mean(mae_lst_task))
    smape_lst.append(np.mean(smape_lst_task))
    r2_lst.append(np.mean(r2_lst_task))

    # print(f"Batch {test_sample_idx}: training time is {end1 - start1:.2f} s")
    # print(f"Batch {test_sample_idx}: inference time is {end2 - end1:.2f} s")
    print(f"Batch {test_sample_idx}: MAE {np.mean(mae_lst_task):.4f}, SMAPE: {np.mean(smape_lst_task):.4f} and "
          f"R^2 : {np.mean(r2_lst_task):.4f}")


print(f"Training time is {np.mean(train_time_lst):.2f} s")
print(f"Inference time is {np.mean(infer_time_lst):.2f} s")
print(f"Meta-Test target set MAE: {np.mean(mae_lst):.4f}, SMAPE: {np.mean(smape_lst):.4f} and R^2: {np.mean(r2_lst):.4f}")
