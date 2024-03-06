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
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Stats
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p, inv_boxcox
from scipy.stats import boxcox_normmax

# Misc
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
# from sklearn.utils.fixes import loguniform
from sklearn.model_selection import train_test_split
from multiprocess.spawn import freeze_support

import torch
import warnings
warnings.filterwarnings(action="ignore")

# Get the directory two levels up
two_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Append to sys.path
sys.path.append(two_up)
from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args, get_encoder
from utils.storage import build_experiment_folder

if __name__ == '__main__':
    freeze_support()  # Fix a multiprocess bug

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
    dataset_df = pd.read_csv(args.work_path +'datasets/ICC2018_cleaned.csv')
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

    nystroem = Nystroem(random_state=train_seed)
    linear_svr = LinearSVR(random_state=train_seed)
    pipe = Pipeline(steps=[('nystroem', nystroem), ('linear_svr', linear_svr)])
    cv = KFold(n_splits=5, shuffle=False)

    if meta_cv:
        param_dist = {
        "nystroem__gamma": [1e-4],
        "nystroem__n_components": [10, 100, 1e3],
        "linear_svr__C": [1e-1, 1, 10, 100, 1e3],
        # "linear_svr__C": [1e3, 1e4],  # loguniform(1e-1, 1e3),
        "linear_svr__epsilon": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3],  # [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        # "linear_svr__epsilon": [0],  # loguniform(1e-4, 10),
        }

        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
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
        nystroem_b = Nystroem(n_components=random_search.best_params_["nystroem__n_components"],
                          gamma=random_search.best_params_["nystroem__gamma"],
                          random_state=test_seed)
        linear_svr_b = LinearSVR(C=random_search.best_params_["linear_svr__C"],
                             epsilon=random_search.best_params_["linear_svr__epsilon"],
                             random_state=test_seed)
        pipe_best = Pipeline(steps=[('nystroem', nystroem_b), ('linear_svr', linear_svr_b)])
    else:
        print("Skip meta training for searching best parameters: ")
        nystroem_b = Nystroem(n_components=1000,
                              gamma=0.0001,
                              random_state=test_seed)
        linear_svr_b = LinearSVR(C=10,
                             epsilon=100,
                             random_state=test_seed)
        pipe_best = Pipeline(steps=[
            ('nystroem', nystroem_b),
            ('linear_svr', linear_svr_b)])

    del train_features, train_labels


    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(64, args.image_channels,
                                            args.image_height, args.image_width))
    encoder = get_encoder(args, device)
    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.collaborative_evaluated_test_set(pipe_best, encoder)
