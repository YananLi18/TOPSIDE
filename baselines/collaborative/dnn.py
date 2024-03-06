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

if __name__ == '__main__':
    freeze_support()  # Fix a multiprocess bug

    # Hyperparameter to be set manually
    os.environ['DATASET_DIR'] = '../../'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    args, device = get_args()
    args.bnn_collaborative = True
    args.bnn_standard_scaler = False
    args.bnn_num_epochs = 100
    args.bnn_learning_rate = 0.01
    args.bnn_batch_size = 64

    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(64, args.image_channels,
                                            args.image_height, args.image_width))
    encoder = get_encoder(args, device)
    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.collaborative_nn_evaluated_test_set(encoder)
