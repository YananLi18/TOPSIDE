import os
import sys
from uu import encode
import torch
import numpy as np
from torch import nn
from torch import optim

from torchinfo import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision  # torch package for vision related things
import torchvision.transforms as transforms
from typing import Dict, List, Tuple

from time import time
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_dir)
from loss import *

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

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


def pretrain_col_run_validation_iter(data_batch, pretrain_model, device, pretrain_model_bz=32):
    # Input [8, 5, 64, 16, 1, 1]
    # Output loss; two 8-item lists, each item is an array with shape [320, 64]
    x_support_set, x_target_set, y_support_set, y_target_set = data_batch
    losses, per_task_preds, per_task_support_preds = [], [], []
    for x_support_task, x_target_task, y_support_task, y_target_task in zip(x_support_set, x_target_set, y_support_set, y_target_set):
        X_train, X_val = torch.squeeze(x_support_task).view(-1, 16).numpy(), torch.squeeze(x_target_task).view(-1, 16).numpy()
        y_train, y_val = y_support_task.view(-1, ).numpy(), y_target_task.view(-1, ).numpy()

        train_dataset = CustomTcDataset(X_train, y_train)
        val_dataset = CustomTcDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=pretrain_model_bz, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=pretrain_model_bz, shuffle=False)

        train_ret, test_ret = [], []
        pretrain_model.eval()
        with torch.no_grad():
            for batch, (X, _) in tqdm(enumerate(train_loader)):
                X = X.to(device)

                # 1. Forward pass
                x_infer = pretrain_model(X)

                train_ret.append(x_infer)
        pred_train = torch.cat(train_ret, dim=0)
        train_pred = pred_train.detach().cpu().numpy()

        pretrain_model.eval()
        with torch.no_grad():
            for batch, (X, _) in tqdm(enumerate(val_loader)):
                X = X.to(device)

                # 1. Forward pass
                x_infer = pretrain_model(X)

                test_ret.append(x_infer)
        pred_test = torch.cat(test_ret, dim=0)
        test_pred = pred_test.detach().cpu().numpy()


        per_task_preds.append(test_pred)
        per_task_support_preds.append(train_pred)
        criterion = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')
        sup_loss = criterion(torch.from_numpy(test_pred).to(device), torch.from_numpy(y_val).unsqueeze(-1).to(device))
        losses.append(sup_loss)
    return torch.mean(torch.stack(losses)), per_task_preds, per_task_support_preds


def nn_pipeline(args, x_support, y_support, x_target, y_target, dic_num, dic_idx):

    if args.bnn_standard_scaler:
        # Split the array
        unscaled_part_support = x_support[:, :6]  # All columns except the last 16
        to_scale_part_support = x_support[:, 6:]  # Only the last 16 columns
        unscaled_part_target = x_target[:, :6]
        to_scale_part_target = x_target[:, 6:]

        # Initialize the StandardScaler
        scaler = StandardScaler()
        # Fit and transform the part to be scaled
        scaled_part_support = scaler.fit_transform(to_scale_part_support)
        scaled_part_target = scaler.transform(to_scale_part_target)

        X_support = np.concatenate((unscaled_part_support, scaled_part_support), axis=1)
        X_target = np.concatenate((unscaled_part_target, scaled_part_target), axis=1)
    else:
        X_support, X_target = x_support, x_target

    train_dataset = CustomDataset(X_support, y_support)
    val_dataset = CustomDataset(X_target, y_target)
    train_loader = DataLoader(train_dataset, batch_size=args.bnn_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bnn_batch_size, shuffle=True)

    dimension = x_support.shape[1]
    device = torch.cuda.current_device()

    base_dim = 64 if args.pretrain_model == 'meta_learning' else 1000
    model = FullNet(args.bnn_collaborative, dic_num, dic_idx, dimension, device, base_dimension=base_dim).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.bnn_learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)
    path = args.work_path + 'baselines/collaborative/tmp/dnn/' + args.label_name + '/' + args.pretrain_model + '/' + args.loss_name + \
        '/' + str(args.corruption_num) + '/' + datetime.now().strftime("%Y_%m_%d/%H_%M_%S/")
    exp = Exp(model, criterion, optimizer, scheduler, path, device)

    start = time()
    results = exp.train(train_dataloader=train_loader, test_dataloader=val_loader, epochs=args.bnn_num_epochs)
    train_time = time() - start

    start = time()
    test_pred = exp.predict(val_loader)
    infer_time = time() - start

    pred_mae = mae(y_target, test_pred)
    pred_smape = smape(y_target, test_pred)
    pred_r2 = r_square(y_target, test_pred)

    return train_time, infer_time, pred_mae, pred_smape, pred_r2


def smape_loss(y_pred, y_true):
    """
    Computes the Symmetric Mean Absolute Percentage Error (SMAPE) loss between the predicted and true values.

    Args:
        y_pred (torch.Tensor): Predicted values of shape (batch_size, ...).
        y_true (torch.Tensor): True values of shape (batch_size, ...).

    Returns:
        torch.Tensor: SMAPE loss.
    """
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2.0
    element_wise_smape = numerator / denominator
    return torch.mean(element_wise_smape).item()


class CustomDataset(Dataset):
    def __init__(self, feature_array, label_array):
        self.feature_array = feature_array
        self.label_array = label_array

    def __len__(self):
        return len(self.feature_array)

    def __getitem__(self, idx):
        return self.feature_array[idx, :], self.label_array[idx, :]  # respectively in shape (H, W, 3) and (1,)


class CustomTcDataset(Dataset):
    def __init__(self, features, labels, flag=True):
        self.features = features  # array
        self.labels = labels  # array
        self.fea_dim = self.features.shape[1]
        self.flag = flag
        if self.flag:
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # 1. Reshape all images to 224x224 (though some models may require different sizes)
                transforms.ToTensor(),  # 2. Turn image values to between 0 & 1
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                                     std=[0.229, 0.224, 0.225])
                # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
            ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        base_fea = self.features[idx, :]
        label = self.labels[idx]
        lst1, lst2, lst3 = [], [], []

        for i in range(self.fea_dim):
            lst1.append(np.roll(base_fea, i))
            lst2.append(np.roll(base_fea, -i-1))
            lst3.append(np.roll(base_fea, self.fea_dim // 2 + i))

        a1, a2, a3 = np.array(lst1), np.array(lst2), np.array(lst3)
        ret_fea = self.preprocess(np.uint8(np.stack((a1, a2, a3), axis=2) * 255)).numpy() if self.flag else np.array([a1, a2, a3])
        return ret_fea, label  # respectively in shape (H, W, 3) and (1,)


class FullNet(nn.Module):
    def __init__(self, bnn_collaborative, dic_num, dic_idx, dimension, device, base_dimension=64):
        super(FullNet, self).__init__()
        '''
        How to set embedding size?
        https://forums.fast.ai/t/embedding-layer-size-rule/50691
        '''

        self.emb1 = nn.Embedding(dic_num['date_sequence'], int(base_dimension/2))
        self.emb2 = nn.Embedding(dic_num['hour_sequence'], int(base_dimension/2))
        self.emb3 = nn.Embedding(dic_num['domain_name'], base_dimension)
        self.emb4 = nn.Embedding(dic_num['isp'], base_dimension)
        self.emb5 = nn.Embedding(dic_num['node_name'], base_dimension)
        self.emb6 = nn.Embedding(dic_num['city'], base_dimension)

        self.first_linear = nn.Linear(dimension-6, base_dimension)
        self.ratio = 0.15

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(base_dimension*10, base_dimension*5),
            nn.ReLU(),
            nn.Dropout(self.ratio),
            nn.Linear(base_dimension*5, base_dimension*2),
            nn.ReLU(),
            nn.Dropout(self.ratio),
            nn.Linear(base_dimension*2, base_dimension),
            nn.ReLU(),
            nn.Dropout(self.ratio),
            nn.Linear(base_dimension, 1),
            # nn.ReLU()
        )
        self.dic_idx = [dic_idx[c] for c in ['date_sequence', 'hour_sequence', 'domain_name', 'isp', 'node_name', 'city']]
        self.device = device
        self.bnn_collaborative = bnn_collaborative

    def forward(self, x):
        len1 = x.shape[1]

        input_value = torch.index_select(x, 1, torch.tensor([i for i in range(len1) if i not in self.dic_idx]).to(self.device))
        input_emb = torch.index_select(x, 1, torch.tensor(self.dic_idx).to(self.device))

        emb1 = self.emb1(input_emb[:, 0].int())
        emb2 = self.emb2(input_emb[:, 1].int())
        emb3 = self.emb3(input_emb[:, 2].int())
        emb4 = self.emb4(input_emb[:, 3].int())
        emb5 = self.emb5(input_emb[:, 4].int())
        emb6 = self.emb5(input_emb[:, 5].int())
        emb12 = torch.cat([emb1, emb2], 1)

        iv = self.first_linear(input_value.float()) if not self.bnn_collaborative else input_value.float()

        input_revise = torch.cat([emb12*iv, emb3*iv, emb4*iv, emb5*iv, emb6*iv,
                                  emb12, emb3, emb4, emb5, emb6], 1)

        y = self.linear_relu_stack(input_revise)
        return y


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'checkpoint.pth')
        self.val_loss_min = val_loss


class Exp(object):

    def __init__(self, model, loss_fn, optimizer, scheduler, path, device):
        self.model = model
        self.path = path
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        if not os.path.exists(path):
            os.makedirs(path)

    def train_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """ Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
        """
        # Put model in train mode
        self.model.train()

        # Setup train loss and train accuracy values
        train_loss, train_smape = 0, 0

        # Loop through data loader data batches
        for batch, (X, y) in tqdm(enumerate(dataloader)):
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)

            # 1. Forward pass
            y_pred = self.model(X)

            # 2. Calculate  and accumulate loss
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()

            self.scheduler.step()

            # Calculate and accumulate accuracy metric across all batches
            # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            # train_acc += (y_pred_class == y).sum().item()/len(y_pred)
            train_smape += smape_loss(y_pred, y)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_smape = train_smape / len(dataloader)
        return train_loss, train_smape

    def test_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to "eval" mode and then performs
        a forward pass on a testing dataset.

        Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
        """
        # Put model in eval mode
        self.model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_smape = 0, 0

        # Turn on inference context manager
        with torch.no_grad():
            # Loop through DataLoader batches
            for batch, (X, y) in tqdm(enumerate(dataloader)):
                # Send data to target device
                X, y = X.to(self.device), y.to(self.device)

                # 1. Forward pass
                y_pred = self.model(X)

                # 2. Calculate and accumulate loss
                loss = self.loss_fn(y_pred, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                # test_pred_labels = test_pred_logits.argmax(dim=1)
                # test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                test_smape += smape_loss(y_pred, y)

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_smape = test_smape / len(dataloader)
        return test_loss, test_smape

    def train(self, train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              epochs: int) -> Dict[str, List]:
        """Trains and tests a PyTorch model.

        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                      train_acc: [...],
                      test_loss: [...],
                      test_acc: [...]}
        For example if training for epochs=2:
                     {train_loss: [2.0616, 1.0537],
                      train_acc: [0.3945, 0.3945],
                      test_loss: [1.2641, 1.5706],
                      test_acc: [0.3400, 0.2973]}
        """
        # Create empty results dictionary
        results = {"train_loss": [], "train_smape": [], "test_loss": [], "test_smape": []}
        early_stopping = EarlyStopping(patience=50, verbose=True)
        # Loop through training and testing steps for a number of epochs
        for epoch in range(epochs):
            start1 = datetime.now()
            train_loss, train_smape = self.train_step(dataloader=train_dataloader)
            start2 = datetime.now()
            test_loss, test_smape = self.test_step(dataloader=test_dataloader)
            start3 = datetime.now()

            # print(
            #   f"Epoch: {epoch+1} | "
            #   f"train_loss: {train_loss:.4f} | "
            #   f"train_smape: {train_smape:.4f} | "
            #   f"train_time: {start2-start1} | "
            #   f"test_loss: {test_loss:.4f} | "
            #   f"test_smape: {test_smape:.4f} | "
            #   f"test_time: {start3 - start2} | "
            # )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_smape"].append(train_smape)
            results["test_loss"].append(test_loss)
            results["test_smape"].append(test_smape)
            early_stopping(test_loss, self.model, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = self.path + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return results

    def predict(self, test_dataloader: torch.utils.data.DataLoader):
        ret = []

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch, (X, _) in tqdm(enumerate(test_dataloader)):
                X = X.to(self.device)

                # 1. Forward pass
                test_pred = self.model(X)

                ret.append(test_pred)
        pred_tensor = torch.cat(ret, dim=0)
        return pred_tensor.detach().cpu().numpy()
