import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from torch.utils.data import Dataset

def get_data(mode, task_name=None, large=False):
    """
    Load and preprocess data for multitask or single task learning.

    Parameters:
    - mode (str): 'multitask' or 'single'.
    - task_name (str): Name of the task (required if mode is 'single').
    - large (bool): Whether to use the large validation dataset.

    Returns:
    - For 'multitask' mode:
        X_train_torch, X_test_torch, target_train, target_test, quantiles
    - For 'single' mode:
        X_train_torch, X_test_torch, y_train_single, y_test_single, quantile_single
    """
    # Define task names
    task_names = [
        'posi_peaktime', 'nega_peaktime', 'arri_time', 'posi_dur',
        'nega_dur', 'posi_pressure', 'nega_pressure', 'posi_impulse'
    ]

    # Define file paths for training and testing data
    base_path = 'data/'

    train_files = {task: f"{base_path}train/{task.replace('_', '_')}.csv" for task in task_names}

    if not large:
        test_files = {task: f"{base_path}test/{task.replace('_', '_')}_test.csv" for task in task_names}
    else:
        test_files = {task: f"{base_path}large/{task.replace('_', '_')}_valid.csv" for task in task_names}

    # Load training and testing data into dictionaries
    df_train = {task: pd.read_csv(train_files[task]) for task in task_names}
    df_test = {task: pd.read_csv(test_files[task]) for task in task_names}

    # Use one of the dataframes for features (assuming features are the same across tasks)
    feature_task = 'nega_dur'  # You can choose any task
    df_train_features = df_train[feature_task]
    df_test_features = df_test[feature_task]

    # Encode 'Status' column if it exists
    if 'Status' in df_train_features.columns:
        le = LabelEncoder()
        df_train_features['Status'] = le.fit_transform(df_train_features['Status'])
        df_test_features['Status'] = le.transform(df_test_features['Status'])

    # Prepare feature matrices by dropping 'ID' and 'Target' columns
    X_train_df = df_train_features.drop(['ID', 'Target'], axis=1)
    X_test_df = df_test_features.drop(['ID', 'Target'], axis=1)

    # Prepare targets for each task
    y_train = {task: df_train[task]['Target'].values for task in task_names}
    y_test = {task: df_test[task]['Target'].values for task in task_names}

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Apply QuantileTransformer to targets
    quantiles = {}
    y_train_normal = {}
    y_test_normal = {}

    for task in task_names:
        quantile = QuantileTransformer(output_distribution='normal', random_state=42)
        y_train_normal[task] = quantile.fit_transform(y_train[task].reshape(-1, 1))
        y_test_normal[task] = quantile.transform(y_test[task].reshape(-1, 1))
        quantiles[task] = quantile

    # Convert features and targets to torch tensors
    X_train_torch = torch.from_numpy(X_train.astype(np.float32))
    X_test_torch = torch.from_numpy(X_test.astype(np.float32))

    for task in task_names:
        y_train_normal[task] = torch.from_numpy(y_train_normal[task].astype(np.float32))
        y_test_normal[task] = torch.from_numpy(y_test_normal[task].astype(np.float32))

    if mode == 'multitask':
        # Concatenate targets along the second dimension
        target_train = torch.cat([y_train_normal[task] for task in task_names], dim=1)
        target_test = torch.cat([y_test_normal[task] for task in task_names], dim=1)
        return X_train_torch, X_test_torch, target_train, target_test, quantiles
    elif mode == 'single':
        if task_name not in task_names:
            raise ValueError(f"Invalid task_name: {task_name}. Must be one of {task_names}")
        y_train_single = y_train_normal[task_name].reshape(-1)
        y_test_single = y_test_normal[task_name].reshape(-1)
        quantile_single = quantiles[task_name]
        return X_train_torch, X_test_torch, y_train_single, y_test_single, quantile_single
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'multitask' or 'single'.")

class BLEVEDataset(Dataset):
    def __init__(self, data, targets):
        """
        Custom Dataset for BLEVE multitask learning.

        Parameters:
        - data (torch.Tensor): Feature data.
        - targets (torch.Tensor): Target data concatenated along the second dimension.
        """
        self.data = data
        self.targets = targets
        self.task_names = [
            'posi_peaktime', 'nega_peaktime', 'arri_time', 'posi_dur',
            'nega_dur', 'posi_pressure', 'nega_pressure', 'posi_impulse'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx]
        targets = self.targets[idx]
        # Create a dictionary mapping task names to target values
        targets_dict = {task: targets[i] for i, task in enumerate(self.task_names)}
        return inputs, targets_dict

class BLEVEDatasetSingle(Dataset):
    def __init__(self, data, target, task_name):
        """
        Custom Dataset for BLEVE single-task learning.

        Parameters:
        - data (torch.Tensor): Feature data.
        - target (torch.Tensor): Target data.
        - task_name (str): Name of the task.
        """
        self.data = data
        self.target = target
        self.task_name = task_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx]
        target = self.target[idx]
        targets_dict = {self.task_name: target}
        return inputs, targets_dict
