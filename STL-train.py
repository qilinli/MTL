import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from LibMTL.config import LibMTL_args
from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device

from LibMTL_Adapt.data_processing import get_data, BLEVEDatasetSingle
from LibMTL_Adapt.model import Encoder
from LibMTL.architecture import HPS
from LibMTL_Adapt.metrics import HuberLoss, MAPE

def parse_args():
    parser = argparse.ArgumentParser(description='STL Training for BLEVE Blast Prediction')
    parser.add_argument('--aug', action='store_true', default=False, help='Data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, choices=['trainval', 'train'], help='Training mode')
    parser.add_argument('--name', default='posi_impulse', type=str,
                        choices=['posi_peaktime', 'nega_peaktime', 'arri_time', 'posi_dur',
                                 'nega_dur', 'posi_pressure', 'nega_pressure', 'posi_impulse'],
                        help='Name of the task')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weighting', default='EW', type=str, help='Weighting method')
    parser.add_argument('--rep_grad', action='store_true', default=False, help='Use rep_grad')
    parser.add_argument('--scheduler', default=None, type=str, help='Scheduler parameter')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    return parser.parse_args()

def main(params):
    print(f"Training parameters: {params}")

    # Set random seed and device
    set_random_seed(params.seed)
    set_device(params.gpu_id)

    # Load data
    X_train_torch, X_test_torch, target_train, target_test, quantile = get_data(
        mode='single', task_name=params.name
    )

    # Create datasets
    data_train = BLEVEDatasetSingle(X_train_torch, target_train, task_name=params.name)
    data_test = BLEVEDatasetSingle(X_test_torch, target_test, task_name=params.name)

    # Create data loaders
    train_loader = DataLoader(data_train, batch_size=512, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=512, shuffle=False)

    # Define task dictionary
    task_dict = {
        params.name: {
            'metrics': ['MAPE'],
            'metrics_fn': MAPE(quantile),
            'loss_fn': HuberLoss(),
            'weight': [1]
        }
    }

    # Number of output channels for the task
    num_out_channels = {params.name: 1}

    # Input dimension
    input_dim = X_train_torch.shape[1]

    # Define encoder class matching the specified architecture
    class Encoder(nn.Module):
        def __init__(self, input_dim):
            super(Encoder, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            )

        def forward(self, x):
            return self.layers(x)

    # Define decoders (output layer)
    decoders = nn.ModuleDict({
        params.name: nn.Linear(256, num_out_channels[params.name])
    })

    # Define optimizer parameters
    optim_param = {'optim': 'adamw', 'lr': 0.001}

    # Additional arguments
    kwargs = {'arch_args': {'input_dim': input_dim}}

    # Create the Trainer
    BLEVENet = Trainer(
        task_dict=task_dict,
        weighting=eval(params.weighting),
        architecture=HPS,
        encoder_class=Encoder,
        decoders=decoders,
        rep_grad=params.rep_grad,
        multi_input=False,
        optim_param=optim_param,
        scheduler_param=params.scheduler,
        **kwargs
    )

    # Train the model
    BLEVENet.train(train_loader, test_loader, epochs=200)

if __name__ == "__main__":
    params = parse_args()
    main(params)
