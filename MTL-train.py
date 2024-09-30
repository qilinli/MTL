import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from LibMTL.config import LibMTL_args
from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.architecture import HPS, MTAN, MMoE
from LibMTL_Adapt.data_processing import get_data, BLEVEDataset
from LibMTL_Adapt.metrics import HuberLoss, BLEVEMetrics, MAPE

def parse_args():
    parser = argparse.ArgumentParser(description='MTL Training for BLEVE Blast Prediction')
    parser.add_argument('--aug', action='store_true', default=False, help='Data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str,
                        choices=['trainval', 'train'], help='Training mode')
    parser.add_argument('--name', default='MTL', type=str, help='Name for saving the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weighting', default='EqualWeighting', type=str,
                        choices=['EW', 'UW', 'GradNorm'],
                        help='Weighting method')
    parser.add_argument('--architecture', default='SharedBottom', type=str,
                        choices=['HPS', 'MTAN', 'MMoE'],
                        help='Architecture method')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    # Additional arguments for specific weighting methods
    parser.add_argument('--alpha', type=float, default=1.5, help='Alpha parameter for GradNorm')
    # Additional arguments for architectures
    parser.add_argument('--experts', type=int, default=8, help='Number of experts for MMoE')
    parser.add_argument('--expert_dim', nargs='+', type=int, default=[32, 32],
                        help='Expert dimensions for MMoE (list of layer sizes)')
    return parser.parse_args()

def main(params):
    print(f"Training parameters: {params}")
    # Set random seed and device
    set_random_seed(params.seed)
    set_device(params.gpu_id)

    # Load data
    X_train_torch, X_test_torch, target_train, target_test, quantiles = get_data(
        mode='multitask', large=False
    )

    # Create datasets
    data_train = BLEVEDataset(X_train_torch, target_train, sev_tar=False)
    data_test = BLEVEDataset(X_test_torch, target_test, sev_tar=False)

    # Create data loaders
    train_loader = DataLoader(data_train, batch_size=512, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=512, shuffle=False)

    # List of task names
    task_names = ['posi_peaktime', 'nega_peaktime', 'arri_time', 'posi_dur',
                  'nega_dur', 'posi_pressure', 'nega_pressure', 'posi_impulse']

    # Define task dictionary
    task_dict = {
        task: {
            'metrics': ['MAPE'],
            'metrics_fn': MAPE(quantiles[i]),
            'loss_fn': HuberLoss(),
            'weight': [1]
        } for i, task in enumerate(task_names)
    }

    # Number of output channels for each task
    num_out_channels = {task: 1 for task in task_names}

    # Input dimension
    input_dim = X_train_torch.shape[1]

    # Define the encoder class
    class Encoder(nn.Module):
        def __init__(self, input_dim):
            super(Encoder, self).__init__()
            self.shared_layers = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            )

        def forward(self, x):
            return self.shared_layers(x)

    # Define decoders for each task
    decoders = nn.ModuleDict({
        task: nn.Linear(256, num_out_channels[task]) for task in task_names
    })

    # Define optimizer parameters
    optim_param = {'optim': 'adamw', 'lr': params.lr}

    # Set up weighting arguments
    if params.weighting == 'GradNorm':
        kwargs = {'weight_args': {'alpha': params.alpha}}
    else:
        kwargs = {'weight_args': {}}

    # Set up architecture arguments
    if params.architecture == 'MTAN':
        kwargs['arch_args'] = {'input_dim': input_dim}
        architecture = MTAN
    elif params.architecture == 'MMoE':
        kwargs['arch_args'] = {
            'experts': params.experts,
            'expert_dim': params.expert_dim
        }
        architecture = MMoE
    else:
        kwargs['arch_args'] = {}
        architecture = HPS

    # Create the Trainer
    BLEVENet = Trainer(
        task_dict=task_dict,
        weighting=eval(params.weighting),
        architecture=architecture,
        encoder_class=Encoder,
        decoders=decoders,
        rep_grad=False,
        multi_input=False,
        optim_param=optim_param,
        scheduler_param=None,
        **kwargs
    )

    # Train the model
    BLEVENet.train(train_loader, test_loader, epochs=200)

    # Save the trained model
    model_name = f"{params.name}_{params.weighting}_{params.architecture.__name__}.pt"
    BLEVENet.save_model(model_name)

if __name__ == "__main__":
    params = parse_args()
    main(params)
