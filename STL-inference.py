import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from LibMTL.utils import set_random_seed, set_device
from LibMTL import Trainer

from LibMTL.architecture import HPS
from LibMTL_Adapt.data_processing import get_data, BLEVEDatasetSingle
from LibMTL_Adapt.metrics import HuberLoss, BLEVEMetrics, MAPE

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for STL BLEVE Blast Prediction')
    parser.add_argument('--aug', action='store_true', default=False, help='Data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str,
                        choices=['trainval', 'train'], help='Training mode')
    parser.add_argument('--name', default='posi_impulse', type=str,
                        choices=['posi_peaktime', 'nega_peaktime', 'arri_time', 'posi_dur',
                                 'nega_dur', 'posi_pressure', 'nega_pressure', 'posi_impulse'],
                        help='Name of the task')
    parser.add_argument('--large_data', action='store_true', default=False,
                        help='Use large dataset')
    parser.add_argument('--testing_size', type=int, default=7200, help='Size of testing data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weighting', default='EW', type=str, help='Weighting method')
    parser.add_argument('--rep_grad', action='store_true', default=False, help='Use rep_grad')
    parser.add_argument('--scheduler', default=None, type=str, help='Scheduler parameter')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    return parser.parse_args()

def main(params):
    print(f"Inference parameters: {params}")

    # Set random seed and device
    set_random_seed(params.seed)
    set_device(params.gpu_id)

    # Load data (only test data needed for inference)
    _, X_test_torch, _, target_test, quantile = get_data(
        mode='single', task_name=params.name, large=params.large_data
    )

    # Limit testing data size if necessary
    X_test_torch = X_test_torch[:params.testing_size]
    target_test = target_test[:params.testing_size]

    # Create test dataset
    data_test = BLEVEDatasetSingle(X_test_torch, target_test, task_name=params.name)

    # Create test data loader
    test_loader = DataLoader(data_test, batch_size=len(X_test_torch), shuffle=False)

    # Define task dictionary
    task_dict = {
        params.name: {
            'metrics': ['R2', 'MAPE', 'RMSE'],
            'metrics_fn': BLEVEMetrics(quantile),
            'loss_fn': HuberLoss(),
            'weight': [1, 0, 0]
        }
    }

    # Number of output channels for the task
    num_out_channels = {params.name: 1}

    # Input dimension
    input_dim = X_test_torch.shape[1]

    # Define encoder class matching the training architecture
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

    # Define decoder (output layer)
    decoders = nn.ModuleDict({
        params.name: nn.Linear(256, num_out_channels[params.name])
    })

    # Define optimizer parameters (required by Trainer, but not used during inference)
    optim_param = {'optim': 'adamw', 'lr': 0.001}

    # Additional arguments
    kwargs = {'arch_args': {'input_dim': input_dim}}

    # Create the Trainer (model)
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

    # Load the trained model weights
    BLEVENet.load_model('model/Single_ArrivalTime_lowest.pt')

    # Perform inference
    BLEVENet.test(test_loader, mode=None, name=params.name, inference=True)

if __name__ == "__main__":
    params = parse_args()
    main(params)
