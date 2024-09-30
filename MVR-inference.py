import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from LibMTL.utils import set_random_seed, set_device
from LibMTL import Trainer
from LibMTL.architecture import HPS
from LibMTL_Adapt.data_processing import get_data, BLEVEDataset
from LibMTL_Adapt.metrics import HuberLoss, BLEVEMetrics

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for MVR BLEVE Blast Prediction')
    parser.add_argument('--large_data', action='store_true', default=False, help='Use large dataset')
    parser.add_argument('--testing_size', type=int, default=7200, help='Size of testing data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weighting', default='EW', type=str, help='Weighting method')
    parser.add_argument('--rep_grad', action='store_true', default=False, help='Use rep_grad')
    parser.add_argument('--scheduler', default=None, type=str, help='Scheduler parameter')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--input_dim', type=int, default=11, help='Number of input features (11 for BLEVE-Open, 20 for BLEVE-Obstacle)')
    return parser.parse_args()

def main(params):
    print(f"Inference parameters: {params}")

    # Set random seed and device
    set_random_seed(params.seed)
    set_device(params.gpu_id)

    # Load data (only test data needed for inference)
    _, X_test_torch, _, target_test, quantiles = get_data(
        mode='multi', input_dim=params.input_dim, large=params.large_data
    )

    # Limit testing data size if necessary
    X_test_torch = X_test_torch[:params.testing_size]
    target_test = target_test[:params.testing_size]

    # Create test dataset
    data_test = BLEVEDataset(X_test_torch, target_test)

    # Create test data loader
    test_loader = DataLoader(data_test, batch_size=len(X_test_torch), shuffle=False)

    # List of task names
    task_names = ['posi_peaktime', 'nega_peaktime', 'arri_time', 'posi_dur',
                  'nega_dur', 'posi_pressure', 'nega_pressure']

    # Define task dictionary
    task_dict = {
        task: {
            'metrics': ['R2', 'MAPE', 'RMSE'],
            'metrics_fn': BLEVEMetrics(quantiles[task]),
            'loss_fn': HuberLoss(),
            'weight': [1, 0, 0]
        } for task in task_names
    }

    # Number of output channels for each task
    num_out_channels = {task: 1 for task in task_names}

    # Input dimension
    input_dim = params.input_dim

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

    # Define decoder (output layer) with 7 outputs
    class Decoder(nn.Module):
        def __init__(self, output_dim):
            super(Decoder, self).__init__()
            self.output_layer = nn.Linear(256, output_dim)

        def forward(self, x):
            return self.output_layer(x)

    # Create a single decoder for all tasks
    decoders = nn.ModuleDict({
        'shared_decoder': Decoder(output_dim=7)
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

    # Modify the forward function to handle multiple outputs
    def custom_forward(self, x):
        rep = self.encoder(x)
        outputs = self.decoders['shared_decoder'](rep)
        # Split outputs for each task
        task_outputs = {}
        for i, task in enumerate(task_names):
            task_outputs[task] = outputs[:, i].unsqueeze(1)
        return task_outputs

    # Bind the custom forward function to the Trainer
    BLEVENet.model_forward = custom_forward.__get__(BLEVENet)

    # Load the trained model weights
    BLEVENet.load_model('mvr_model.pt')

    # Perform inference
    BLEVENet.test(test_loader, mode=None, inference=True)

if __name__ == "__main__":
    params = parse_args()
    main(params)