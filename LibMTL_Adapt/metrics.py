import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

class HuberLoss(nn.Module):
    """Huber Loss function."""

    def __init__(self):
        super(HuberLoss, self).__init__()
        self.loss_fn = nn.HuberLoss()

    def forward(self, pred, gt):
        """
        Compute the Huber loss between predictions and ground truth.

        Args:
            pred (torch.Tensor): Predicted values.
            gt (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed Huber loss.
        """
        # Ensure gt is the same shape as pred
        gt = gt.view_as(pred)
        loss = self.loss_fn(pred, gt)
        return loss

class MAPE:
    """Calculate the Mean Absolute Percentage Error (MAPE)."""

    def __init__(self, quantile):
        """
        Initialize the MAPE metric.

        Args:
            quantile (QuantileTransformer): Transformer used to inverse-transform the data.
        """
        self.quantile = quantile
        self.records = []

    def update(self, pred, gt):
        """
        Update the metric with new predictions and ground truth values.

        Args:
            pred (torch.Tensor): Predicted values.
            gt (torch.Tensor): Ground truth values.
        """
        # Inverse transform the predictions and ground truth
        pred_np = self.quantile.inverse_transform(pred.cpu().numpy())
        gt_np = self.quantile.inverse_transform(gt.cpu().numpy().reshape(-1, 1))

        # Compute MAPE
        score = mean_absolute_percentage_error(gt_np, pred_np)
        self.records.append(score)

    def compute(self):
        """
        Compute the average MAPE over all records.

        Returns:
            float: Average MAPE multiplied by 100.
        """
        return np.mean(self.records) * 100

    def reset(self):
        """Reset the stored records."""
        self.records = []

class BLEVEMetrics:
    """Calculate R², MAPE, and RMSE metrics."""

    def __init__(self, quantile):
        """
        Initialize the BLEVE metrics.

        Args:
            quantile (QuantileTransformer): Transformer used to inverse-transform the data.
        """
        self.quantile = quantile
        self.r2_list = []
        self.mape_list = []
        self.rmse_list = []

    def update(self, pred, gt):
        """
        Update the metrics with new predictions and ground truth values.

        Args:
            pred (torch.Tensor): Predicted values.
            gt (torch.Tensor): Ground truth values.
        """
        # Inverse transform the predictions and ground truth
        pred_np = self.quantile.inverse_transform(pred.cpu().numpy())
        gt_np = self.quantile.inverse_transform(gt.cpu().numpy().reshape(-1, 1))

        # Compute R²
        r2 = r2_score(gt_np, pred_np)
        self.r2_list.append(r2)

        # Compute MAPE
        mape = mean_absolute_percentage_error(gt_np, pred_np)
        self.mape_list.append(mape)

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(gt_np, pred_np))
        self.rmse_list.append(rmse)

    def compute(self):
        """
        Compute the average metrics over all records.

        Returns:
            dict: Dictionary containing average R², MAPE, and RMSE.
        """
        return {
            'R2': np.mean(self.r2_list) * 100,
            'MAPE': np.mean(self.mape_list) * 100,
            'RMSE': np.mean(self.rmse_list)
        }

    def reset(self):
        """Reset the stored metrics."""
        self.r2_list = []
        self.mape_list = []
        self.rmse_list = []
