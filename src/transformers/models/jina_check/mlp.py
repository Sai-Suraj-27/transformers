import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        bias1=True,
        bias2=True,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else in_features * 4
        )
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(
            hidden_features, out_features, bias=bias2, **factory_kwargs
        )

    def forward(self, x, adapter_mask=None):
        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            fc1_dtype = next(self.fc1.parameters()).dtype
            y = torch.empty(
                *x.shape[:-1], self.fc1.out_features, dtype=fc1_dtype, device=x.device
            )
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_tensor = x[task_indices]
                task_y = self.fc1(task_tensor, task_id=task_id)
                y[task_indices] = task_y
        else:
            y = self.fc1(x)

        y = self.activation(y)

        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            fc2_dtype = next(self.fc2.parameters()).dtype
            out = torch.empty(
                *y.shape[:-1], self.fc2.out_features, dtype=fc2_dtype, device=y.device
            )
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_tensor = y[task_indices]
                task_out = self.fc2(task_tensor, task_id=task_id)
                out[task_indices] = task_out
        else:
            out = self.fc2(y)

        return out if not self.return_residual else (out, x)
