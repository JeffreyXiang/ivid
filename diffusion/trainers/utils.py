import io
import torch
import torch.nn as nn
import torch.distributed as dist

def cycle(dl):
    while True:
        for data in dl:
            yield data

def read_file_dist(path):
    """
    Read the binary file distributedly.
    File is only read once by the rank 0 process and broadcasted to other processes.

    Returns:
        data (io.BytesIO): The binary data read from the file.
    """
    # read file
    size = torch.LongTensor(1).cuda()
    if dist.get_rank() == 0:
        with open(path, 'rb') as f:
            data = f.read()
        data = torch.ByteTensor(
            torch.ByteStorage.from_buffer(data)
        ).cuda()
        size[0] = data.shape[0]
    # broadcast size
    dist.broadcast(size, src=0)
    if dist.get_rank() != 0:
        data = torch.ByteTensor(size[0].item()).cuda()
    # broadcast data
    dist.broadcast(data, src=0)
    # convert to io.BytesIO
    data = data.cpu().numpy().tobytes()
    data = io.BytesIO(data)
    return data


# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )
    

def zero_grad(model_params):
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()
            
