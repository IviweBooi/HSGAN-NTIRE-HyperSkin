from easydict import EasyDict
import torch
import utils

opt = EasyDict({
    'network_type': 'hsgan',
    'in_channels': 3,
    'out_channels': 31,
    'start_channels': 64,
    'latent_channels': 16,
    'pad': 'reflect',
    'activ': 'lrelu',
    'norm': 'none',
    'init_type': 'xavier',
    'init_gain': 0.02,
    'gpu_ids': '0',
    'multi_gpu': False,
    'load_name': ''
})

model = utils.create_generator(opt)

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
