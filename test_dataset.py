from easydict import EasyDict
import utils

opt = EasyDict({
    'data_type': 'arad',
    'process_type': 'gen',
    'baseroot_train': '/cbio/projects/046/iviwe/NTIRE/train',
    'baseroot_val': '/cbio/projects/046/iviwe/NTIRE/val',
    'crop_size': 256
})

trainset = utils.create_dataset(opt)
valset = utils.create_dataset_val(opt)

print("Train size:", len(trainset))
print("Val size:", len(valset))

x, y = trainset[0]

print("RGB tensor shape:", x.shape)
print("HS tensor shape:", y.shape)
