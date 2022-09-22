import torch

# from model.resnet import model_conv
import torch.nn as nn
import yaml
import os
from torchvision import models


# folder to load config file
CONFIG_PATH = "config"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("config.yaml")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

model_conv = models.__dict__[config["model"]["model_type"]](
    pretrained=config["model"]["pretrained"]
)

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, config["model"]["num_features"])

# model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
# optimizer = torch.optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(
    model_conv.fc.parameters(), lr=config["hyperparams"]["learning_rate"]
)

# Decay LR by a factor of 0.1 every 5 epochs
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config["hyperparams"]["lr_scheduler"]["factor"],
    patience=config["hyperparams"]["lr_scheduler"]["patience"],
)

numEpochs = config["hyperparams"]["num_epochs"]
