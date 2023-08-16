import torch
import random
import numpy as np
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')


class InceptionModule(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(InceptionModule, self).__init__()
        
        self.bottleneck = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv10 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv20 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv40 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        self.conv01 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.bn = torch.nn.BatchNorm1d(
            num_features=4 * filters
        )

    def forward(self, x):
        x0 = self.bottleneck(x)
        x1 = self.conv10(x0)
        x2 = self.conv20(x0)
        x3 = self.conv40(x0)
        x4 = self.conv01(self.max_pool(x))
        y = torch.concat([x1, x2, x3, x4], dim=1)
        y = torch.nn.functional.relu(self.bn(y))
        return y


class InceptionBlock(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(InceptionBlock, self).__init__()
        
        self.module1 = InceptionModule(
            input_size=input_size,
            filters=filters,
        )
        
        self.module2 = InceptionModule(
            input_size=4 * filters,
            filters=filters,
        )
        
        self.module3 = InceptionModule(
            input_size=4 * filters,
            filters=filters,
        )

        self.conv = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.bn = torch.nn.BatchNorm1d(
            num_features=4 * filters
        )
    
    def forward(self, x):
        y = self.module1(x)
        y = self.module2(y)
        y = self.module3(y)
        y = y + self.bn(self.conv(x))
        y = torch.nn.functional.relu(y)
        return y


class Processor(torch.nn.Module):
    def __init__(self, mu, sigma):
        super(Processor, self).__init__()
        self.mu = torch.nn.Parameter(data=torch.tensor(mu).float(), requires_grad=False)
        self.sigma = torch.nn.Parameter(data=torch.tensor(sigma).float(), requires_grad=False)
    
    def forward(self, x):
        with torch.no_grad():
            return (x - self.mu) / self.sigma


class Lambda(torch.nn.Module):
    
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)


class InceptionModel(torch.nn.Module):
    def __init__(self, input_size, num_classes, filters, depth, mu, sigma, seed):
        super(InceptionModel, self).__init__()
        
        modules = OrderedDict()
        modules['processor'] = Processor(mu, sigma)
        for i in range(depth):
            modules[f'block_{i}'] = InceptionBlock(
                input_size=input_size if i == 0 else 4 * filters,
                filters=filters,
            )
        modules['avg_pool'] = Lambda(f=lambda x: torch.mean(x, dim=-1))
        modules['linear'] = torch.nn.Linear(in_features=4 * filters, out_features=num_classes)
        self.model = init_params(model=torch.nn.Sequential(modules), seed=seed)
    
    def forward(self, x):
        return self.model(x)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    
def glorot_uniform(param):
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(param)
    return torch.nn.init.uniform_(param, a=-np.sqrt(6 / (fan_in + fan_out)), b=np.sqrt(6 / (fan_in + fan_out)))


def init_params(model, seed):
    set_seed(seed)
    state_dict = model.state_dict()
    for param in state_dict:
        if "Conv1d" in param or "linear" in param:
            if "weight" in param:
                state_dict[param] = glorot_uniform(state_dict[param])
            elif "bias" in param:
                state_dict[param] = torch.nn.init.zeros_(state_dict[param])
    model.load_state_dict(state_dict)
    return model
