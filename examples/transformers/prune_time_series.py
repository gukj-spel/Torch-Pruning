import torch
import torch_pruning as tp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import onnx
from onnx2torch import convert
from model.models.Transformer import Transformer
# Load weather data
weather_data = pd.read_csv('examples/transformers/data/weather.csv')
weather_head = pd.read_csv('examples/transformers/data/weather_head.csv')
model_path = "examples/transformers/model/checkpoint.pth"
# class WeatherDataset(Dataset):
#     def __init__(self, data):
#         self.data = torch.tensor(data.values, dtype=torch.float32)
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]

# dataset = WeatherDataset(weather_data)
# dataloader = DataLoader(dataset, batch_size=1)

model = torch.load(model_path)

"""
example_inputs is a sample input tensor used by the pruner to:
1. Trace the model's computation graph
2. Calculate FLOPs and parameters
3. Simulate pruning operations
It's created from the first batch of weather data
"""
# example_inputs = next(iter(dataloader)) 
example_inputs = torch.randn(1,3,224,224)

imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

pruner = tp.pruner.BasePruner(
    model, 
    example_inputs, 
    global_pruning=False,
    importance=imp,
    iterative_steps=1,
    pruning_ratio=0.5,
    output_transform=lambda out: out.sum(),  # Sum all output values for time series
)

for g in pruner.step(interactive=True):
    g.prune()


print(model)
pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
print("Base MACs: %f M, Pruned MACs: %f M"%(base_macs/1e6, pruned_macs/1e6))
print("Base Params: %f M, Pruned Params: %f M"%(base_params/1e6, pruned_params/1e6))