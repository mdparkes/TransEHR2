from numpy import ndarray
from torch import Tensor
from typing import Dict, List, Union


EventAssociatedDataEntry = Dict[str, List[List[ndarray]]]
StaticDataEntry = List[ndarray]
ValueAssociatedDataEntry = Dict[str, Dict[str, List[List[ndarray]]]]
TargetDataEntry = Dict[str, ndarray]

# Data types created by preprocessing functions that act on MixedDataset. Used as input to models.
EventAssociatedTensorData = Dict[str, Tensor]
StaticTensorData = Tensor
ValueAssociatedTensorData = Dict[str, Union[Dict[str, Union[Tensor, List[Tensor]]], Tensor]]
TargetTensorData = Dict[str, Tensor]

MixedTensorDataset = Dict[
    str, Union[ValueAssociatedTensorData, EventAssociatedTensorData, StaticTensorData, TargetTensorData]
]
