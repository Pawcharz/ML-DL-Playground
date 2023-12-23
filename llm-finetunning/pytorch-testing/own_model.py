from torch import nn
from transformers import ViTForImageClassification

class AutoCompositeModel(nn.Module):
  def __init__(self):
    super(AutoCompositeModel, self).__init__()
    
    self.pretrained = ViTForImageClassification.from_pretrained(
      "google/vit-base-patch16-224",
      num_labels=1000
    )
    self.my_new_layers = nn.Sequential(
      nn.LayerNorm(1000),
      nn.Linear(1000, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 2)
    )
  
  def forward(self, x):
    x = self.pretrained(x).logits
    x = self.my_new_layers(x)
    return x
  
  from torch import nn
from transformers import ViTForImageClassification

class CompositeModel(nn.Module):
  def __init__(self, additional_layers = None):
    super(CompositeModel, self).__init__()
    
    self.pretrained = ViTForImageClassification.from_pretrained(
      "google/vit-base-patch16-224",
      num_labels=1000
    )
    
    if additional_layers == None:  
      self.additional_layers = nn.Sequential(
        nn.LayerNorm(1000),
        nn.Linear(1000, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
      )
    else:
      self.additional_layers = additional_layers
  
  def forward(self, x):
    x = self.pretrained(x).logits
    x = self.additional_layers(x)
    return x