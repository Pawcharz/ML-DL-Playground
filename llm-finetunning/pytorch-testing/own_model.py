from torch import nn
from transformers import ViTForImageClassification

class MyCompositeModel(nn.Module):
  def __init__(self):
    super(MyCompositeModel, self).__init__()
    
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