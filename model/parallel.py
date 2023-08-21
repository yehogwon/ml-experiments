import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from torch.cuda.amp import autocast

# TODO: torch.amp.autocast support
class ParallelWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._parallel_model = nn.DataParallel(model)
    
    @autocast()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._parallel_model(*args, **kwargs)
    
    def __getattr__(self, name: str): 
        try: 
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def __setattr__(self, name: str, val): 
        try: 
            super().__setattr__(name, val)
        except AttributeError:
            setattr(self.model, name, val)
