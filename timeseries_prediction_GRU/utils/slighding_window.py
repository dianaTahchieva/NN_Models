import numpy as np
import torch

class SlighdingWindow():
    
    def __init__(self, data, window_length, device=torch.device('cuda')):
        
        self.data = data
        
        self.window_length = window_length
        
        self.starts = self.generate_sliding_windows()
        
        self.device = device
        self.ndata = len(self.starts)
        
    def generate_sliding_windows(self):
        return np.arange(0, len(self.data) - self.window_length)
    
    def __len__(self):
        return self.ndata
    
    def __getitem__(self, idx):
        
        if (isinstance(idx, slice) or isinstance(idx, np.ndarray) or isinstance(idx, list)):
            #indexes are some kind of list, so loop through them to return dataset elements
            #return torch.tensor([self.data[i: i + self.window_length] for i in self.starts[idx]], device=self.device)
            return np.array([np.expand_dims(self.data[i: i + self.window_length], axis=-1) for i in self.starts[idx]])
        else:
            #assume integer index
            return np.expand_dims(self.data[self.starts[idx]: self.starts[idx] + self.window_length], axis=-1)
    
    
"""    
if __name__ == "__main__":
    
    data = np.random.rand(1000)
    
    dataset = MyDataset(data, 60)
    
    print ("dataset length:", len(dataset))
    
    print (dataset[0].shape)
    print (dataset[0])
    
    print (dataset[0:5].shape)
    print (dataset[0:5])
"""