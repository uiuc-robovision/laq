from torch.utils import data as tdata

class LambdaLoader(tdata.Dataset):
    def __init__(self,length,func):
        self.length=length
        self.func=func

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        return self.func(index)

class IterableLambdaLoader(tdata.IterableDataset):
    def __init__(self,func,args=[]):
        self.func=func
        self.args=args

    def __iter__(self):
        return self.func(*self.args)
