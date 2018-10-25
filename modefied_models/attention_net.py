import torch


class Attnet(torch.nn.Module):
    def __init__(self, feat_in):
        super(Attnet, self).__init__()
        self.attnet = torch.nn.Linear(feat_in, 1)
        self.softmax = torch.nn.Softmax(dim=0)      

    def forward(self, x):
        aweight = self.attnet(x.detach())
        return self.softmax(aweight), aweight


class Attnet1(torch.nn.Module):
    def __init__(self, feat_in):
        super(Attnet1, self).__init__()
        self.attnet = torch.nn.Linear(feat_in, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        aweight = self.attnet(x.detach())
        sw = self.sigmoid(aweight)

        return sw/sw.sum()+1e-5, sw
