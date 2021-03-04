import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x, y, z):
        embedded_x = self.embedding_net(x)
        embedded_y = self.embedding_net(y)
        embedded_z = self.embedding_net(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

    def get_embedding(self, x):
        return self.embedding_net(x)
