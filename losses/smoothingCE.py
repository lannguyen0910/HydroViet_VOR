import torch
import torch.nn as nn


class SmoothingCE(nn.Module):
    def __init__(self, alpha=1e-6, ignore_index=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, outputs, targets):
        batch_size, n_classes = outputs.shape
        # print('Output shape: ', outputs)
        y_hot = torch.zeros(outputs.shape).scatter_(
            1, targets.unsqueeze(1), 1.0)
        # print('Y hot: ', y_hot)
        y_smooth = (1 - self.alpha) * y_hot + self.alpha / n_classes
        func = nn.functional.log_softmax(outputs, 1)
        # print('Func: ', func)
        loss = torch.sum(- y_smooth * func, -1).sum()

        if self.reduction:
            loss /= batch_size

        return loss


# Output shape:  tensor([[0.0534, -1.6113, -0.6135, -0.4439, -0.4907],
#                        [1.2012, -0.7692, -0.3297, -0.0055,  1.0024],
#                        [0.0691, -0.7182,  0.1875, -1.6878,  0.5579]], requires_grad=True)
# Y hot:  tensor([[0., 0., 1., 0., 0.],
#                 [0., 0., 0., 1., 0.],
#                 [0., 0., 1., 0., 0.]])
def testSmoothCELoss():
    loss = SmoothingCE()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print('input: ', input)
    print('target: ', target)
    output = loss(input, target)
    output.backward()

    print(output)
