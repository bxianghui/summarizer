import torch
from torch import nn


class SimpleCCELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self._loss = nn.NLLLoss(reduction = 'none')
        self.device = device
    
    
    def forward(self, y_pred, y_true):
        # mask = [batch_size, seq_len, 1]
        y_mask = torch.where(y_true > 1., torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device)).unsqueeze(-1)
        # [batch_size, seq_len, 1] -> [batch_size, seq_len - 1]
        y_mask = y_mask[:, 1:, 0]
        
        # [batch_size, seq_len, vocab_size] -> [batch_size, seq_len-1, vocab_size] -> [batch_size * (seq_len-1), vocab_size]
        y_pred = y_pred[:, :-1, :]
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1], -1))
        # [batch_size, seq_len] -> [batch_size, seq_len -1] -> [batch_size * (seq_len - 1), ]
        y_true = y_true[:, 1:]
        y_true = y_true.flatten()

        # [batch_size * (seq_len - 1), ]
        loss = self._loss(y_pred, y_true)
        # [batch_size, seq_len - 1]
        loss = loss.reshape(y_mask.shape)
        loss *= y_mask
        # just one number
        loss = torch.sum(loss) / torch.sum(y_mask)
        return loss
        
