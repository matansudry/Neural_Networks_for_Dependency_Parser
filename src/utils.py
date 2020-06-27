import torch
import torch.nn as nn


def loss_decision_func(self, device, batch, prints=False):
    words, tags, lens, y = batch
    out = self.model.forward(words.to(device), tags.to(device), lens, device, prints=prints)
    print('out', out.shape) if prints else None
    mask = (y > self.model.y_pad).int()
    print('mask', mask.shape) if prints else None
    print('y', y.shape) if prints else None

    flat_out = out[mask == 1.]
    flat_y = y[mask == 1.]
    loss = self.criterion(flat_out.to(device), flat_y.to(device).long())

    return loss, flat_y, flat_out, mask, out, y


class AdditiveAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 attention_dim,
                 activation=nn.Tanh(),
                 bias=True,
                 include_root=False,
                 **kwargs):

        super(AdditiveAttention, self).__init__()

        self.include_root = include_root

        self.mlp1 = nn.Linear(input_dim, attention_dim, bias=bias)
        self.mlp2 = nn.Linear(input_dim, attention_dim, bias=bias)

        self.activation = activation

        self.out = nn.Linear(attention_dim, 1, bias=bias)

    def forward(self, x, prints=False):
        print('AdditiveAttention input', x.shape) if prints else None
        # [batch_size, max_len, input_dim]

        x1 = self.mlp1(x)
        print('mlp1', x1.shape) if prints else None
        # [batch_size, max_len, attention_dim]

        x2 = self.mlp2(x)
        print('mlp2', x2.shape) if prints else None
        # [batch_size, max_len, attention_dim]

        x1 = x1.unsqueeze(1)
        print('unsqueeze_x1', x1.shape) if prints else None
        # [batch_size, 1, max_len, attention_dim]

        x2 = x2.unsqueeze(2)
        print('unsqueeze_x2', x2.shape) if prints else None
        # [batch_size, max_len, 1, attention_dim]

        x = x1 + x2
        print('outer_add_x1_x2', x.shape) if prints else None
        # [batch_size, max_len, max_len, attention_dim]

        # del ROOT from modifiers dimention (dim=1)
        if not self.include_root:
            x = x[:, 1:, :, :]
            print('del ROOT', x.shape) if prints else None
            # [batch_size, max_len - 1, max_len, attention_dim]

        x = self.activation(x)
        print('activation', x.shape) if prints else None
        # [batch_size, max_len - 1, max_len, attention_dim]

        x = self.out(x)
        print('out', x.shape) if prints else None
        # [batch_size, max_len - 1, max_len, 1]

        x = x.squeeze(-1)
        print('squeeze', x.shape) if prints else None
        # [batch_size, max_len - 1, max_len]

        return x


class DotAttention(nn.Module):
    def __init__(self,
                 include_root=False,
                 **kwargs):

        super(DotAttention, self).__init__()

        self.include_root = include_root

    def forward(self, x, prints=False):

        print('DotAttention input', x.shape) if prints else None
        # [batch_size, max_len, input_dim]

        x = torch.einsum('sae, sbe -> sab', x, x)
        print('dot attention', x.shape) if prints else None
        # [batch_size, max_len - 1, max_len]

        # del ROOT from modifiers dimention (dim=1)
        if not self.include_root:
            x = x[:, 1:, :]
            print('del ROOT', x.shape) if prints else None
            # [batch_size, max_len - 1, max_len]

        return x


class MultiplicativeAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 bias=True,
                 include_root=False,
                 **kwargs):
        super(MultiplicativeAttention, self).__init__()
        self.include_root = include_root
        self.W = nn.Linear(input_dim, input_dim, bias=bias)

    def forward(self, x, prints=False):
        print('DotAttention input', x.shape) if prints else None
        # [batch_size, max_len, input_dim]

        x1 = self.W(x)
        print('x1 = x.T @ W', x.shape) if prints else None
        # [batch_size, max_len, input_dim]

        x = torch.einsum('sai, sbi -> sab', x1, x)
        print('x1 @ x', x.shape) if prints else None
        # [batch_size, max_len, max_len]

        # del ROOT from modifiers dimention (dim=1)
        if not self.include_root:
            x = x[:, 1:, :]
            print('del ROOT', x.shape) if prints else None
            # [batch_size, max_len - 1, max_len]

        return x

