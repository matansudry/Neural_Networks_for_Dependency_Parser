import torch


def loss_decision_func(self, device, batch, prints=False):
    words, tags, lens, y = batch
    out = self.model.forward(words.to(device), tags.to(device), lens, device, prints=prints)
    print('out', out.shape) if prints else None
    mask = (y > self.model.y_pad).int()
    print('mask', mask.shape) if prints else None
    print('y', y.shape) if prints else None
    
    flat_out = out[mask == 1.]  # .transpose(2, 1)
    flat_y = y[mask == 1.]
    loss = self.criterion(flat_out.to(device), flat_y.to(device).long())
    
    return loss, flat_y, flat_out, mask, out, y


