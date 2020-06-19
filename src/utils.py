import torch


def loss_decision_func(self, device, batch, prints=False):
    words, tags, lens, y = batch
    out = self.model.forward(words.to(device), tags.to(device), lens, device, prints=prints)
    mask = (y > self.model.y_pad).int()
    
    flat_out = out.transpose(2, 1)[mask == 1.]
    flat_y = y[mask == 1.]
    loss = self.criterion(flat_out.to(device), flat_y.to(device).long())
    
    return loss, flat_y, flat_out, mask, out, y


