from model import *
from data import *
import numpy as np
import torch
from data import CLASSES

batch_size = 16

train_dataset = SolarDataset()
test_dataset = SolarDataset(mode='test')
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              collate_fn=_collate_fn, num_workers=4)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                             collate_fn=_collate_fn, num_workers=4)

model = KWS().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss().cuda()

from tqdm import tqdm
import logging
logging.basicConfig(filename='log.log',level=logging.INFO)

epochs = 100
print_step = 10

for epoch in range(epochs):
    pbar = tqdm(train_dataloader)
    total_loss = 0
    total_sample = 0
    correct = 0
    model.train()
    cur_step = 0
    logging.info("[training]training start")
    for batch in pbar:
        optimizer.zero_grad()
        x, y = batch

        x = x.cuda()
        y = y.cuda()
        
        logits = model(x)
        
        loss = criterion(logits,y)

        total_loss += loss.item()
        total_sample += y.size(0)
        
        pred = logits.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).sum()

        loss.backward()
        optimizer.step()
        cur_step += 1
        pbar.set_description("loss: {}\tacc:{}\t".format(total_loss/cur_step, (correct/total_sample)*100))
        
        if cur_step%print_step==0:
            logging.info("[training]epoch {}\tloss: {}\tacc:{}\t".format(epoch,total_loss/cur_step, (correct/total_sample)*100))
    
    logging.info("[test]test start")
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        total_loss = 0
        total_sample = 0
        correct = 0
        cur_step = 0
        model.eval()
        for batch in pbar:
            x, y = batch

            x = x.cuda()
            y = y.cuda()

            logits = model(x)

            loss = criterion(logits,y)

            total_loss += loss.item()
            total_sample += y.size(0)

            pred = logits.data.max(1, keepdim=True)[1]
            cur_step += 1
            correct += pred.eq(y.data.view_as(pred)).sum()
    print("[test]epoch {}\tloss: {}\tacc:{}\t".format(epoch,total_loss/cur_step, (correct/total_sample)*100))
    logging.info("[test]epoch {}\tloss: {}\tacc:{}\t".format(epoch,total_loss/cur_step, (correct/total_sample)*100))
