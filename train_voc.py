from torch.optim import Adam

from demo import convertSyncBNtoBN
from fcos import FCOSDetector
from dataset_voc import VOCDataset

import math, time
import torch
from torch.utils.tensorboard import SummaryWriter

train_dataset = VOCDataset('D:/projects_python/_datasets/VOCdevkit_2012/VOC2012', resize_size=[512, 800], split='trainval')
val_dataset = VOCDataset('D:/projects_python/_datasets/VOCdevkit_2007/VOC2007', resize_size=[512, 800], split='trainval')

model = FCOSDetector(mode="training").cuda()
model.load_state_dict(torch.load("./voc2012_512x800_epoch30_loss0.6646.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

BATCH_SIZE = 1
EPOCHS = 30
WARMPUP_STEPS_RATIO = 0.12
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         collate_fn=val_dataset.collate_fn)
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = TOTAL_STEPS * WARMPUP_STEPS_RATIO

GLOBAL_STEPS = 1
LR_INIT = 0.001
LR_END = 0.0001

writer = SummaryWriter(log_dir="./logs")


def lr_func():
    if GLOBAL_STEPS < WARMPUP_STEPS:
        lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
        )
    return float(lr)


model.train()

for epoch in range(EPOCHS):
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        lr = lr_func()
        for param in optimizer.param_groups:
            param['lr'] = lr

        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.backward()
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)

        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f ctr_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e" % \
              (
                  GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0], losses[1], losses[2], cost_time,
                  lr))

        writer.add_scalar("loss/cls_loss", losses[0], global_step=GLOBAL_STEPS)
        writer.add_scalar("loss/ctr_loss", losses[1], global_step=GLOBAL_STEPS)
        writer.add_scalar("loss/reg_loss", losses[2], global_step=GLOBAL_STEPS)
        writer.add_scalar("lr", lr, global_step=GLOBAL_STEPS)

        GLOBAL_STEPS += 1

    torch.save(model.state_dict(), "./voc2012_512x800_epoch%d_loss%.4f.pth" % (epoch + 1, loss.item()))
