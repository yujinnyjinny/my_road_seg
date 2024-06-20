import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
import wandb
from utils.kari_road_dataset import KariRoadDataset
#from utils.metrics import ConfusionMatrix
#from utils.loss import ce_loss
from utils.plots import plot_image

def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name
    # tensorboard settings
    log_dir = Path('logs')/name
    wandb.init(id=opt.name, resume='allow')
    wandb.config.update(opt)
    # Augmentation
    train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

    # Train dataloader
    num_workers = min([min([os.cpu_count(), 32]), batch_size])  
    train_dataset = KariRoadDataset('./data/kari-road', train=True)                
    # Validation dataset
    val_dataset = KariRoadDataset('./data/kari-road', train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, drop_last=True)
    
    # Network model
    model = MyVGG1()

    # GPU-support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:   # multi-GPU
       model = torch.nn.DataParallel(model)
    model.to(device)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # AMP
    if torch.cuda.is_available() and opt.amp == True:
        scaler = torch.cuda.amp.GradScaler()
        print('[AMP Enabled]')
    else:
        scaler = None

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # loading a weight file (if exists)
    weight_file = Path('weights')/(name + '.pth')
    best_accuracy = 0.0
    start_epoch, end_epoch = (0, epochs)
    if os.path.exists(weight_file):
        checkpoint = torch.load(weight_file)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print('resumed from epoch %d' % start_epoch)

    # training/validation
    for epoch in range(start_epoch, end_epoch):
        print('epoch: %d/%d' % (epoch, end_epoch-1))
        t0 = time.time()
        epoch_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device, scaler)
        t1 = time.time()
        print('loss=%.4f (took %.2f sec)' % (epoch_loss, t1-t0))
        lr_scheduler.step()
        # validation
        val_epoch_loss, accuracy = val_one_epoch(val_dataloader, model, loss_fn, device)
        print('[validation] loss=%.4f, accuracy=%.4f' % (val_epoch_loss, accuracy))
        # saving the best status into a weight file
        if accuracy > best_accuracy:
            best_weight_file = Path('weights')/(name + '_best.pth')
            best_accuracy = accuracy
            state = {'model': model.state_dict(), 'epoch': epoch, 'best_accuracy': best_accuracy}
            torch.save(state, best_weight_file)
            print('best accuracy=>saved\n')
        # saving the current status into a weight file
        state = {'model': model.state_dict(), 'epoch': epoch, 'best_accuracy': best_accuracy}
        torch.save(state, weight_file)
        # tensorboard logging
        wandb.log({'train_epoch_loss': epoch_loss, 'val_epoch_loss': val_epoch_loss, 'val_accuracy': accuracy})

def train_one_epoch(train_dataloader, model, loss_fn, optimizer, device, scaler=None):
    model.train()
    losses = [] 
    for i, (imgs, targets) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()   # zeros the parameter gradients
        if scaler is None:
            preds = model(imgs)     # forward
            loss = loss_fn(preds, targets) # calculates the iteration loss  
            loss.backward()         # backward
            optimizer.step()        # update weights
        else:
            with torch.cuda.amp.autocast():
                preds = model(imgs)     # forward
                loss = loss_fn(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        
        # print the iteration loss every 100 iterations
        if i % 100 == 0:
            print('\t iteration: %d/%d, loss=%.4f' % (i, len(train_dataloader)-1, loss))    
        losses.append(loss.item())
    return torch.tensor(losses).mean().item()


def val_one_epoch(val_dataloader, model, loss_fn, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    for i, (imgs, targets) in enumerate(val_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            losses.append(loss.item())
            preds = torch.argmax(preds, axis=1) 
            total += preds.size(0)
            correct += (preds == targets).sum().item()
    accuracy = correct/total
    return torch.tensor(losses).mean().item(), accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='batch size')
    parser.add_argument('--name', default='myvgg11_amp', help='name for the run')
    parser.add_argument('--amp', action='store_true', help='use of amp')

    opt = parser.parse_args()

    train(opt)
