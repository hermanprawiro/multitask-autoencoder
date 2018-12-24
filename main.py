import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from tensorboardX import SummaryWriter
from torchvision.datasets import MNIST
from utils import AverageMeter, calc_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--max-epoch', type=int, default=3)
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-2)
parser.add_argument('--vanilla', action='store_true', help='Use Linear layer instead of Convolutional layer')
parser.add_argument('--summary-name', type=str, default=None)

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.encoder = nn.Linear(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.input_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, x):
        h = self.encoder(x.flatten(start_dim=1))
        h = F.relu(h)

        logits = self.classifier(h)

        reconstruction = self.decoder(h)
        reconstruction = torch.sigmoid(reconstruction)
        reconstruction = reconstruction.view_as(x)

        return logits, reconstruction

class ConvAutoencoder(nn.Module):
    def __init__(self, hidden_size, num_class):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.encoder1 = nn.Conv2d(1, self.hidden_size // 2, kernel_size=3, stride=2, padding=1) # out: (16, 14, 14)
        self.encoder2 = nn.Conv2d(self.hidden_size // 2, self.hidden_size, kernel_size=3, stride=2, padding=1) # out: (32, 7, 7)
        self.decoder1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 2, kernel_size=4, stride=2, padding=1) # out: (16, 14, 14)
        self.decoder2 = nn.ConvTranspose2d(self.hidden_size // 2, 1, kernel_size=4, stride=2, padding=1) # out: (1, 28, 28)
        
        self.classifier = nn.Conv2d(self.hidden_size, self.num_class, kernel_size=7)

    def forward(self, x):
        h = self.encoder1(x)
        h = F.relu(h)
        h = self.encoder2(h)
        h = F.relu(h)

        logits = self.classifier(h)
        logits = logits.flatten(start_dim=1)

        reconstruction = self.decoder1(h)
        reconstruction = F.relu(reconstruction)
        reconstruction = self.decoder2(reconstruction)
        reconstruction = torch.sigmoid(reconstruction)

        return logits, reconstruction

def main():
    args = parser.parse_args()

    max_epoch = args.max_epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    summary_name = './runs/{}'.format(args.summary_name) if not args.summary_name == None else None

    writer = SummaryWriter(log_dir=summary_name, comment='epoch={} batch={} lr={}'.format(max_epoch, batch_size, learning_rate))

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(45, translate=(0.1, 0.1), scale=(0.5, 2.0)),
        torchvision.transforms.ToTensor()
    ])

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    trainset = MNIST('./data/mnist', train=True, transform=transform_train)
    testset = MNIST('./data/mnist', train=False, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    if args.vanilla:
        model = Autoencoder(784, 128, 10)
    else:
        model = ConvAutoencoder(16, 10)

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    # optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    criterion = (classification_criterion, reconstruction_criterion)

    for epoch in range(max_epoch):
        train(train_loader, model, optimizer, criterion, writer, epoch)
        validate(test_loader, model, optimizer, criterion, writer, epoch)


def visualize(inputs, outputs, writer, train=True, step=0):
    inputs_grid = torchvision.utils.make_grid(inputs)
    outputs_grid = torchvision.utils.make_grid(outputs)

    writer.add_image('{}/input'.format('train' if train else 'val'), inputs_grid, step)
    writer.add_image('{}/output'.format('train' if train else 'val'), outputs_grid, step)

def train(loader, model, optimizer, criterion, writer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    total_iter = len(loader)
    end = time.time()
    for i, (inputs, targets) in enumerate(loader):
        logits, reconstruction = model(inputs)

        classification_criterion, reconstruction_criterion = criterion
        classification_loss = classification_criterion(logits, targets)
        reconstruction_loss = reconstruction_criterion(reconstruction, inputs)
        total_loss = classification_loss + reconstruction_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        batch_size = inputs.size(0)
        prec1, prec5 = calc_accuracy(logits, targets, topk=(1, 5))
        losses.update(total_loss.item(), batch_size)
        classification_losses.update(classification_loss.item(), batch_size)
        reconstruction_losses.update(reconstruction_loss.item(), batch_size)
        top1.update(prec1[0], batch_size)
        top5.update(prec5[0], batch_size)

        global_step = (epoch * total_iter) + i + 1
        writer.add_scalar('train/loss/total', losses.val, global_step)
        writer.add_scalar('train/loss/cross-entropy', classification_losses.val, global_step)
        writer.add_scalar('train/loss/mse', reconstruction_losses.val, global_step)

        writer.add_scalar('train/accuracy/top1', top1.val, global_step)
        writer.add_scalar('train/accuracy/top5', top5.val, global_step)

        if i % 50 == 0:
            print('Epoch {0} [{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'C Loss {closs.val:.4f} ({closs.avg:.4f})\t'
                'R Loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, i + 1, total_iter, 
                batch_time=batch_time, data_time=data_time, loss=losses,
                closs=classification_losses, rloss=reconstruction_losses,
                top1=top1, top5=top5)
            )

            visualize(inputs, reconstruction, writer, train=True, step=global_step)

@torch.no_grad()
def validate(loader, model, optimizer, criterion, writer, epoch=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    total_iter = len(loader)
    end = time.time()
    for i, (inputs, targets) in enumerate(loader):
        logits, reconstruction = model(inputs)

        classification_criterion, reconstruction_criterion = criterion
        classification_loss = classification_criterion(logits, targets)
        reconstruction_loss = reconstruction_criterion(reconstruction, inputs)
        total_loss = classification_loss + reconstruction_loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        batch_size = inputs.size(0)
        prec1, prec5 = calc_accuracy(logits, targets, topk=(1, 5))
        losses.update(total_loss.item(), batch_size)
        classification_losses.update(classification_loss.item(), batch_size)
        reconstruction_losses.update(reconstruction_loss.item(), batch_size)
        top1.update(prec1[0], batch_size)
        top5.update(prec5[0], batch_size)

        global_step = (epoch * total_iter) + i + 1
        writer.add_scalar('val/loss/total', losses.val, global_step)
        writer.add_scalar('val/loss/cross-entropy', classification_losses.val, global_step)
        writer.add_scalar('val/loss/mse', reconstruction_losses.val, global_step)

        writer.add_scalar('val/accuracy/top1', top1.val, global_step)
        writer.add_scalar('val/accuracy/top5', top5.val, global_step)

        if i % 50 == 0:
            print('Test [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'C Loss {closs.val:.4f} ({closs.avg:.4f})\t'
                'R Loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i + 1, total_iter, 
                batch_time=batch_time, data_time=data_time, loss=losses,
                closs=classification_losses, rloss=reconstruction_losses,
                top1=top1, top5=top5)
            )

            visualize(inputs, reconstruction, writer, train=False, step=global_step)
    print('***\t'
        'Loss {loss.avg:.4f}\t'
        'Prec@1 {top1.avg:.3f}\t'
        'Prec@5 {top5.avg:.3f}'.format(
            loss=losses, top1=top1, top5=top5
        )
    )

if __name__ == "__main__":
    main()