#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
import os
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import Salicon
import argparse
from pathlib import Path
from shallow_net import CNN
import pickle
torch.backends.cudnn.benchmark = True




parser = argparse.ArgumentParser(
    description="Train a simple CNN for saliency prediction",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"

parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("second-all/logs"), type=Path)
parser.add_argument('--step', default=78, type=int, metavar='N',help='number of iterations to halve LR')
parser.add_argument("--learning-rate", default=3e-2, type=float, help="Learning rate")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument("--batch-size",default=128,type=int,help="Number of images within each mini-batch",)
parser.add_argument("--epochs",default=1,type=int,help="Number of epochs (passes through the entire dataset) to train for",)
parser.add_argument("--val-frequency",default=2,type=int,help="How frequently to test the model on the validation set in number of epochs",)
parser.add_argument("--log-frequency",default=10,type=int,help="How frequently to save logs to tensorboard in number of steps",)
parser.add_argument("--print-frequency",default=10,type=int,help="How frequently to print progress to the command line in number of steps",)
parser.add_argument("-j","--worker-count",default=cpu_count(),type=int,help="Number of worker processes used to load data.",)




if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    #stores the train and val unpickled files into respective variables
    train_dataset= Salicon('train.pkl')
    val_dataset= Salicon('val.pkl')

    #loads the train and test data set as train_loader/test_loader so it is structured in a correct way
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    print(train_loader)

    model = CNN(height=96, width=96, channels=3)
    ## select mseloss since its identical to euclidean distance
    criterion = nn.MSELoss()

    #define the optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                nesterov=True)
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )
    torch.save(model, "model.pkl")


    summary_writer.close()

#creates a trainer class for it to be called in the main function
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,


    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0


    def train(
            self,
            epochs: int,
            val_frequency: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0
    ):
        global iterations
        self.model.train()
        i = 0
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                #output = self.model.forward(batch)
                #print(output.shape)
                #import sys;
                #sys.exit(1)
                logits = self.model.forward(batch)
                #start off with a zero gradient and access the learning rate so that it is halved every x epoch

                ## TASK 10: Compute the backward pass

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                def Adjust_lr_scheduler(optimizer):
                    # lr = init_lr * 0.5
                    scheduler= (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1))
                    # for param_group in optimizer.param_groups:
                    #    param_group['lr'] = param_group['lr'] * 0.5
                    return scheduler
                Adjust_lr_scheduler(self.optimizer).step()

                with torch.no_grad():
                    preds = logits

                loss = self.criterion(preds,labels)
                loss.backward()
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate() # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
            self.model.train()
            if (epoch + 1) % 0.5 == 0:
                torch.save(self.model, "checkpoint_model.pkl")
            i += 1
            print('Epoch ' + str(epoch) + ', Batch ' + str(i) + ' trained.')


    def print_metrics(self, epoch, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )
    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars("loss",{"train": float(loss.item())},self.step)
        self.summary_writer.add_scalar("time/data", data_load_time, self.step)
        self.summary_writer.add_scalar("time/data", step_time, self.step)

    def validate(self):
        results = {"preds": [], "gts": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.test_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += float(loss.item())
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["gts"].extend(list(labels.cpu().numpy()))
        print("pickling the files now")
        print("the files have been pickled")
        average_loss = total_loss / len(self.test_loader)


        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}")



def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}' \
                        f'_lr={args.learning_rate}' \
                        f'_momentum=0.9_'+\
                        f"run_"
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())