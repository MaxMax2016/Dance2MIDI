
from core.engine import BaseEngine
from core.dataloaders import DataLoaderFactory
from core.models import ModelFactory
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim
import time
from core.dataloaders.dataset import D2MIDIDataset
from core.optimizer import CustomSchedule
from core.metrics import compute_epiano_accuracy
from pprint import pprint
from datetime import datetime
import os
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='train_default.log',level=logging.DEBUG, format=LOG_FORMAT)

class Engine(BaseEngine):

    def __init__(self, args):
        currentDateAndTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_path=os.path.join(args.experiment_path,currentDateAndTime)
        os.makedirs(self.experiment_path, exist_ok=True)

        self.summary_writer = SummaryWriter(log_dir=self.experiment_path)
        self.model_builder = ModelFactory(args)
        self.dataset_builder = DataLoaderFactory(args)

        self.train_ds = self.dataset_builder.build(split='train') 
        self.test_ds = self.dataset_builder.build(split='val') 
        self.ds: D2MIDIDataset = self.train_ds.dataset

        self.train_criterion = nn.CrossEntropyLoss(
            ignore_index=self.ds.NPAD
        )
        self.val_criterion = nn.CrossEntropyLoss(
            ignore_index=self.ds.NPAD
        )
        self.model: nn.Module = self.model_builder.build(device=torch.device('cuda'), wrapper=nn.DataParallel)
        optimizer = optim.Adam(self.model.parameters(), lr=0., betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = CustomSchedule(
            args.emb_dim,
            optimizer=optimizer,
        )
        # load checkpoint
        if args.load_checkpoint:
            logging.info(f'Load checkpoint from {args.load_checkpoint}')
            checkpoint = torch.load(args.load_checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            logging.info(f'Loaded checkpoint from epoch {self.start_epoch}')
        else:
            self.start_epoch = 0

        self.num_epochs = args.num_epochs

        logging.info(f'Use control: {self.ds.use_control}')

    def train(self, epoch=0):

        num_iters = len(self.train_ds)
        self.model.train()
        epoch_loss = 0
        acc_meter = 0

        for i, data in enumerate(self.train_ds):
            midi_x, midi_y = data['midi_x'], data['midi_y']

            if self.ds.use_pose:
                feat = data['pose']
            elif self.ds.use_rgb:
                feat = data['rgb']
            elif self.ds.use_flow:
                feat = data['flow']
            else:
                raise Exception('No feature!')

            feat, midi_x, midi_y = (
                feat.cuda(non_blocking=True),
                midi_x.cuda(non_blocking=True),
                midi_y.cuda(non_blocking=True)
            )

            if self.ds.use_control:
                control = data['control']
                control = control.cuda(non_blocking=True)
                control = control
            else:
                control = None
       
            output = self.model(feat, midi_x, pad_idx=self.ds.NPAD, control=control)     
            loss = self.train_criterion(output.view(-1, output.shape[-1]),midi_y.flatten())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            # Todo
            acc = compute_epiano_accuracy(output, midi_y, pad_idx=self.ds.NPAD)

            logging.info(
                f'Train [{epoch}/{self.num_epochs}][{i}/{num_iters}]\t'
                f'Train_loss {loss.item()}\t'
                f'Accuracy {acc.item()}'
            )
            epoch_loss += loss.item()
            acc_meter += acc.item()
        epoch_loss /= num_iters
        acc_meter /= num_iters
        self.summary_writer.add_scalar('train/loss', epoch_loss, epoch)
        self.summary_writer.add_scalar('train/acc', acc_meter, epoch)
        return epoch_loss,acc_meter

    def test(self, epoch=0):
        num_iters = len(self.test_ds)
        self.model.eval()
        epoch_loss = 0
        acc_meter = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_ds):
                midi_x, midi_y = data['midi_x'], data['midi_y']

                if self.ds.use_pose:
                    feat = data['pose']
                elif self.ds.use_rgb:
                    feat = data['rgb']
                elif self.ds.use_flow:
                    feat = data['flow']
                else:
                    raise Exception('No feature!')

                feat, midi_x, midi_y = (
                    feat.cuda(non_blocking=True),
                    midi_x.cuda(non_blocking=True),
                    midi_y.cuda(non_blocking=True)
                )

                if self.ds.use_control:
                    control = data['control']
                    control = control.cuda(non_blocking=True)
                    control = control
                else:
                    control = None

                output = self.model(feat, midi_x, pad_idx=self.ds.NPAD, control=control)

                """
                For CrossEntropy
                output: [B, T, D] -> [BT, D]
                target: [B, T] -> [BT]
                """
                loss = self.val_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

                acc = compute_epiano_accuracy(output, midi_y,pad_idx=self.ds.NPAD)

                logging.info(
                    f'Val [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
                    f'{loss.item()}\t{acc.item()}'
                )
                epoch_loss += loss.item()
                acc_meter += acc.item()
            epoch_loss /= num_iters
            acc_meter /= num_iters
            self.summary_writer.add_scalar('val/loss', epoch_loss, epoch)
            self.summary_writer.add_scalar('val/acc', acc_meter, epoch)

        return epoch_loss,acc_meter

    @staticmethod
    def epoch_time(start_time: float, end_time: float):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self):
        best_loss = float('inf')

        # start training
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time = time.time()
            train_loss,train_acc = self.train(epoch)
            val_loss,val_acc = self.test(epoch)
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            logging.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            if is_best:
                torch.save(
                    {
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_loss': best_loss,
                        'epoch': epoch,
                    },
                    self.experiment_path+'/best.pth'
                )

    def close(self):
        self.summary_writer.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_workers', default=256, type=int, help='number of workers')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--load_checkpoint', default=None, type=str, help='load checkpoint')
    # dataset config
    parser.add_argument('--dset', default='D2MIDI', type=str, help='dataset name')
    parser.add_argument('--duration', default=30, type=float, help='duration of the video')
    parser.add_argument('--fps', default=22, type=int, help='fps of the video')
    parser.add_argument('--pose_layout', default='body25', type=str, help='pose layout')
    parser.add_argument('--duplication', default=10, type=int, help='duplication')
    parser.add_argument('--events_per_sec', default=80, type=int, help='events per sec')
    parser.add_argument('--random_shift_rate', default=0.2, type=float, help='random shift rate')
    parser.add_argument('--split_csv_dir', type=str)
    parser.add_argument('--midi', type=str)
    parser.add_argument('--pose', type=str)
    # models config
    parser.add_argument('--name', default='music_transformer', type=str)
    parser.add_argument('--emb_dim', default=512, type=int)
    parser.add_argument('--clip', default='null', type=str)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--rpr', default=True, type=bool)
    parser.add_argument('--decoder_max_seq', default=4096+1, type=int)
    parser.add_argument('--pose_net_layers', default=10, type=int)
    # experiment
    parser.add_argument('--experiment_path', default='exps/', type=str, help='experiment path')
    
    args = parser.parse_args()

    print('=' * 100)
    pprint(args)
    print('=' * 100)
    engine = Engine(args)
    engine.run()
    engine.close()


if __name__ == '__main__':
    main()
