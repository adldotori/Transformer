import os
import os.path as osp
import pandas as pd
import re
import pickle
import argparse
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
# from .optim import SchedAdam

from .model import *
from .dataloader import *

random.seed(17)
torch.manual_seed(17)
torch.backends.cudnn.deterministic = True

class Trainer():
    def __init__(self, args):
        self.args = args

        self.dataset = KorEngDataset(self.args)
        kor_vocab_size, eng_vocab_size = self.dataset.get_vocab_size()

        self.model = Transformer(kor_vocab_size=kor_vocab_size,
                            eng_vocab_size=eng_vocab_size).to(device)
    
        # Scheduling Optimzer
        self.optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9)

        self.loss = nn.CrossEntropyLoss().to(device)

    def train(self):
        self.train_iter, self.valid_iter, self.test_iter = self.dataset.dataloader()
        self.train_iter = self._sample_data(self.train_iter)

        self.model.train()
        best_valid_loss = 1e9
        pbar = tqdm(range(self.args.step), initial=0, dynamic_ncols=True, smoothing=0.01)

        for i in pbar:
            batch = next(self.train_iter)
            res = self.model(batch.kor, batch.eng)

            res = res.view(-1, res.shape[-1])
            trg = torch.transpose(batch.eng, 0, 1).contiguous().view(-1)
            
            loss = self.loss(res, trg)

            self.optimizer.zero_grad()
            loss.backward()     
            self.optimizer.step()

            if not i % 10000:
                valid_loss = self.evaluate()

                pbar.set_description(
                    (
                        f"loss : {valid_loss:.4f}"
                    )
                )
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), f'checkpoint/{self.args.name}_best.pt')

    def evaluate(self):
        loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_iter:
                res = self.model(batch.kor, batch.eng)

                res = res.view(-1, res.shape[-1])
                trg = torch.transpose(batch.eng, 0, 1).contiguous().view(-1)

                loss += self.loss(res, trg)

        loss /= len(self.valid_iter)

        return loss

    def infer(self):
        self.model.load_state_dict(torch.load(f'checkpoint/{self.args.name}_best.pt'))
        self.model.eval()

        tokenizer = self.dataset.get_tokenizer()
        kor, eng = self.dataset.get_vocab()
        
        while True:
            print('>>>', end=' ')
            input_str = input()
            print(input_str)
            token = tokenizer(input_str)
            print(token)
            index = [kor.vocab.stoi[i] for i in token]
            print(index)
            start = eng.vocab.stoi['<sos>']

            src = torch.LongTensor(index).unsqueeze(0).to(device)
            src = torch.transpose(src, 0, 1)
            trg = torch.zeros(1, 80).long().to(device)
            trg[0][0] = start

            for i in range(80):
                _trg = torch.transpose(trg, 0, 1)
                output = self.model(src, _trg)
                output = output.squeeze(0).max(dim=-1, keepdim=False)[1]
                # print(eng.vocab.itos[output.data[i]])
                trg[0][i] = output.data[i]
                if output.data[i] == eng.vocab.stoi['<eos>']:
                    break
            for i in range(80):
                print(eng.vocab.itos[trg[0][i]], end=' ')
            print()

            
            

    
    def _sample_data(self, loader):
        while True:
            for batch in loader:
                yield batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--step', type=int, default=1000000)
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--kor_token_path', type=str, default='kor_token.pkl')
    parser.add_argument('--eng_token_path', type=str, default='eng_token.pkl')
    parser.add_argument('--kor_vocab_path', type=str, default='kor.pkl')
    parser.add_argument('--eng_vocab_path', type=str, default='eng.pkl')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()