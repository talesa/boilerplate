import torch
from torch import tensor

import matplotlib.pyplot as plt

import datetime
import random

import configargparse
import os.path
yml_base_dir = ''
yml_files = ['config']
yml_paths = [os.path.join(yml_base_dir, yml_file+'.yml') for yml_file in yml_files]
parser = configargparse.get_arg_parser(default_config_files=yml_paths)

import model
from utils import git_revision

def main():
    parser = configargparse.get_argument_parser()
    
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=150000)
    parser.add_argument('--epochs', type=int, default=2000000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path', default='checkpoints/')
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    
    args.runid = datetime.datetime.now().isoformat() + '_' + git_revision()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    train(args)
    
def train(args):
    model = model.model(device=args.device)

    optimizer = torch.optim.Adam(q_meta_net.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    
    zero = tensor([0.], device=args.device)
    one = tensor([1.], device=args.device)

    for e in range(args.epochs):
        loss = 0
        # TODO think about numerical stability here
        loss += torch.zeros(1).mean()
        print(e, loss.item()) if e % 100 == 0 else None
        
        if torch.isnan(loss).any(): 
            raise Exception(f'NaN loss on epoch {e}, terminating')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if e % 1000 == 0:
            torch.save(model.state_dict(), args.path+args.runid)

if __name__ == '__main__':
    main()
