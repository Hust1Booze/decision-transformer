import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset
from create_dataset import load_epochs
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000) # default 500000
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

set_seed(args.seed)
#test123


class BipartiteNodeData():
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, 
                 constraint_features=None, 
                 edge_indices=None, 
                 edge_features=None, 
                 variable_features=None,
                 candidates=None, 
                 candidate_choice=None, 
                 candidate_scores=None, 
                 score=None):
        super().__init__()
        if constraint_features is not None:
            self.constraint_features = torch.FloatTensor(constraint_features)
        if edge_indices is not None:
            self.edge_index = torch.LongTensor(edge_indices.astype(np.int64))
        if edge_features is not None:
            self.edge_attr = torch.FloatTensor(edge_features).unsqueeze(1)
        if variable_features is not None:
            self.variable_features = torch.FloatTensor(variable_features)
        if candidates is not None:
            self.candidates = torch.LongTensor(candidates)
            self.num_candidates = len(candidates)
        if candidate_choice is not None:
            self.candidate_choices = torch.LongTensor(candidate_choice)
        if candidate_scores is not None:
            self.candidate_scores = torch.FloatTensor(candidate_scores)
        if score is not None:
            self.score = torch.FloatTensor(score)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = int(max(actions) + 1)
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size

        # change for scip 
        ##states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        #states = states / 255.
        states = []
        for _data in self.data[idx:done_idx]:
            sample_observation, sample_action, sample_action_set, sample_scores = _data
            # We note on which variables we were allowed to branch, the scores as well as the choice 
            # taken by strong branching (relative to the candidates)
            candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
            try:
                candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
                score = []
            except (TypeError, IndexError):
                # only given one score and not in a list so not iterable
                score = torch.FloatTensor([sample_scores])
                candidate_scores = []
            candidate_choice = torch.where(candidates == sample_action)[0][0]
            graph = BipartiteNodeData(sample_observation.row_features, sample_observation.edge_features.indices, 
                                sample_observation.edge_features.values, sample_observation.column_features,
                                candidates, candidate_choice, candidate_scores, score)
        
            # We must tell pytorch geometric how many nodes there are, for indexing purposes
            graph.num_nodes = sample_observation.row_features.shape[0]+sample_observation.column_features.shape[0]

            states +=[graph]

        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps
data_path = '/home/liutf/code/decision-transformer/atari/pure_strong_branch/set_covering/max_steps_None/set_covering_n_rows_500_n_cols_1000/samples/samples_3'
obss, actions, returns, done_idxs, rtgs, timesteps = load_epochs(data_path)
#obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()
