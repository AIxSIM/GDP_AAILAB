from sklearn.mixture import GaussianMixture
from math import exp
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from models_seq.eps_models import EPSM
from loader.dataset import TrajFastDataset
from models_seq.seq_models import Destroyer, Restorer
import numpy as np
import matplotlib.pyplot as plt
import random

from itertools import cycle
from os.path import join
from utils.coors import wgs84_to_gcj02


class Trainer:
    def __init__(self, model: nn.Module, dataset, model_path, model_name):
        self.model = model 
        self.device = model.device
        self.dataset = dataset
        self.model_path = model_path
        self.model_name = model_name

    def train(self, n_epoch, batch_size, lr, remove_region=None):
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # split train test
        train_num = int(0.8 * len(self.dataset))
        train_dataset, test_dataset = random_split(self.dataset, [train_num , len(self.dataset) - train_num])

        # randomly removed edge for new A' and defined sampler that only sample paths that satisfy A'

        if remove_region is not None:
            print(f'remove {remove_region}')
            A_new = self.dataset.edit(removal={"regions": remove_region}, direct_change=False)
            torch.save(A_new, join(self.model_path, f"{self.model_name}_{remove_region}_A_new.pt"))
            train_sampler = CustomPathBatchSampler(train_dataset, batch_size=batch_size, adjacency_matrix=A_new, shuffle=True)
            test_sampler = CustomPathBatchSampler(test_dataset, batch_size=batch_size, adjacency_matrix=A_new, shuffle=False)
        else:
            train_sampler = None
            test_sampler = None

        trainloader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        testloader = DataLoader(test_dataset, batch_sampler=test_sampler,
                                collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        self.model.train()
        iter, train_loss_avg = 0, 0
        kl_loss_avg, ce_loss_avg, con_loss_avg = 0, 0, 0
        try:
            for epoch in range(n_epoch):
                for xs in trainloader:
                    kl_loss, ce_loss, con_loss = self.model(xs)
                    if ce_loss.item() < 60:
                        loss = kl_loss
                    else:
                        loss = kl_loss + ce_loss + con_loss
                        # loss = kl_loss + ce_loss
                    train_loss_avg += loss.item()
                    kl_loss_avg += kl_loss.item()
                    ce_loss_avg += ce_loss.item()
                    con_loss_avg += con_loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    # TODO: clip norm
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    iter += 1
                    if iter % 100 == 0 or iter == 1:
                        # eval test
                        denom = 1 if iter == 1 else 100
                        test_kl, test_ce, test_con = next(self.eval_test(testloader))
                        test_loss = test_kl + test_ce + test_con
                        print(f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, (kl: {kl_loss_avg / denom: .4f}, ce: {ce_loss_avg / denom: .4f}, co: {con_loss_avg / denom: .4f}), test loss: {test_loss: .4f}, (kl: {test_kl: .4f}, ce: {test_ce: .4f}, co: {test_con: .4f})")
                        train_loss_avg, kl_loss_avg, ce_loss_avg, con_loss_avg = 0., 0., 0., 0.
                model_name = f"{self.model_name}_iter_{iter}.pth"
                torch.save(self.model, join(self.model_path, model_name))
        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"
        # save
        self.model.eval()
        # model_name = f"finished_{iter}.pth"
        # torch.save(self.model, join(self.model_path, model_name))
        # print("save finished!")

    def train_gmm(self, gmm_samples, n_comp):
        gmm = GaussianMixture(n_components=n_comp, covariance_type="tied")
        gmm_samples = min(len(self.dataset), gmm_samples)
        lenghts = np.array([len(self.dataset[k]) for k in range(gmm_samples)]).reshape(-1, 1)
        gmm.fit(lenghts)
        self.model.gmm = gmm  


    def eval_test(self, test_loader):
        with torch.no_grad():
            for txs in cycle(test_loader):
                kl_loss, ce_loss, test_con = self.model(txs)
                yield (kl_loss.item(), ce_loss.item(), test_con.item())

    def drop_edges_symmetric(self, A, drop_ratio=0.1):
        A = A.clone().cpu()
        N = A.size(0)

        row_idx, col_idx = torch.triu_indices(N, N, offset=1)
        edge_mask = A[row_idx, col_idx] == 1
        edge_indices = torch.stack([row_idx[edge_mask], col_idx[edge_mask]], dim=0)

        num_edges = edge_indices.size(1)
        num_to_drop = int(num_edges * drop_ratio)

        perm = torch.randperm(num_edges)
        edges_to_drop = edge_indices[:, perm[:num_to_drop]]

        A[edges_to_drop[0], edges_to_drop[1]] = 0
        A[edges_to_drop[1], edges_to_drop[0]] = 0

        return A


class Trainer_SimTime:
    def __init__(self, model: nn.Module, dataset, model_path):
        self.model = model
        self.device = model.device
        self.dataset = dataset
        self.model_path = model_path

    def custom_collate_fn(self, data):
        paths = [torch.tensor(item[0], dtype=torch.float32) for item in data]
        times = [torch.tensor(item[1], dtype=torch.float32).unsqueeze(0) for item in data]
        return paths, times

    def train(self, n_epoch, batch_size, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        # split train test
        train_num = int(0.8 * len(self.dataset))
        train_dataset, test_dataset = random_split(self.dataset, [train_num , len(self.dataset) - train_num])

        trainloader = DataLoader(train_dataset, batch_size,
                                    collate_fn=self.custom_collate_fn
                                )

        testloader = DataLoader(test_dataset, batch_size,
                                    collate_fn=self.custom_collate_fn
                                )

        self.model.train()
        iter, train_loss_avg = 0, 0
        kl_loss_avg, ce_loss_avg, con_loss_avg = 0, 0, 0
        try:
            for epoch in range(n_epoch):
                for xs in trainloader:
                    kl_loss, ce_loss, con_loss = self.model(xs)
                    if ce_loss.item() < 60:
                        loss =  kl_loss
                    else:
                        loss = kl_loss + ce_loss + con_loss
                        # loss = kl_loss + ce_loss
                    train_loss_avg += loss.item()
                    kl_loss_avg += kl_loss.item()
                    ce_loss_avg += ce_loss.item()
                    con_loss_avg += con_loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    # TODO: clip norm
                    optimizer.step()
                    iter += 1
                    if iter % 100 == 0 or iter == 1:
                        # eval test
                        denom = 1 if iter == 1 else 100
                        test_kl, test_ce, test_con = next(self.eval_test(testloader))
                        test_loss = test_kl + test_ce + test_con
                        print(f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, (kl: {kl_loss_avg / denom: .4f}, ce: {ce_loss_avg / denom: .4f}, co: {con_loss_avg / denom: .4f}), test loss: {test_loss: .4f}, (kl: {test_kl: .4f}, ce: {test_ce: .4f}, co: {test_con: .4f})")
                        train_loss_avg, kl_loss_avg, ce_loss_avg, con_loss_avg = 0., 0., 0., 0.
        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"
        # save
        self.model.eval()
        # model_name = f"finished_{iter}.pth"
        # torch.save(self.model, join(self.model_path, model_name))
        # print("save finished!")

    def train_gmm(self, gmm_samples, n_comp):
        gmm = GaussianMixture(n_components=n_comp, covariance_type="tied")
        gmm_samples = min(len(self.dataset), gmm_samples)
        lenghts = np.array([len(self.dataset[k][0]) for k in range(gmm_samples)]).reshape(-1, 1)
        gmm.fit(lenghts)
        self.model.gmm = gmm

            
    def eval_test(self, test_loader):
        with torch.no_grad():
            for txs in cycle(test_loader):
                kl_loss, ce_loss, test_con = self.model(txs)
                yield (kl_loss.item(), ce_loss.item(), test_con.item())


class CustomPathBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, adjacency_matrix, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.A = adjacency_matrix
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        current_batch = []
        for idx in indices:
            path = self.dataset[idx]
            if self._valid_path(path):
                current_batch.append(idx)
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []

    def _valid_path(self, path):
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.A[u, v] == 0:
                return False
        return True

    def __len__(self):
        return len(self.dataset) // self.batch_size


class Trainer_disc:
    def __init__(self, model: nn.Module, dataset, model_path, model_name):
        self.model = model
        self.device = model.device
        self.dataset = dataset
        self.model_path = model_path
        self.model_name = model_name

    def train(self, n_epoch, batch_size, lr, remove_region=None, remove_random=False):
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # split train test
        train_num = int(0.8 * len(self.dataset))
        train_dataset, test_dataset = random_split(self.dataset, [train_num , len(self.dataset) - train_num])

        # randomly removed edge for new A' and defined sampler that only sample paths that satisfy A'

        if remove_region is not None:
            print(f'remove {remove_region}')
            A_new = self.dataset.edit(removal={"regions": remove_region}, direct_change=False)
            torch.save(A_new, join(self.model_path, f"{self.model_name}_{remove_region}_A_new.pt"))
            train_sampler = CustomPathBatchSampler(train_dataset, batch_size=batch_size, adjacency_matrix=A_new, shuffle=True)
            test_sampler = CustomPathBatchSampler(test_dataset, batch_size=batch_size, adjacency_matrix=A_new, shuffle=False)
        elif remove_random:
            A_new = self.dataset.edit(is_random=True, direct_change=False)
            train_sampler = CustomPathBatchSampler(train_dataset, batch_size=batch_size, adjacency_matrix=A_new, shuffle=True)
            test_sampler = CustomPathBatchSampler(test_dataset, batch_size=batch_size, adjacency_matrix=A_new, shuffle=False)
        else:
            train_sampler = None
            test_sampler = None
            A_new = self.dataset.A

        trainloader_A = DataLoader(train_dataset, batch_sampler=None,
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        trainloader_new = DataLoader(train_dataset, batch_sampler=train_sampler,
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        testloader_A = DataLoader(test_dataset, batch_sampler=None,
                                collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        testloader_new = DataLoader(test_dataset, batch_sampler=test_sampler,
                                collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        self.model.train()
        iter, train_loss_avg = 0, 0
        kl_loss_avg, ce_loss_avg, con_loss_avg = 0, 0, 0
        try:
            for epoch in range(n_epoch):
                for xs, newxs in zip(trainloader_A, trainloader_new):
                    loss = self.model(xs, newxs, self.dataset.A, A_new)
                    train_loss_avg += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    # TODO: clip norm
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    iter += 1
                    if iter % 100 == 0 or iter == 1:
                        # eval test
                        denom = 1 if iter == 1 else 100
                        # test_kl, test_ce, test_con = next(self.eval_test(testloader))
                        # test_loss = test_kl + test_ce + test_con
                        # print(f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, (kl: {kl_loss_avg / denom: .4f}, ce: {ce_loss_avg / denom: .4f}, co: {con_loss_avg / denom: .4f}), test loss: {test_loss: .4f}, (kl: {test_kl: .4f}, ce: {test_ce: .4f}, co: {test_con: .4f})")
                        print(f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}")
                        train_loss_avg = 0.
                model_name = f"{self.model_name}_iter_{iter}.pth"
                torch.save(self.model, join(self.model_path, model_name))
        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"
        # save
        self.model.eval()
        # model_name = f"finished_{iter}.pth"
        # torch.save(self.model, join(self.model_path, model_name))
        # print("save finished!")

    def train_gmm(self, gmm_samples, n_comp):
        gmm = GaussianMixture(n_components=n_comp, covariance_type="tied")
        gmm_samples = min(len(self.dataset), gmm_samples)
        lenghts = np.array([len(self.dataset[k]) for k in range(gmm_samples)]).reshape(-1, 1)
        gmm.fit(lenghts)
        self.model.gmm = gmm


    def eval_test(self, test_loader):
        with torch.no_grad():
            for txs in cycle(test_loader):
                kl_loss, ce_loss, test_con = self.model(txs)
                yield (kl_loss.item(), ce_loss.item(), test_con.item())

    def drop_edges_symmetric(self, A, drop_ratio=0.1):
        A = A.clone().cpu()
        N = A.size(0)

        row_idx, col_idx = torch.triu_indices(N, N, offset=1)
        edge_mask = A[row_idx, col_idx] == 1
        edge_indices = torch.stack([row_idx[edge_mask], col_idx[edge_mask]], dim=0)

        num_edges = edge_indices.size(1)
        num_to_drop = int(num_edges * drop_ratio)

        perm = torch.randperm(num_edges)
        edges_to_drop = edge_indices[:, perm[:num_to_drop]]

        A[edges_to_drop[0], edges_to_drop[1]] = 0
        A[edges_to_drop[1], edges_to_drop[0]] = 0

        return A

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    max_T = 100
    dataset = TrajFastDataset("chengdu", ["20161101"], "./sets_data", device, is_pretrain=True)
    betas = torch.linspace(0.0001, 10, max_T)
    # old beta: 0.01, 15, 50
    
    destroyer = Destroyer(dataset.A, betas, max_T, device)
    eps_model = EPSM(dataset.n_vertex, x_emb_dim=50, hidden_dim=20, dims=[100, 120, 200], device=device, pretrain_path="./sets_data/chengdu_node2vec.pkl")
    restorer = Restorer(eps_model, destroyer, device)
    
    trainer = Trainer(restorer, dataset, device, "./sets_model")
    trainer.train_gmm(gmm_samples=50000, n_comp=5)
    trainer.train(n_epoch=50, batch_size=16, lr=0.0005)
    
    restorer.eval()
    paths = restorer.sample_wo_len(100)
    
    multiple_locs = []
    for path in paths:
        locs = [[wgs84_to_gcj02(dataset.G.nodes[v]["lng"], dataset.G.nodes[v]["lat"])[1], 
                 wgs84_to_gcj02(dataset.G.nodes[v]["lng"], dataset.G.nodes[v]["lat"])[0]] 
                for v in path]
        multiple_locs.append(locs)
        print(locs)
    
    