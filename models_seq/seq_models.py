import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from models_seq.eps_models import EPSM
from models_seq.disc_models import Discriminator
from torch.distributions.utils import probs_to_logits, clamp_probs
from collections import defaultdict
import numpy as np


class Destroyer:
    
    def __init__(self, A, betas, max_T, device) -> None:
        self.A = A.clone().detach().to(device, dtype=torch.float32)
        self.n_vertex = A.shape[0]
        self.device = device
        A_ = (self.A - torch.diag(self.A.sum(dim=0)).to(self.device))
        self.betas = betas.to(device=self.device)
        self.max_T = max_T
        self.matrices = torch.zeros(self.max_T + 1, self.n_vertex, self.n_vertex).to(self.device)
        self.Q = torch.zeros_like(self.matrices).to(self.device)
        self.matrices[0] = torch.eye(self.n_vertex, device=self.device)
        self.Q[0] = torch.eye(self.n_vertex, device=self.device)
        for i in range(1, self.max_T + 1):
            self.Q[i] = torch.linalg.matrix_exp(A_ * self.betas[i - 1])
            self.matrices[i] = self.Q[i] @ self.matrices[i - 1]
        
        # print(self.matrices[1][23], self.matrices[1][23][23])
        # print(self.matrices[self.max_T // 2][23], self.matrices[self.max_T // 2][23][23])
        # print(self.matrices[-1][23], self.matrices[-1][23][23])
        # print("#####")
        # print(self.matrices[1][1172], self.matrices[1][1172][1172])
        # print(self.matrices[self.max_T // 2][1172], self.matrices[self.max_T // 2][1172][1172])
        # print(self.matrices[-1][1172], self.matrices[-1][1172][1172])

    
    def get_Q(self):
        Q = self.Q 
        del self.Q 
        return Q
        
    def diffusion(self, xs, ts, ret_distr=False):
        # xs: list of labels in tensor | tensor of logits
        # ts: [t_1, ..., t_n]
        # x_diffused: list of diffused labels in tensor
        # print(xs)
        lengths = [x.shape[0] for x in xs]
        batch_size, horizon = len(lengths), max(lengths)
        if type(xs) is torch.Tensor and xs.dim() == 3:
            xs = rearrange(xs, "b h c -> c (b h)").to(self.device)
            x_distr = self.matrices[ts[0]] @ xs
            x_distr = rearrange(x_distr, "c (b h) -> b h c", h=horizon)
            return x_distr
        else:
            xs_padded = pad_sequence(xs, batch_first=True, padding_value=1.).to(self.device).long()
            # multiply one hot equivalent to pick the specific column
            # [1, 3, 2] -> [1,1,1..,1, 3,3,3...,3, 2,2,...2]
            ts_padded = ts.view(-1, 1).repeat(1, horizon).view(-1,)
            x_distr_padded = self.matrices[ts_padded, :, xs_padded.view(-1)]
            if ret_distr:
                return x_distr_padded
            x_diffused_padded = torch.multinomial(x_distr_padded, 1).view(batch_size, -1)
            x_diffused = [x_diffused_padded[k][:length] for k, length in enumerate(lengths)]
            return x_diffused
    
  
  
class Restorer(nn.Module):
    def __init__(self, eps_model: EPSM, destroyer: Destroyer, device):
        super().__init__()
        self.n_vertex = destroyer.n_vertex
        self.eps_model = eps_model
        self.model_device = self.eps_model.device
        self.device = device
        self.destroyer = destroyer
        self.des_device = destroyer.device
        self.max_T = self.destroyer.max_T
        self.matrices = self.destroyer.matrices
        self.A = destroyer.A
        self.Q = self.destroyer.get_Q()
        self.Q = self.Q.to(self.device)
        self.max_deg = self.A.sum(1).max()

        self.applying_mask_intermediate = False
        self.applying_mask_intermediate_temperature = False
    
    def forward(self, xs):
        # xs: list of tensors of labels
        batch_size = len(xs)
        if batch_size == 0:
            import pdb
            pdb.set_trace()
        lengths = torch.Tensor([x.shape[0] for x in xs]).long().to(self.device)
        
        # uniformly choose t
        ts = torch.randint(1, self.max_T + 1, [batch_size]).to(self.device)
        
        x_t = self.destroyer.diffusion(xs, ts, ret_distr=False)
        xt_padded = pad_sequence(x_t, batch_first=True, padding_value=0).long()
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=0).long()
        horizon = xt_padded.shape[1]
        ts_padded = ts.view(-1, 1).repeat(1, horizon)
        
        # true_probs_unorm = Q_t @ x_t * \bar{E}_{t-1} @ x_0 both x_0 and x_t is categorical
        EtXt = self.Q[ts_padded.view(-1,).to(self.device), :, xt_padded.view(-1).to(self.device)]
        true_probs_unorm = EtXt * self.matrices[ts_padded.view(-1,) - 1, :, xs_padded.view(-1)].to(self.device)
        true_probs = true_probs_unorm / true_probs_unorm.sum(1, keepdim=True)
        true_probs = clamp_probs(true_probs)
        true_probs = rearrange(true_probs, "(b h) c -> b h c", h=horizon)
        
        x0_pred_logits = self.restore(xt_padded.to(self.model_device), lengths.to(self.model_device), ts.to(self.model_device))
        x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
        # pred_probs_unorm = E_t @ x_t * \bar{E}_{t-1} @ \hat{x}_0  x_0 is logits while x_t is categorical
        Et_minus_one_bar_hat_x0 = (self.matrices[ts - 1] @ x0_pred_probs.transpose(2, 1).to(self.des_device)).to(self.device)
        Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")
        pred_probs_unorm = EtXt * Et_minus_one_bar_hat_x0
        pred_probs = pred_probs_unorm / torch.clamp(pred_probs_unorm.sum(1, keepdim=True), min=1e-8)
        pred_logits = probs_to_logits(pred_probs)
        pred_logits = rearrange(pred_logits, "(b h) c -> b h c", h=horizon)
        eps = 0.000001
        kl_loss = sum([F.kl_div(pred_logits[k][:l] + eps, true_probs[k][:l], reduction="batchmean") for k, l in enumerate(lengths)])
        ce_loss = sum([F.cross_entropy(x0_pred_logits[k][:lengths[k]].to(x) + eps, x[:lengths[k]].long(), reduction="mean") for k, x in enumerate(xs)])
        con_loss = -sum([((self.A @ (x0_pred_probs[k, 1:l, :] + eps).log().T).T * x0_pred_probs[k, :l-1, :]).mean() for k, l in enumerate(lengths)]) / batch_size
        con_loss += -sum([((self.A @ (x0_pred_probs[k, :l-1, :] + eps).log().T).T * x0_pred_probs[k, 1:l, :]).mean() for k, l in enumerate(lengths)]) / batch_size
        # ce_loss = torch.zeros_like(kl_loss)
        # con_loss = torch.zeros_like(kl_loss)
        if torch.isnan(kl_loss):
            print('kl_loss nan')
            import pdb
            pdb.set_trace()
        if torch.isnan(ce_loss):
            print('ce_loss nan')
            import pdb
            pdb.set_trace()
        if torch.isnan(con_loss):
            print('con_loss nan')
            import pdb
            pdb.set_trace()
        return kl_loss, ce_loss, con_loss * 100
         
    def restore(self, xt_padded, lengths=None, ts=None):
        # xt_padded: b, h value is vertex number
        # ts: b value is time for each
        batch_size = xt_padded.shape[0]
        if ts is None:
            ts = torch.Tensor([self.max_T]).repeat(batch_size).to(self.device)
        x0_pred_logits = self.eps_model(xt_padded, lengths, ts)
        return x0_pred_logits
    
    def sample(self, n_samples: int, batch_traj_num=200):
        assert hasattr(self, "gmm")
        lengths = self.gmm.sample(n_samples)[0].reshape(-1).astype(int)
        lengths = np.sort(lengths[lengths > 0])
        lengths = torch.Tensor(lengths).long().to(self.device)
        # batch_traj_num = 200
        n_batch = n_samples // batch_traj_num
        paths = []
        for b in range(n_batch):
            paths.extend(self.sample_with_len(lengths[b * batch_traj_num: min((b + 1) * batch_traj_num, n_samples)]))
        return paths
    
    def sample_with_len(self, lengths, ret_distr=False, xt=None, T=None, ret_trace=False):
        ############################################## YM
        applying_mask_intermediate = self.applying_mask_intermediate
        applying_mask_intermediate_temperature = self.applying_mask_intermediate_temperature
        ## 1. No constrint
        # self.A = torch.ones_like(self.A)

        ## 2. Random constrarint
        # upper_tri_prob = torch.bernoulli(torch.full((1470, 1470), 0.01)).int().to(self.A.device)
        # upper_tri = torch.triu(upper_tri_prob)
        # self.A = upper_tri + upper_tri.T - torch.diag(torch.diag(upper_tri))

        ## 3. Only one node active
        # self.A = torch.zeros((1470, 1470), dtype=torch.int).to(self.A.device)
        # self.A[0, :] = 1
        # self.A[:, 0] = 1

        ## Check symmetric or not
        # is_symmetric = torch.allclose(self.A, self.A.T)
        # print("Is symmetric:", is_symmetric)
        # print(self.A)
        ##############################################

        if ret_trace:
            reverse_trace = defaultdict(list) # t -> [path1, path2,...]
        if T is None:
            T = self.max_T
        # use x0 directly
        print('mask_int: ', applying_mask_intermediate)
        print('mask_int_temp: ', applying_mask_intermediate_temperature)
        n_samples = lengths.shape[0]
        horizon = max(lengths)
        if xt is None:
            xt = torch.randint(0, self.n_vertex, [n_samples, horizon]).to(self.device)
        else:
            xt = xt.to(self.device)
        with torch.no_grad():
            for t in range(T, 0, -1):
                ts = torch.Tensor([t]).long().to(self.device).repeat(n_samples)
                x0_pred_logits = self.restore(xt, lengths, ts) 
                x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
                # pred_probs_unorm = E_t @ x_t * \bar{E}_{t-1} @ \hat{x}_0  x_0 is logits while x_t is categorical
                EtXt = self.Q[t, :, xt.view(-1)].T
                Et_minus_one_bar_hat_x0 = self.matrices[ts - 1] @ x0_pred_probs.transpose(2, 1)
                Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")
                pred_probs_unorm = EtXt * Et_minus_one_bar_hat_x0
                sum_probs = torch.clamp(pred_probs_unorm.sum(1, keepdim=True), min=1e-8)
                pred_probs = pred_probs_unorm / sum_probs
                mask = (sum_probs == 1e-8)[:, 0]
                pred_probs[mask] = 1.0 / pred_probs.shape[1]

                if applying_mask_intermediate:
                    pred_prob_ = rearrange(pred_probs, "(b h) c -> b h c", b=n_samples)
                    xt = torch.zeros([n_samples, horizon]).long().to(self.device)

                    x_mask = pred_prob_[:, 0].clone()
                    if (self.A.sum(dim=1) == 0).sum() != 0:
                        x_mask[:, self.A.sum(dim=1) == 0] = 0.
                    xt[:, 0] = torch.multinomial(x_mask, 1).view(-1)

                    for k in range(1, horizon):
                        if applying_mask_intermediate_temperature:
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * (pred_prob_[:, k]) * ((self.max_T - t) / self.max_T) + pred_prob_[:, k] * ( t / self.max_T)
                        else: ## Hard topology on every xt
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * (pred_prob_[:, k])  # b * v
                        random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                        x_next_masked_prob[random] = 1.
                        if applying_mask_intermediate_temperature:
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * x_next_masked_prob * ((self.max_T - t) / self.max_T) + x_next_masked_prob * (t / self.max_T)
                        else:  ## Hard topology on every xt
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * x_next_masked_prob  # b * v
                        xt[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)
                else:
                    xt = torch.multinomial(pred_probs, num_samples=1, replacement=True)
                    xt = rearrange(xt, "(b h) 1 -> b h", b=n_samples) #torch.Size([n_samples, horizon])

                if ret_trace:
                    reverse_trace[t] = [xt[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]

            x = torch.zeros_like(xt).long().to(self.device)

            x_mask = x0_pred_probs[:, 0].clone()
            if (self.A.sum(dim=1)==0).sum() != 0:
                x_mask[:, self.A.sum(dim=1)==0] = 0.
            x[:, 0] = torch.multinomial(x_mask, 1).view(-1)

            for k in range(1, horizon):
                x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * (x0_pred_probs[:, k]) # b * v
                random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                x_next_masked_prob[random] = 1.
                x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * x_next_masked_prob
                try:
                    x[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)
                except:
                    import pdb
                    pdb.set_trace()
            x_list = [x[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]
            if ret_trace:
                reverse_trace[0] = x_list
                return reverse_trace
            if ret_distr:
                return x_list, x0_pred_probs
            return x_list
        
    def beam_search(self, x0_pred_probs, lengths, n_beam):
        self.max_deg = 8
        batch_size, horizon, vertex = x0_pred_probs.shape
        x = torch.zeros([batch_size, horizon]).long().to(self.device)
        x0_pred_probs = x0_pred_probs.cpu()
        for k in range(batch_size):
            probs, frontiers = x0_pred_probs[k, 0].topk(n_beam)
            probs = probs.cpu().tolist()
            frontiers = frontiers.cpu().tolist()
            
            length = lengths[k]
            pred = torch.zeros([length, n_beam]).long().to(self.device)
            # traverse along the horizon
            for h in range(1, length):
                new_frontiers = []
                for i, f in enumerate(frontiers):
                    val, ind = self.A[f].topk(self.max_deg)
                    valid_ind = ind[val > 0]  # get connected
                    for adj in valid_ind:
                        new_frontiers.append(((probs[i] * x0_pred_probs[k, h, adj]).item(), adj.item(), f)) # probs, ind, pred
                new_frontiers.sort(reverse=True)
                new_frontiers = new_frontiers[:n_beam]
                frontiers = [each[1] for each in new_frontiers]
                probs = [each[0] for each in new_frontiers]
                pred[h, :] = torch.Tensor([each[2] for each in new_frontiers]).long().to(self.device)
                
            # sample
            probs = torch.Tensor(probs).to(self.device)
            probs[probs < 1e-5] = 1e-3
            choice = torch.multinomial(probs, num_samples=1).squeeze()
            x[k, :length - 1] = pred[1:, choice]
            x[k, length - 1] = frontiers[choice]
        return x        
    
    def eval_nll(self, real_paths):
        total = len(real_paths)
        nlls = np.zeros(total)
        batch_traj_num = 200
        n_batch = (total + batch_traj_num - 1) // batch_traj_num
        
        for k in range(n_batch):
            left = k * batch_traj_num
            right = min((k + 1) * batch_traj_num, total)
            batch_size = right - left
            xs = [torch.tensor(path).to(self.device) for path in real_paths[left: right]]
            lengths = [x.shape[0] for x in xs]
            ts = torch.tensor([self.max_T // 20]).repeat(batch_size).to(self.device)
            x_t = self.destroyer.diffusion(xs, ts)
            xt_padded = pad_sequence(x_t, batch_first=True, padding_value=0).long()
            lengths = torch.Tensor(lengths).long().to(self.device)
            _, probs = self.sample_with_len(lengths, ret_distr=True, xt=xt_padded, T=self.max_T // 20)
            for i, path in enumerate(real_paths[left: right]):
                logits = torch.masked_fill(probs_to_logits(probs[i])[1:lengths[i], :], self.A[path[:-1]] == 0, -1e20)
                prob = torch.softmax(logits, dim=-1)
                nlls[i + left] -= (prob[0, path[0]] + 0.001).log()
                nlls[i + left] -= (prob[torch.arange(lengths[i] - 1), path[1:]] + 0.00001).log().sum()
        return nlls

    def edit(self, removal=None, is_random=False, G=None, direct_change=False):  # removal : {"nodes": [xxx, yyy, zzz], "edges": [[XXX, YYY], [ZZZ, WWW]], "regions" : list of [[min_lat, max_lat], [min_lng, max_lng]]}
        if (removal is None) and (not is_random):
            exit("Please check edit in seq_models.py")

        if is_random:
            size = 0.01
            min_lat, max_lat = 999, -999
            min_lng, max_lng = 999, -999
            for node, data in G.nodes(data=True):
                lat, lng = data.get("lat"), data.get("lng")

                if lat < min_lat:
                    min_lat = lat
                if lat > max_lat:
                    max_lat = lat
                if lng < min_lng:
                    min_lng = lng
                if lng > max_lng:
                    max_lng = lng
            # print (min_lat, max_lat, max_lat - min_lat) # 36.3270948 36.3699729 0.04287810000000292
            # print (min_lng, max_lng, max_lng - min_lng) # 127.3170026 127.3692761 0.05227349999999831
            start_lat, start_lng = np.random.uniform(min_lat, max_lat), np.random.uniform(min_lng, max_lng)

            removal_region = [[start_lat, start_lat+size], [start_lng, start_lng+size]]
            print ("Random removal region: ", removal_region)

            removal = {"regions": [removal_region]}

        new_A = self.A.clone().detach()

        if "nodes" in removal.keys():
            for node in removal["nodes"]:
                new_A[node, :], new_A[:, node] = 0, 0

        if "edges" in removal.keys():
            for node1, node2 in removal["edges"]:
                new_A[node1, node2], new_A[node2, node1] = 0, 0

        if "regions" in removal.keys():
            for node, data in G.nodes(data=True):
                lat, lng = data.get("lat"), data.get("lng")
                for lat_range, lng_range in removal["regions"]:
                    if lat_range[0] <= lat <= lat_range[1] and lng_range[0] <= lng <= lng_range[1]:
                        new_A[node, :], new_A[:, node] = 0, 0
                        break
        print(f'remove {(self.A.data - new_A.data).sum()/2} pairs.')

        assert torch.all(new_A.transpose(0, 1) == new_A)

        if direct_change:
            self.A.data = new_A.data
        else:
            return new_A


class Restorer_SimTime(nn.Module):
    def __init__(self, eps_model: EPSM, destroyer: Destroyer, device):
        super().__init__()
        self.n_vertex = destroyer.n_vertex
        self.eps_model = eps_model
        self.model_device = self.eps_model.device
        self.device = device
        self.destroyer = destroyer
        self.des_device = destroyer.device
        self.max_T = self.destroyer.max_T
        self.matrices = self.destroyer.matrices
        self.A = destroyer.A
        self.Q = self.destroyer.get_Q()
        self.Q = self.Q.to(self.device)
        self.max_deg = self.A.sum(1).max()

    def forward(self, xs):
        # xs: list of tensors of labels
        batch_size = len(xs[0])
        sim_time = torch.stack(xs[1])
        xs = xs[0]
        lengths = torch.Tensor([x.shape[0] for x in xs]).long().to(self.device)

        # uniformly choose t
        ts = torch.randint(1, self.max_T + 1, [batch_size]).to(self.device)

        x_t = self.destroyer.diffusion(xs, ts, ret_distr=False)
        xt_padded = pad_sequence(x_t, batch_first=True, padding_value=0).long()
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=0).long()
        horizon = xt_padded.shape[1]
        ts_padded = ts.view(-1, 1).repeat(1, horizon)

        # true_probs_unorm = Q_t @ x_t * \bar{E}_{t-1} @ x_0 both x_0 and x_t is categorical
        EtXt = self.Q[ts_padded.view(-1,).to(self.device), :, xt_padded.view(-1).to(self.device)]
        true_probs_unorm = EtXt * self.matrices[ts_padded.view(-1,) - 1, :, xs_padded.view(-1)].to(self.device)
        true_probs = true_probs_unorm / true_probs_unorm.sum(1, keepdim=True)
        true_probs = rearrange(true_probs, "(b h) c -> b h c", h=horizon)
        zeros_mask = true_probs.eq(0)

        x0_pred_logits = self.restore(xt_padded.to(self.model_device), sim_time.to(self.model_device), lengths.to(self.model_device), ts.to(self.model_device))
        x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
        # pred_probs_unorm = E_t @ x_t * \bar{E}_{t-1} @ \hat{x}_0  x_0 is logits while x_t is categorical
        Et_minus_one_bar_hat_x0 = (self.matrices[ts - 1] @ x0_pred_probs.transpose(2, 1).to(self.des_device)).to(self.device)
        Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")
        pred_probs_unorm = EtXt * Et_minus_one_bar_hat_x0
        pred_probs = pred_probs_unorm / torch.clamp(pred_probs_unorm.sum(1, keepdim=True), min=1e-8)
        pred_logits = probs_to_logits(pred_probs)
        pred_logits = rearrange(pred_logits, "(b h) c -> b h c", h=horizon)
        eps = 0.000001
        eps_2 = 1e-20
        kl_loss = sum([F.kl_div(pred_logits[k][:l] + eps, true_probs[k][:l] + eps_2, reduction="batchmean") for k, l in enumerate(lengths)])
        ce_loss = sum([F.cross_entropy(x0_pred_logits[k][:lengths[k]].to(x) + eps, x[:lengths[k]].long(), reduction="mean") for k, x in enumerate(xs)])
        con_loss = -sum([((self.A @ (x0_pred_probs[k, 1:l, :] + eps).log().T).T * x0_pred_probs[k, :l-1, :]).mean() for k, l in enumerate(lengths)]) / batch_size
        con_loss += -sum([((self.A @ (x0_pred_probs[k, :l-1, :] + eps).log().T).T * x0_pred_probs[k, 1:l, :]).mean() for k, l in enumerate(lengths)]) / batch_size
        return kl_loss, ce_loss, con_loss * 100

    def restore(self, xt_padded, sim_time, lengths=None, ts=None):
        # xt_padded: b, h value is vertex number
        # ts: b value is time for each
        batch_size = xt_padded.shape[0]
        if ts is None:
            ts = torch.Tensor([self.max_T]).repeat(batch_size).to(self.device)
        x0_pred_logits = self.eps_model(xt_padded, lengths, ts, sim_time)
        return x0_pred_logits

    def sample(self,  n_samples: int, sim_time=None):
        assert hasattr(self, "gmm")
        lengths = self.gmm.sample(n_samples)[0].reshape(-1).astype(int)
        lengths = np.sort(lengths[lengths > 0])
        lengths = torch.Tensor(lengths).long().to(self.device)
        batch_traj_num = 200
        n_batch = n_samples // batch_traj_num
        paths = []
        for b in range(n_batch):
            paths.extend(self.sample_with_len(lengths[b * batch_traj_num: min((b + 1) * batch_traj_num, n_samples)], sim_time=sim_time))
        return paths

    def sample_with_len(self, lengths, sim_time=None, ret_distr=False, xt=None, T=None, ret_trace=False):
        if ret_trace:
            reverse_trace = defaultdict(list) # t -> [path1, path2,...]
        if T is None:
            T = self.max_T
        # use x0 directly
        n_samples = lengths.shape[0]
        if sim_time is None:
            sim_time = torch.randint(0,24, [n_samples, 1]).float().to(self.device)
        elif type(sim_time) is int:
            sim_time = torch.Tensor([sim_time]).repeat(n_samples, 1).to(self.device)
        else:
            raise ValueError("sim_time should be int or None")
        horizon = max(lengths)
        if xt is None:
            xt = torch.randint(0, self.n_vertex, [n_samples, horizon]).to(self.device)
        else:
            xt = xt.to(self.device)
        with torch.no_grad():
            for t in range(T, 0, -1):
                ts = torch.Tensor([t]).long().to(self.device).repeat(n_samples)
                x0_pred_logits = self.restore(xt, sim_time, lengths, ts)
                x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
                # pred_probs_unorm = E_t @ x_t * \bar{E}_{t-1} @ \hat{x}_0  x_0 is logits while x_t is categorical
                EtXt = self.Q[t, :, xt.view(-1)].T
                Et_minus_one_bar_hat_x0 = self.matrices[ts - 1] @ x0_pred_probs.transpose(2, 1)
                Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")
                pred_probs_unorm = EtXt * Et_minus_one_bar_hat_x0
                pred_probs = pred_probs_unorm / torch.clamp(pred_probs_unorm.sum(1, keepdim=True), min=1e-8)
                xt = torch.multinomial(pred_probs, num_samples=1, replacement=True)
                xt = rearrange(xt, "(b h) 1 -> b h", b=n_samples)
                if ret_trace:
                    reverse_trace[t] = [xt[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]
            # torch.multinomial(x0_pred_probs, num_samples=1, replacement=True)
            # x = self.beam_search(x0_pred_probs, lengths, n_beam=10)
            x = torch.zeros_like(xt).long().to(self.device)
            x[:, 0] = torch.multinomial(x0_pred_probs[:, 0], 1).view(-1)
            for k in range(1, horizon):
                x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * (x0_pred_probs[:, k]) # b * v
                random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                x_next_masked_prob[random] = 1.
                x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * x_next_masked_prob
                x[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)
            x_list = [x[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]
            if ret_trace:
                reverse_trace[0] = x_list
                return reverse_trace
            if ret_distr:
                return x_list, x0_pred_probs
            return x_list

    def beam_search(self, x0_pred_probs, lengths, n_beam):
        self.max_deg = 8
        batch_size, horizon, vertex = x0_pred_probs.shape
        x = torch.zeros([batch_size, horizon]).long().to(self.device)
        x0_pred_probs = x0_pred_probs.cpu()
        for k in range(batch_size):
            probs, frontiers = x0_pred_probs[k, 0].topk(n_beam)
            probs = probs.cpu().tolist()
            frontiers = frontiers.cpu().tolist()

            length = lengths[k]
            pred = torch.zeros([length, n_beam]).long().to(self.device)
            # traverse along the horizon
            for h in range(1, length):
                new_frontiers = []
                for i, f in enumerate(frontiers):
                    val, ind = self.A[f].topk(self.max_deg)
                    valid_ind = ind[val > 0]  # get connected
                    for adj in valid_ind:
                        new_frontiers.append(((probs[i] * x0_pred_probs[k, h, adj]).item(), adj.item(), f)) # probs, ind, pred
                new_frontiers.sort(reverse=True)
                new_frontiers = new_frontiers[:n_beam]
                frontiers = [each[1] for each in new_frontiers]
                probs = [each[0] for each in new_frontiers]
                pred[h, :] = torch.Tensor([each[2] for each in new_frontiers]).long().to(self.device)

            # sample
            probs = torch.Tensor(probs).to(self.device)
            probs[probs < 1e-5] = 1e-3
            choice = torch.multinomial(probs, num_samples=1).squeeze()
            x[k, :length - 1] = pred[1:, choice]
            x[k, length - 1] = frontiers[choice]
        return x

    def eval_nll(self, real_paths):
        total = len(real_paths)
        nlls = np.zeros(total)
        batch_traj_num = 200
        n_batch = (total + batch_traj_num - 1) // batch_traj_num

        for k in range(n_batch):
            left = k * batch_traj_num
            right = min((k + 1) * batch_traj_num, total)
            batch_size = right - left
            xs = [torch.tensor(path[0]).to(self.device) for path in real_paths[left: right]]
            lengths = [x.shape[0] for x in xs]
            ts = torch.tensor([self.max_T // 20]).repeat(batch_size).to(self.device)
            x_t = self.destroyer.diffusion(xs, ts)
            xt_padded = pad_sequence(x_t, batch_first=True, padding_value=0).long()
            lengths = torch.Tensor(lengths).long().to(self.device)
            _, probs = self.sample_with_len(lengths, ret_distr=True, xt=xt_padded, T=self.max_T // 20)
            for i, path in enumerate(real_paths[left: right]):
                logits = torch.masked_fill(probs_to_logits(probs[i])[1:lengths[i], :], self.A[path[0][:-1]] == 0, -1e20)
                prob = torch.softmax(logits, dim=-1)
                nlls[i + left] -= (prob[0, path[0][0]] + 0.001).log()
                nlls[i + left] -= (prob[torch.arange(lengths[i] - 1), path[0][1:]] + 0.00001).log().sum()
        return nlls


class Discriminator_module(nn.Module):
    def __init__(self, disc_model: Discriminator, destroyer: Destroyer, device):
        super().__init__()
        self.n_vertex = destroyer.n_vertex
        self.disc_model = disc_model
        self.model_device = self.disc_model.device
        self.device = device
        self.destroyer = destroyer
        self.des_device = destroyer.device
        self.max_T = self.destroyer.max_T
        self.matrices = self.destroyer.matrices
        self.A = destroyer.A
        self.Q = self.destroyer.get_Q()
        self.Q = self.Q.to(self.device)
        self.max_deg = self.A.sum(1).max()

        self.applying_mask_intermediate = False
        self.applying_mask_intermediate_temperature = False

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, orgxs, newxs):
        # xs: list of tensors of labels
        batch_size_A = len(orgxs)
        batch_size_new = len(newxs)
        batch_size = batch_size_A + batch_size_new
        if batch_size_A == 0:
            import pdb
            pdb.set_trace()
        lengths = torch.Tensor([x.shape[0] for x in orgxs+newxs]).long().to(self.device)

        # discriminator label
        labels_orgxs = torch.zeros(batch_size_A, dtype=torch.float, device=self.device)
        labels_newxs = torch.ones(batch_size_new, dtype=torch.float, device=self.device)
        labels = torch.cat([labels_orgxs, labels_newxs], dim=0)  # shape: (9,)

        # uniformly choose t
        ts = torch.randint(1, self.max_T + 1, [batch_size]).to(self.device)

        orgx_t = self.destroyer.diffusion(orgxs, ts[:batch_size_A], ret_distr=False)
        newx_t = self.destroyer.diffusion(newxs, ts[batch_size_A:], ret_distr=False)
        orgxt_padded = pad_sequence(orgx_t, batch_first=True, padding_value=0).long()
        newxt_padded = pad_sequence(newx_t, batch_first=True, padding_value=0).long()
        xt_padded = torch.cat((orgxt_padded, newxt_padded), dim=0)

        disc_logits = self.discriminate(xt_padded.to(self.model_device), lengths.to(self.model_device),
                                        ts.to(self.model_device), adj_matrix)
        loss = self.criterion(disc_logits, labels)

        return loss

    def discriminate(self, xt_padded, lengths=None, ts=None, adj_matrix=None):
        # xt_padded: b, h value is vertex number
        # ts: b value is time for each
        batch_size = xt_padded.shape[0]
        if ts is None:
            ts = torch.Tensor([self.max_T]).repeat(batch_size).to(self.device)
        x0_pred_logits = self.disc_model(xt_padded, lengths, ts, adj_matrix)
        return x0_pred_logits
