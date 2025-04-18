import pickle
from torch.utils.data import Dataset
import h5py
from os.path import join, exists
import torch
import numpy as np
import networkx as nx
from loader.node2vec import get_node2vec


class TrajFastDataset(Dataset):
    def __init__(self, city, dates, path, device, is_pretrain):
        super().__init__()
        name = city
        self.device = device
        
        shrink_G_path = join(path, f"{name}_shrink_G.pkl")
        shrink_A_path = join(path, f"{name}_shrink_A.ts")
        shrink_NZ_path = join(path, f"{name}_shrink_NZ.pkl")
        
        if exists(shrink_G_path):
            print("loading")
            self.G = pickle.load(open(shrink_G_path, "rb"))
            self.A = pickle.load(open(shrink_A_path, "rb"))
            self.shrink_nonzero_dict = pickle.load(open(shrink_NZ_path, "rb"))
            print("finished")
        else:
            self.G = pickle.load(open(join(path, f"{name}_G.pkl"), "rb"))
            self.n_vertex = len(self.G.nodes)
            self.A_orig = torch.load(join(path, f"{name}_A.ts"), map_location=torch.device("cpu"))
            print("loading path...")
            self.v_paths = np.loadtxt(join(path, f"{name}_v_paths.csv"), delimiter=',') 
            print("finish loading")
            nonzeros = np.nonzero(self.v_paths.sum(0))[0]
            self.nonzeros = nonzeros
            print(f"shrink into {nonzeros.shape[0]} nodes")
            B = self.A_orig[nonzeros, :]
            self.A = B[:, nonzeros]
            self.v_paths = self.v_paths[:, nonzeros]
            self.length = self.v_paths.shape[0]
            self.shrink_nonzero_dict = dict()
            for k in range(nonzeros.shape[0]):
                self.shrink_nonzero_dict[nonzeros[k]] = k
        
            # shrink G
            G_shrink = nx.Graph()
            shrink_node_attrs = [(k, {"lat": self.G.nodes[nonzeros[k]]["lat"], "lng": self.G.nodes[nonzeros[k]]["lng"]}) for k in range(self.nonzeros.shape[0])]
            G_shrink.add_nodes_from(shrink_node_attrs)
            for i in range(self.A.shape[0]):
                for j in range(self.A.shape[0]):
                    if self.A[i, j] > 0.5:
                        G_shrink.add_edge(i, j)
            self.G = G_shrink
            self.A = self.A.to(self.device)
            print("finish shrink")
            pickle.dump(self.G, open(shrink_G_path, "wb"))
            pickle.dump(self.A, open(shrink_A_path, "wb"))
            pickle.dump(self.shrink_nonzero_dict, open(shrink_NZ_path, "wb"))
        
        
        self.n_vertex = len(self.G.nodes)
        self.dates = dates
        h5_file = join(path, f"{city}_h5_paths.h5")
        self.f = h5py.File(h5_file, "r")
        sample_len = [self.f[date]["state_prefix"].shape[0] - 1 for date in dates]
        accu_len = [0 for _ in range(len(sample_len) + 1)]
        for k, l in enumerate(sample_len):
            accu_len[k + 1] = accu_len[k] + l
        self.accu_len = accu_len
        self.total_len = accu_len[-1]
        # if pretrain
        if is_pretrain:
            embed_path = join(path, f"{city}_node2vec.pkl")
            path_path = join(path, f"{city}_path.pkl")
            get_node2vec(self.G, embed_path, path_path)
        
    def __upper_bound(self, num):
        l, r = 0, len(self.accu_len)
        while l < r:
            m = (l + r) // 2
            if self.accu_len[m] <= num:
                l = m + 1
            else:
                r = m
        return l
            
    def __getitem__(self, index):
        idx = self.__upper_bound(index) - 1
        date = self.dates[idx]
        offset = index - self.accu_len[idx]
        pleft, pright = self.f[date]["state_prefix"][offset], self.f[date]["state_prefix"][offset + 1]
        # return self.__filter(self.f[date]["states"][pleft: pright])
        return [self.shrink_nonzero_dict[node] for node in self.f[date]["states"][pleft: pright]]
        
    def __len__(self):
        return self.total_len
    
    def __filter(self, points):
        points_filtered = []
        
        showup = set()
        for k, node in enumerate(points):
            node = self.shrink_nonzero_dict[node]
            if node not in showup:
                showup.add(node)
                points_filtered.append(node)
            else:
                while points_filtered[-1] != node:
                    showup.discard(points_filtered[-1])
                    points_filtered.pop()
        return points_filtered
    
    def get_real_paths(self, num=500): ## org : num=500
        choices = np.random.choice(a=self.total_len, size=num, replace=False).tolist() # org : replace = False / 비복원추출
        return [self.__getitem__(c) for c in choices]

    def edit(self, removal=None, is_random=False, direct_change=False): # removal : {"nodes": [xxx, yyy, zzz], "edges": [[XXX, YYY], [ZZZ, WWW]], "regions" : list of [[min_lat, max_lat], [min_lng, max_lng]]}
        if (removal is None) and (not is_random):
            exit("Please check edit in dataset.py")

        if is_random:
            size = 0.01
            min_lat, max_lat = 999, -999
            min_lng, max_lng = 999, -999
            for node, data in self.G.nodes(data=True):
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
            for node, data in self.G.nodes(data=True):
                lat, lng = data.get("lat"), data.get("lng")
                for lat_range, lng_range in removal["regions"]:
                    if lat_range[0] <= lat <= lat_range[1] and lng_range[0] <= lng <= lng_range[1]:
                        new_A[node, :], new_A[:, node] = 0, 0
                        break

        assert torch.all(new_A.transpose(0, 1) == new_A)

        if direct_change:
            self.A.data = new_A.data
        else:
            return new_A


class TrajFastDataset_SimTime(Dataset):
    def __init__(self, city, dates, path, device, is_pretrain):
        super().__init__()
        name = city
        self.device = device

        shrink_G_path = join(path, f"{name}_shrink_G.pkl")
        shrink_A_path = join(path, f"{name}_shrink_A.ts")
        shrink_NZ_path = join(path, f"{name}_shrink_NZ.pkl")

        if exists(shrink_G_path):
            print("loading")
            self.G = pickle.load(open(shrink_G_path, "rb"))
            self.A = pickle.load(open(shrink_A_path, "rb"))
            self.shrink_nonzero_dict = pickle.load(open(shrink_NZ_path, "rb"))
            print("finished")
        else:
            self.G = pickle.load(open(join(path, f"{name}_G.pkl"), "rb"))
            self.n_vertex = len(self.G.nodes)
            self.A_orig = torch.load(join(path, f"{name}_A.ts"), map_location=torch.device("cpu"))
            print("loading path...")
            self.v_paths = np.loadtxt(join(path, f"{name}_v_paths.csv"), delimiter=',')
            print("finish loading")
            nonzeros = np.nonzero(self.v_paths.sum(0))[0]
            self.nonzeros = nonzeros
            print(f"shrink into {nonzeros.shape[0]} nodes")
            B = self.A_orig[nonzeros, :]
            self.A = B[:, nonzeros]
            self.v_paths = self.v_paths[:, nonzeros]
            self.length = self.v_paths.shape[0]
            self.shrink_nonzero_dict = dict()
            for k in range(nonzeros.shape[0]):
                self.shrink_nonzero_dict[nonzeros[k]] = k

            # shrink G
            G_shrink = nx.Graph()
            shrink_node_attrs = [(k, {"lat": self.G.nodes[nonzeros[k]]["lat"], "lng": self.G.nodes[nonzeros[k]]["lng"]}) for k in range(self.nonzeros.shape[0])]
            G_shrink.add_nodes_from(shrink_node_attrs)
            for i in range(self.A.shape[0]):
                for j in range(self.A.shape[0]):
                    if self.A[i, j] > 0.5:
                        G_shrink.add_edge(i, j)
            self.G = G_shrink
            self.A = self.A.to(self.device)
            print("finish shrink")
            pickle.dump(self.G, open(shrink_G_path, "wb"))
            pickle.dump(self.A, open(shrink_A_path, "wb"))
            pickle.dump(self.shrink_nonzero_dict, open(shrink_NZ_path, "wb"))


        self.n_vertex = len(self.G.nodes)
        self.dates = dates
        h5_file = join(path, f"{city}_h5_paths.h5")
        self.f = h5py.File(h5_file, "r")
        sample_len = [self.f[date]["state_prefix"].shape[0] - 1 for date in dates]
        accu_len = [0 for _ in range(len(sample_len) + 1)]
        for k, l in enumerate(sample_len):
            accu_len[k + 1] = accu_len[k] + l
        self.accu_len = accu_len
        self.total_len = accu_len[-1]
        # if pretrain
        if is_pretrain:
            embed_path = join(path, f"{city}_node2vec.pkl")
            path_path = join(path, f"{city}_path.pkl")
            print(embed_path)

            get_node2vec(self.G, embed_path, path_path)

    def __upper_bound(self, num):
        l, r = 0, len(self.accu_len)
        while l < r:
            m = (l + r) // 2
            if self.accu_len[m] <= num:
                l = m + 1
            else:
                r = m
        return l

    def __getitem__(self, index):
        idx = self.__upper_bound(index) - 1
        date = self.dates[idx]
        offset = index - self.accu_len[idx]
        pleft, pright = self.f[date]["state_prefix"][offset], self.f[date]["state_prefix"][offset + 1]
        sim_time = self.f[date]["sim_times"][offset]
        # return self.__filter(self.f[date]["states"][pleft: pright])
        return [self.shrink_nonzero_dict[node] for node in self.f[date]["states"][pleft: pright]], sim_time

    def __len__(self):
        return self.total_len

    def __filter(self, points):
        points_filtered = []

        showup = set()
        for k, node in enumerate(points):
            node = self.shrink_nonzero_dict[node]
            if node not in showup:
                showup.add(node)
                points_filtered.append(node)
            else:
                while points_filtered[-1] != node:
                    showup.discard(points_filtered[-1])
                    points_filtered.pop()
        return points_filtered

    def get_real_paths(self, num=500): ## org : num=500
        choices = np.random.choice(a=self.total_len, size=num, replace=False).tolist() # org : replace = False / 비복원추출
        return [self.__getitem__(c) for c in choices]

    def get_real_paths_sim_time(self, sim_time, num=500):
        return_list = []
        for i in range(self.total_len):
            if self.f[self.dates[0]]["sim_times"][i] == sim_time:
                return_list.append(self.__getitem__(i))
        if len(return_list) < num:
            return return_list
        else:
            choices = np.random.choice(a=len(return_list), size=num, replace=False).tolist()
            return [return_list[c] for c in choices]
