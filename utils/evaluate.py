from typing import Any
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from utils.visual import draw_heatmap

class Evaluator:
    def __init__(self, real_paths, gen_paths, model, n_vertex, dataset, name="e1", sim_time=False) -> None:
        self.real_paths = real_paths
        self.gen_paths = gen_paths
        self.n_vertex = n_vertex
        self.model = model
        self.name = name
        self.dataset = dataset
        self.sim_time = sim_time

    @staticmethod
    def JS_divergence(p, q):
        M = (p + q)/2
        return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
    
    @staticmethod
    def KL_divergence(p,q):
        return scipy.stats.entropy(p, q)
    
    def calculate_divergences(self):
        real_edge_distr = np.zeros((self.n_vertex, self.n_vertex))
        gen_edge_distr = np.zeros((self.n_vertex, self.n_vertex))
        
        real_len_distr = np.zeros(self.n_vertex + 1)
        gen_len_distr = np.zeros(self.n_vertex + 1)
        
        for path in self.real_paths:
            if self.sim_time:
                path = path[0]
            for a, b in zip(path, path[1:]):
                real_edge_distr[a][b] += 1
            real_len_distr[len(path)] += 1
            
        for path in self.gen_paths:
            for a, b in zip(path, path[1:]):
                gen_edge_distr[a][b] += 1
            gen_len_distr[len(path)] += 1
                
        real_edge_distr /= np.sum(real_edge_distr)
        gen_edge_distr /= np.sum(real_edge_distr)
        real_len_distr /= np.sum(real_len_distr)
        gen_len_distr /= np.sum(gen_len_distr)
        
        edge_distr_kl = Evaluator.KL_divergence(real_edge_distr.reshape(-1) + 1e-5, gen_edge_distr.reshape(-1) + 1e-5)
        edge_distr_js = Evaluator.JS_divergence(real_edge_distr.reshape(-1) + 1e-5, gen_edge_distr.reshape(-1) + 1e-5)
    
        
        res_dict = {
            "KLEV": edge_distr_kl, 
            "JSEV": edge_distr_js, 
        }
        
        plt.plot(real_len_distr)
        plt.plot(gen_len_distr)
        plt.legend(["real", "gen"])
        plt.savefig(f"{self.name}_a.pdf")
        plt.clf()        
        
        return res_dict
    
    def calculate_nll(self):
        nlls = self.model.eval_nll(self.real_paths)
        nll_min = np.min(nlls)
        nll_max = np.max(nlls)
        nll_avg = np.mean(nlls)
        res_dict = {
            "nll_avg": nll_avg,
            "nll_min": nll_min, 
            "nll_max": nll_max, 
        }
        return res_dict
    
    def eval_all(self):
        div_dict = self.calculate_divergences()
        nll_dict = self.calculate_nll()
        return dict(div_dict, **nll_dict)

    def _convert_from_id_to_lat_lng(self, paths, sim_time=False):
        path_coors = []
        for path in paths:
            if sim_time:
                path_coors.append([[self.dataset.G.nodes[v]["lat"], self.dataset.G.nodes[v]["lng"]] for v in path[0]])
            else:
                path_coors.append([[self.dataset.G.nodes[v]["lat"], self.dataset.G.nodes[v]["lng"]] for v in path])
        return path_coors

    def eval(self, suffix):
        planned_paths_coors = self._convert_from_id_to_lat_lng(self.gen_paths, False)
        gen_path_count = draw_heatmap(planned_paths_coors, f"./figs/seq_gen_{suffix}.html", colors=["red"] * len(planned_paths_coors), no_points=False)
        orig_paths_coors = self._convert_from_id_to_lat_lng(self.real_paths, self.sim_time)
        orig_path_count = draw_heatmap(orig_paths_coors, f"./figs/seq_real_{suffix}.html", colors=["blue"] * len(orig_paths_coors), no_points=False)
        average_mse = 0.
        for key in gen_path_count.keys():
            if key in orig_path_count:
                value1 = gen_path_count[key] / len(planned_paths_coors)
                value2 = orig_path_count[key] / len(orig_path_count)
                average = (value1 - value2) ** 2
                average_mse += average
        average_mse = average_mse / len(gen_path_count)
        print(average_mse)
        return average_mse

