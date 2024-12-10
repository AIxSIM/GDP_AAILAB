from typing import Any
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from utils.visual import draw_gps

class Evaluator:
    def __init__(self, real_paths, gen_paths, model, n_vertex, name="e1") -> None:
        self.real_paths = real_paths
        self.gen_paths = gen_paths
        self.n_vertex = n_vertex
        self.model = model
        self.name = name
        
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

    def _convert_from_id_to_lat_lng(self, paths):
        path_coors = []
        for path in paths:
            path_coors.append([[self.dataset.G.nodes[v]["lat"], self.dataset.G.nodes[v]["lng"]] for v in path])
        return path_coors

    def eval(self, planned_paths, orig_paths, suffix):
        planned_paths_coors = self._convert_from_id_to_lat_lng(self.gen_paths)
        draw_gps(planned_paths_coors, f"./figs/seq_gen_{suffix}.html", colors=["red"] * 10, no_points=False)
        orig_paths_coors = self._convert_from_id_to_lat_lng(self.real_paths)
        draw_gps(orig_paths_coors, f"./figs/seq_real_{suffix}.html", colors=["blue"] * 10, no_points=False)
