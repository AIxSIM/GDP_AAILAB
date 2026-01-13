from typing import Any
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from utils.visual import draw_heatmap, draw_paths

class Evaluator:
    def __init__(self, real_paths, gen_paths, model, n_vertex, dataset, name="e1", sim_time=False, A=None, removal=None) -> None:
        self.real_paths = real_paths
        self.gen_paths = gen_paths
        self.n_vertex = n_vertex
        self.model = model
        self.name = name
        self.dataset = dataset
        self.sim_time = sim_time
        self.A = A
        self.removal = removal

    @staticmethod
    def JS_divergence(p, q):
        M = (p + q)/2
        return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
    
    @staticmethod
    def KL_divergence(p,q):
        return scipy.stats.entropy(p, q)
    
    def calculate_divergences(self):
        import pdb
        pdb.set_trace()
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
        nlls = self.model.eval_nll_fix(self.real_paths)
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

    def A_vis(self, suffix):
        A_idx = (self.A != 0).nonzero(as_tuple=False).tolist()
        A_coors = self._convert_from_id_to_lat_lng(A_idx, False)
        A_highlight_coors = self._convert_from_id_to_lat_lng(self.removal["edges_reverse"], False)
        A_count = draw_heatmap(A_coors, f"./figs/seq_A_{suffix}.html", colors=["blue"], no_points=False, weight=3, highlight=A_highlight_coors)

    def eval(self, suffix):

        # x_min, x_max = 36.361, 36.362
        # y_min, y_max = 127.3575, 127.3585
        #
        # # Finding the path index and coordinate index
        # indices_in_range = []
        # for path_index, path in enumerate(planned_paths_coors):
        #     for coord_index, (x, y) in enumerate(path):
        #         if x_min <= x <= x_max and y_min <= y <= y_max:
        #             indices_in_range.append((path_index, coord_index))
        # unique_path_indices = sorted(set([path_index for path_index, _ in indices_in_range]))

        # path draw
        idx_for_analysis = [189, 346, 414, 434, 435, 458, 459, 473, 492, 532, 650, 655, 662, 718, 725, 743, 764, 765, 773, 800, 812, 878, 895, 923, 939, 953, 964, 971, 984, 987, 989, 995, 1000, 1070, 1073, 1094, 1106, 1108, 1128, 1131, 1136, 1167, 1176, 1186, 1192, 1193, 1240, 1250, 1274, 1276, 1302, 1312, 1319, 1325, 1333, 1353, 1359, 1366, 1371, 1381, 1405, 1410, 1451, 1459, 1463, 1472, 1491, 1494, 1512, 1521, 1537, 1544, 1551, 1555, 1578, 1606, 1610, 1629, 1661, 1666, 1672, 1706, 1719, 1743, 1763, 1775, 1794, 1797, 1815, 1818, 1819, 1845, 1877, 1907, 1923, 1929, 1947, 1950, 1951, 1963]
        filtered_paths = [self.gen_paths[i] for i in idx_for_analysis]
        for i in range(len(idx_for_analysis)):
            draw_paths([filtered_paths[i]], self.dataset.G, f"./figs_path_analysis/PATH_{i}_seq_gen_{suffix}.html")
        real_filtered_paths = [self.real_paths[i] for i in idx_for_analysis]
        for i in range(len(idx_for_analysis)):
            try:
                draw_paths([real_filtered_paths[i]], self.dataset.G, f"./figs_path_analysis/PATH_{i}_seq_real_{suffix}.html")
            except:
                print(f'Loop! ./figs_path_analysis/PATH_{i}_seq_real_{suffix}.html')

        planned_paths_coors = self._convert_from_id_to_lat_lng(self.gen_paths, False)
        gen_path_count = draw_heatmap(planned_paths_coors, f"./figs/seq_gen_{suffix}.html", colors=["red"] * len(planned_paths_coors), no_points=False)
        orig_paths_coors = self._convert_from_id_to_lat_lng(self.real_paths, self.sim_time)
        orig_path_count = draw_heatmap(orig_paths_coors, f"./figs/seq_real_{suffix}.html", colors=["blue"] * len(orig_paths_coors), no_points=False)
        average_mse = 0.
        for key in gen_path_count.keys():
            if key in orig_path_count:
                value1 = gen_path_count[key]#  / len(planned_paths_coors)
                value2 = orig_path_count[key]#   / len(orig_path_count)
                average = max((value1 - value2), (value2 - value1))
                # print(average)
                average_mse += average
        average_mse = average_mse / len(gen_path_count)
        print('average_mse :', average_mse)

        x = np.array([gen_path_count[key] for key in gen_path_count if key in orig_path_count])
        y = np.array([orig_path_count[key] for key in gen_path_count if key in orig_path_count])

        mean_y = np.mean(y)
        ss_total = np.sum((y - mean_y) ** 2)
        ss_residual = np.sum((y - x) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        print('r_squared :', r_squared)

        return average_mse

