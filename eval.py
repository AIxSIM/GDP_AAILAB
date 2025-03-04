from os.path import join
import torch
from loader.gen_graph import DataGenerator
from loader.dataset import TrajFastDataset , TrajFastDataset_SimTime
from utils.argparser import get_argparser
from utils.evaluate import Evaluator


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    parser = get_argparser()
    args = parser.parse_args()

    # set device
    if args.device == "default":
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(device)

    # set dataset
    if args.d_name == "":
        n_vertex = args.n_vertex
        name = f"v{args.n_vertex}_p{args.n_path}_{args.min_len}{args.max_len}"
        dataset = DataGenerator(args.n_vertex, args.n_path, args.min_len, args.max_len, device, args.path, name)
    elif args.d_name != "":
        date = "20190701" if "dj" in args.d_name else "dj"
        if args.sim_time == True:
            dataset = TrajFastDataset_SimTime(args.d_name, [date], args.path, device, is_pretrain=True)
        elif args.sim_time == False:
            dataset = TrajFastDataset(args.d_name, [date], args.path, device, is_pretrain=True)
    
        n_vertex = dataset.n_vertex
        print(f"vertex: {n_vertex}")

    if args.method != "plan":
        model = torch.load(join(args.model_path, f"{args.model_name}.pth"), map_location=device)
        model.device = device
        model.eps_model.device = device
        # TODO: need to change device for all modules in model
        model.eval()

        # gen_paths: list of lists (len: eval_num, element: list of nodes)
        # real_paths: list of lists (len: eval_num, element: list of nodes)
        gen_paths = model.sample(args.eval_num)
        for i in range(24):
            gen_paths_sim_time = model.sample(args.eval_num, sim_time = i)
            gen_average_len = 0
            for path in gen_paths_sim_time:
                gen_average_len += len(path)
            gen_average_len /= args.eval_num
            print(f"gen_average_len_sim_time={i}: {gen_average_len}")
            real_average_len = 0    
            real_paths_sim_time = dataset.get_real_paths_sim_time(sim_time = i, num = args.eval_num)
            for path in real_paths_sim_time:
                real_average_len += len(path[0])
            real_average_len /= len(real_paths_sim_time)
            print(f"real_average_len_sim_time={i}: {real_average_len}")
            torch.save(gen_paths, join(args.model_path, f"gen_paths_sim_time={i}.pth"))
            evaluator = Evaluator(real_paths_sim_time, gen_paths_sim_time, model, n_vertex, dataset=dataset,
                                    name=join(args.res_path, f"{args.model_name}_pure_gen_{i}"), sim_time=args.sim_time)
            evaluator.eval(suffix=args.d_name+f"_sim_time={i}")

        real_paths = dataset.get_real_paths(args.eval_num)

        torch.save(gen_paths, join(args.model_path, "gen_paths.pth"))
        evaluator = Evaluator(real_paths, gen_paths, model, n_vertex, dataset=dataset,
                              name=join(args.res_path, f"{args.model_name}_pure_gen"), sim_time=args.sim_time)
        evaluator.eval(suffix=args.d_name)
        res = evaluator.eval_all()
        print(res)
        with open(join(args.res_path, f"{args.model_name}.res"), "w") as f:
            f.writelines(str(res))

    if args.method == "plan":
        raise NotImplementedError
        # from utils.evaluate_plan import Evaluator

        # suffix = "dj" if args.d_name == "dj" else "dj"
        # evaluator = Evaluator(model, dataset)
        # evaluator.eval(args.eval_num, suffix)