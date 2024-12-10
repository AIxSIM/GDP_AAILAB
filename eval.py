from os.path import join
import torch
from loader.gen_graph import DataGenerator
from loader.dataset import TrajFastDataset
from utils.argparser import get_argparser
from utils.evaluate import Evaluator


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    parser = get_argparser()
    args = parser.parse_args()

    # set device
    if args.device == "default":
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(device)

    # set dataset
    if args.d_name == "":
        n_vertex = args.n_vertex
        name = f"v{args.n_vertex}_p{args.n_path}_{args.min_len}{args.max_len}"
        dataset = DataGenerator(args.n_vertex, args.n_path, args.min_len, args.max_len, device, args.path, name)
    elif args.d_name != "":
        date = "20190701" if args.d_name == "dj" else "dj"
        dataset = TrajFastDataset(args.d_name, [date], args.path, device, is_pretrain=True)
        n_vertex = dataset.n_vertex
        print(f"vertex: {n_vertex}")

    if args.method != "plan":
        model = torch.load(join(args.model_path, f"{args.model_name}.pth"))
        model = model.to(device)
        model.eval()

        gen_paths = model.sample(args.eval_num)
        real_paths = dataset.get_real_paths(args.eval_num)
        torch.save(gen_paths, join(args.model_path, "gen_paths.pth"))
        evaluator = Evaluator(real_paths, gen_paths, model, n_vertex,
                              name=join(args.res_path, f"{args.model_name}_pure_gen"))
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