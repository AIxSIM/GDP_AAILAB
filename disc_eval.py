from os.path import join
import torch
from loader.gen_graph import DataGenerator
from loader.dataset import TrajFastDataset , TrajFastDataset_SimTime, TrajFastShortestDataset
from utils.argparser import get_argparser
from utils.evaluate import Evaluator
import numpy as np
import random
import time
from models_seq.seq_models import Destroyer

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
        # if args.sim_time == True:
        #     dataset = TrajFastDataset_SimTime(args.d_name, [date], args.path, device, is_pretrain=True)
        # elif args.sim_time == False:
        #     dataset = TrajFastDataset(args.d_name, [date], args.path, device, is_pretrain=True)
        dataset = TrajFastShortestDataset(args.d_name, [date], args.path, device, is_pretrain=True, shuffle=False, index=args.shortest_new_idx, shortest_data_path=args.shortest_data_path)

        n_vertex = dataset.n_vertex
        print(f"vertex: {n_vertex}")

    if args.method != "plan":
        model = torch.load(join(args.model_path, f"{args.model_name}.pth"), map_location=device)
        disc = torch.load(join(args.disc_path, f"{args.disc_name}.pth"), map_location=device)
        model.device = device
        model.eps_model.device = device
        disc.device = device
        # TODO: need to change device for all modules in model
        model.eval()
        disc.eval()

        model.applying_mask_intermediate = args.applying_mask_intermediate
        model.applying_mask_intermediate_temperature = args.applying_mask_intermediate_temperature

        # if args.newA_file != 'None':
        #     new_A = torch.load(args.newA_file, map_location=device)
        # elif args.min_lat != -1:
        #     new_A = model.edit(removal={"regions" : [[[args.min_lat, args.max_lat], [args.min_lng, args.max_lng]]]},
        #                G=dataset.G, direct_change=False)
        # else:
        #     raise NotImplementedError
        # destroyer_new = Destroyer(new_A, model.destroyer.betas, args.max_T, device)
        destroyer_new = None

        # gen_paths: list of lists (len: eval_num, element: list of nodes)
        # real_paths: list of lists (len: eval_num, element: list of nodes)
        real_paths = dataset.get_real_paths(args.eval_num)
        start_time = time.time()
        gen_paths = model.sample_with_disc(args.eval_num, args.batch_traj_num, real_paths=real_paths, destroyer_new=destroyer_new, disc=disc, guidance_scale=args.guidance_scale)
        print(f'Sampling time: {time.time() - start_time} seconds')

        torch.save(gen_paths, join(args.model_path, f"DISC_{args.model_name}_{args.save_name}_gen_paths.pth"))
        evaluator = Evaluator(real_paths, gen_paths, model, n_vertex, dataset=dataset,
                              name=join(args.res_path, f"DISC_{args.model_name}_{args.save_name}_pure_gen"), sim_time=args.sim_time)
        evaluator.eval(suffix=f"DISC_{args.model_name}_{args.save_name}")
        res = evaluator.eval_all()
        print(res)
        with open(join(args.res_path, f"DISC_{args.model_name}_{args.save_name}.res"), "w") as f:
            f.writelines(str(res))

    if args.method == "plan":
        raise NotImplementedError
        # from utils.evaluate_plan import Evaluator

        # suffix = "dj" if args.d_name == "dj" else "dj"
        # evaluator = Evaluator(model, dataset)
        # evaluator.eval(args.eval_num, suffix)