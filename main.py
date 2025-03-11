from os.path import join
import torch
from loader.gen_graph import DataGenerator
from loader.dataset import TrajFastDataset, TrajFastDataset_SimTime
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

    # before train, record the infob
    with open(join(args.model_path, f"{args.model_name}.info"), "w") as f:
        f.writelines(str(args))

    # set model
    if args.method == "seq":
        from models_seq.seq_models import Destroyer, Restorer, Restorer_SimTime
        from models_seq.eps_models import EPSM, EPSM_SimTime
        from models_seq.trainer import Trainer, Trainer_SimTime

        suffix = args.d_name

        betas = torch.linspace(args.beta_lb, args.beta_ub, args.max_T)
        destroyer = Destroyer(dataset.A, betas, args.max_T, device)
        pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")
        dims = eval(args.dims)

        ######################################################## unconditional EPSM ####################################################################
        if not args.sim_time:
            eps_model = EPSM(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=dims, device=device,
                            hidden_dim=args.hidden_dim, pretrain_path=pretrain_path)
            model = Restorer(eps_model, destroyer, device)
            trainer = Trainer(model, dataset, args.model_path)

        ##################################################################################################################################################


        ############################################ simulation-time conditioned EPSM ####################################################################
        elif args.sim_time:
            eps_model = EPSM_SimTime(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=dims, device=device,
                            hidden_dim=args.hidden_dim, pretrain_path=pretrain_path)

            model = Restorer_SimTime(eps_model, destroyer, device)

            trainer = Trainer_SimTime(model, dataset, args.model_path)
        ##################################################################################################################################################

        trainer.train_gmm(gmm_samples=args.gmm_samples, n_comp=args.gmm_comp)
        trainer.train(args.n_epoch, args.bs, args.lr)
        model.eval()
        torch.save(model, join(args.model_path, f"{args.model_name}.pth"))
        model.eval()

    elif args.method == "plan":
        from planner.planner import Planner
        from planner.trainer import Trainer

        suffix = args.d_name

        pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")
        restorer = torch.load(f"./sets_model/no_plan_gen_{suffix}.pth")
        destroyer = restorer.destroyer
        model = Planner(dataset.G, dataset.A, restorer, destroyer, device, x_emb_dim=args.x_emb_dim,
                        pretrain_path=pretrain_path)
        trainer = Trainer(model, dataset, device, args.model_path)
        trainer.train(args.n_epoch, args.bs, args.lr)
        model.eval()
        torch.save(model, join(args.model_path, f"{args.model_name}.pth"))

    if args.method != "plan":
        gen_paths = model.sample(args.eval_num)
        real_paths = dataset.get_real_paths(args.eval_num)
        torch.save(gen_paths, join(args.model_path, "gen_paths.pth"))
        evaluator = Evaluator(real_paths, gen_paths, model, n_vertex, dataset=dataset,
                              name=join(args.res_path, f"{args.model_name}_pure_gen"), sim_time = args.sim_time)
        evaluator.eval(suffix=args.d_name)
        res = evaluator.eval_all()
        print(res)
        with open(join(args.res_path, f"{args.model_name}.res"), "w") as f:
            f.writelines(str(res))

    if args.method == "plan":
        from utils.evaluate_plan import Evaluator

        suffix = args.d_name
        evaluator = Evaluator(model, dataset)
        evaluator.eval(args.eval_num, suffix)