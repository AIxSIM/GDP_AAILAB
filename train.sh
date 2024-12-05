# more hyper parameter setting please refer to utils/argparser.py


# no plan gen
python3 main.py -device "default" -path "./sets_data" -model_path "./sets_model" -res_path "./sets_res" \
        -d_name "dj"  -model_name "no_plan_gen_dj" -method "seq" \
        -beta_lb 0.0001 -beta_ub 10 -max_T 100 -gmm_comp 5 -dims "[100, 120, 200]" -hidden_dim 32 \
        -n_epoch 5 -bs 16 -lr 0.0005 -gmm_samples 100000 -eval_num 2000 # org : 5000
#

# planning
python3 main.py -device "default" -path "./sets_data" -model_path "./sets_model" -res_path "./sets_res" \
        -d_name "dj"  -model_name "plan_dj" -method "plan" \
        -x_emb_dim 100 -n_epoch 5 -bs 32 -lr 0.001  -eval_num 100
