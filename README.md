# **GDP_AAILAB**

This is the code complemented by AAILAB in SyntheticData project.

## Directories
```
.
└── Graph-Diffusion-Planning/
    ├── loader/
    │   ├── preprocess/
    │   │   └── mm/
    │   │       ├── fetch_rdnet.py
    │   │       ├── mapmatching.py
    │   │       ├── process_all.py
    │   │       ├── refine_gps.py
    │   │       └── utils.py
    │   ├── dataset.py
    │   ├── gen_graph.py
    │   └── node2vec.py
    ├── models_seq/
    │   ├── blocks.py
    │   ├── eps_models.py
    │   ├── seq_models.py
    │   └── trainer.py
    ├── planner/
    │   ├── planner.py
    │   └── trainer.py
    ├── sets_data
    ├── sets_model
    ├── sets_res
    ├── utils/
    │   ├── argparser.py
    │   ├── coors.py
    │   ├── evaluate.py
    │   ├── evaluate_plan.py
    │   ├── evaluate_plan_dtw.py
    │   ├── fetch_navi.py
    │   └── visual.py
    ├── figs
    ├── main.py
    └── train.sh
```

## Data Preparation
Directories regarding data preparations are as follows: 
```bash
sets_data/
  |--real/
    |--map/
    |--raw/
    |--trajectories/
```
The GPS file under directory ``raw/``. The GPS file should contains ``[index, path_id, timestamp, longitude, latitude]``. 
Using the min, max of ``[longitude, latitude]``, update the list variables ``bounds`` in ``Graph-Diffusion-Planning/loader/preprocess/mm/process_all.py``

Then, execute the code below for data preparation.
```bash
python -m loader.preprocess.mm.process_all
```

## Model Training
You can train the model by executing ``bash ./train.sh``.
Then two commands are executed sequentially.
(1) No plan generation 
```python
python3 main.py -device "cuda" -path "./sets_data" -model_path "./sets_model" -res_path "./sets_res" -d_name "dj"  -model_name "no_plan_gen_dj" -method "seq" -beta_lb 0.0001 -beta_ub 10 -max_T 100 -gmm_comp 5 -dims "[100, 120, 200]" -hidden_dim 32 -n_epoch 100 -bs 16 -lr 0.005 -gmm_samples 100000 -eval_num 2000
```
(2) Planning
```python
python3 main.py -device "default" -path "./sets_data" -model_path "./sets_model" -res_path "./sets_res" \
    -d_name "<d_name>"  -model_name "plan_dj" -method "plan" \
    -x_emb_dim 100 -n_epoch 5 -bs 32 -lr 0.001  -eval_num 100
```
As a result, you can find the model at ``sets_models/`` and the evaluation results at ``sets_res/``. The result figure of planning is located in ``figs/``

## Model Evaluation
```python
python eval.py -device "cuda" -path "./sets_data" -model_path "./sets_model" -res_path "./sets_res" -d_name "dj"  -model_name "no_plan_gen_dj" -method "seq" -eval_num 2000
```

## Acknowledgements
This work is heavily built upon the code from GRAPH-CONSTRAINED DIFFUSION FOR END-TO-END PATH PLANNING in [ICLR 2024](https://iclr.cc/virtual/2024/poster/17513). Our reference codes are available in [github](https://github.com/dingyuan-shi/Graph-Diffusion-Planning).
