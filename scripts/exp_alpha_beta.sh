datasets=("higgs")
alphas=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
betas=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
(
for d in "${datasets[@]}"; do
    for alpha in "${alphas[@]}"; do
        for beta in "${betas[@]}"; do
            CUDA_VISIBLE_DEVICES=7 nohup python main.py -d "$d" --alpha "$alpha" --beta "$beta" >> "./logs/exp_alpha_beta_v1.txt" 2>&1
            CUDA_VISIBLE_DEVICES=7 nohup python accuracy_globalsearch.py -d "$d" >> "./logs/exp_alpha_beta_v1.txt" 2>&1
            echo "Running dataset: $d, alpha: $alpha, beta: $beta" >> "./logs/exp_alpha_beta_v1.txt"
        done
    done
done
) &