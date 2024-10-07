datasets=("aucs" "terrorist" "rm" "3sources" "WikipediaArticles" "acm" "freebase" "dblp" "imdb" "higgs")
alphas=(0.0)
betas=(0.4)
(
for d in "${datasets[@]}"; do
    for alpha in "${alphas[@]}"; do
        for beta in "${betas[@]}"; do
            CUDA_VISIBLE_DEVICES=7 nohup python main.py -d "$d" --alpha "$alpha" --beta "$beta" >> "./logs/exp_wo_Lmat_v1.txt" 2>&1
            CUDA_VISIBLE_DEVICES=7 nohup python accuracy_globalsearch.py -d "$d" >> "./logs/exp_wo_Lmat_v1.txt" 2>&1
            echo "Running dataset: $d, alpha: $alpha, beta: $beta" >> "./logs/exp_wo_Lmat_v1.txt"
        done
    done
done
) &