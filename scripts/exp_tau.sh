# datasets=("aucs" "terrorist" "higgs")
# datasets=("aucs" "dblp" "freebase" "3sources" "imdb" "rm" "terrorist" "WikipediaArticles" "higgs")
datasets=("aucs" "terrorist" "rm" "3sources" "WikipediaArticles" "acm" "freebase" "dblp" "imdb" "higgs")
alphas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
(
for d in "${datasets[@]}"; do
    for alpha in "${alphas[@]}"; do
        CUDA_VISIBLE_DEVICES=7 nohup python accuracy_globalsearch.py -d "$d" --tau "$alpha" >> "./logs/exp_tau.txt" 2>&1
        echo "Running dataset: $d, tau: $alpha" >> "./logs/exp_tau.txt"
    done
done
) &