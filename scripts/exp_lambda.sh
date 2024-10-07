# datasets=("aucs" "dblp" "freebase" "3sources" "imdb" "rm" "terrorist" "WikipediaArticles" "higgs")
datasets=("3sources")
(
for d in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=5 nohup python accuracy_globalsearch.py -d "$d" >> "./logs/exp_lamda.txt" 2>&1
done
) &