datasets=("aucs" "dblp" "freebase" "3sources" "imdb" "rm" "terrorist" "WikipediaArticles" "higgs")
# datasets=("aucs" "dblp")
(
for d in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=5 nohup python accuracy_globalsearch_vote.py -d "$d" >> "./logs/exp_wo_EM.txt" 2>&1
done
) &