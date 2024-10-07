# datasets=("aucs" "dblp" "freebase" "3sources" "acm" "imdb" "rm" "terrorist" "WikipediaArticles" "higgs")
# datasets=("rm" "terrorist")
datasets=("aucs" "terrorist" "rm" "3sources" "WikipediaArticles" "acm" "freebase" "dblp" "imdb" "higgs")
# datasets=("aucs" "dblp")
(
for d in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=5 nohup python main.py -d "$d" --alpha 0.8 --beta 0.4 >> "./logs/${d}_EnMCS.txt" 2>&1
    CUDA_VISIBLE_DEVICES=5 nohup python accuracy_globalsearch.py -d "$d" >> "./logs/${d}_EnMCS.txt" 2>&1
done
) &