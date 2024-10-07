# datasets=("aucs" "dblp" "freebase" "3sources" "acm" "imdb" "rm" "terrorist" "WikipediaArticles" "higgs")
# datasets=("rm" "terrorist")
datasets=("aucs" "terrorist" "rm" "3sources" "WikipediaArticles" "acm" "freebase" "dblp" "imdb" "higgs")
# datasets=("aucs" "dblp")
(
for d in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=5 nohup python accuracy_globalsearch.py -d "$d" >> "./logs/exp_varyinglambda.txt" 2>&1
    echo "Running dataset: $d" >> "./logs/exp_varyinglambda.txt"
done
) &