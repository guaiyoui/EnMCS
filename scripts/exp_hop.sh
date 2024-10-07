datasets=("aucs" "terrorist" "rm" "3sources" "WikipediaArticles" "acm" "freebase" "dblp" "imdb" "higgs")
# datasets=("aucs")
hops=(1 2 3 4 5)
(
sleep 3600
for d in "${datasets[@]}"; do
    for hop in "${hops[@]}"; do
            CUDA_VISIBLE_DEVICES=7 nohup python main.py -d "$d" --hops "$hop" >> "./logs/exp_hop.txt" 2>&1
            CUDA_VISIBLE_DEVICES=7 nohup python accuracy_globalsearch.py -d "$d" >> "./logs/exp_hop.txt" 2>&1
            echo "Running dataset: $d, hop: $hop" >> "./logs/exp_hop.txt"
    done
done
) &