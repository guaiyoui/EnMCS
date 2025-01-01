Code for the paper "Ensemble-based Deep Multilayer Community Search" which is submitted for reviewing in VLDB 2025.


### Training && Search in HoloSearch and EMerge
```
bash ./run_EnMCS.sh
```


### Search and EMerge
```
bash ./run_EnMCS_Search.sh
```

### Reproduce the experiments
```
bash ./scripts/exp_xxx.sh
```

### Folder Structure

    .
    ├── dataset                             # the folder containing all the datasets
    ├── models                              # 
    ├── logs                                # the running logs
    ├── scripts                             # the scripts to run the model and to reproduce the experiments
    ├── accelerate_EM.py                    # the Emerge function
    ├── accuracy_globalsearch_parallel.py   # online parallel community identification
    ├── accuracy_globalsearch.py            # online community identification
    ├── args.py                             # the setting and hyperparameters of the model
    ├── DMG.py                              # the training
    ├── EM.py                               # the Emerge function
    ├── embedder.py                         # the pre-training function
    ├── evaluate.py                         # the evaluate function
    ├── functions.py                        # the help functions
    ├── main.py                             # the overall entrance of the framework
    ├── mat2txt.py                          # the datasets pre-processing module
    ├── model.py                            # the graph diffusion encoder model module
    ├── multilayer.py                       # the data structure of multilayer graph
    └── README.md
