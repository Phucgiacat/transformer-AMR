### New features:
- Integrating external knowledge (**A**bstract **M**eaning **R**epresentation - AMR) into _fairseq_ .
### Usage:
- To install this editable version:
```bash
cd amrNMT/codes
sudo pip3 insall --editable fairseq_g
```
- Preprocessing data:
```bash
!python3 ./preprocess.py --source-lang en \
                         --amr amr \
                         --target-lang vi \
                         --trainpref "tmp/train.bpe" \
                         --validpref "tmp/tst2012.bpe" \
                         --testpref "tmp/tst2013.bpe" \
                         --destdir "preprocessed" 
```
  Just removing ```--amr``` param if you **don't** want to work with AMR.
 
 - Training:
 ```bash
!CUDA_VISIBLE_DEVICES=0 python3 ./train.py "preprocessed" \
                                          --arch lstm  \
                                          --with_amr True \
                                          --aggr maxpooling \
                                          --graph-encoder-embed-dim 128 \
                                          --graph-hidden-dim 256 \
                                          --graph-in-dropout 0.2 \
                                          --concat-in-aggr "True" \
                                          --n-highways 1 \
                                          --n-graph-layers 3 \
                                          --direction bi \
                                          --lr 0.001 \
                                          --clip-norm 0.1 \
                                          --dropout 0.2 \
                                          --max-tokens 3500 \
                                          --criterion label_smoothed_cross_entropy \
                                          --optimizer adam \
                                          --label-smoothing 0.1 \
                                          --lr-scheduler fixed \
                                          --force-anneal 200 \
                                          --save-dir "checkpointsgraph_uni/lstm" \
                                          --max-epoch 10 \
                                          --log-format tqdm \
                                          --log-interval 100 
```
Arguments: \
    ``--with-amr`` : specific that model is trained with AMR. \
    ``--aggr``: aggregator, possible choices: mean, maxpooling, gcn. Default: maxpooling. \
    ``--graph-encoder-embed-dim``: default: 128. \
    ``--graph-hidden-dim``: default: 256. \
    ``--graph-in-dropout 0.2``: default: 0.2. \
    ``--concat-in-aggr``: concat or add result in aggregators. Default: True \
    ``--with-highway``: use highway layers as bridge between embedding layer and learning phase. Default: True \
    ``--n-highways``: number of highway layers. Default: 1 \
    ``--n-graph-layers``: number of graph layers. Default: 2 \
    ``--direction``: possible choices: fw, bw, bi. Default: bi. \
    **Note**:  ``--concat-in-aggr`` must be False when setting ``--aggr`` = gcn
 - Generating:
 ```bash
!python3 ./generate.py "preprocessed" 
                        --with-amr True \
                        --path "checkpointsgraph_uni/lstm/checkpoint_best.pt" \
                        --batch-size 128 \
                        --beam 5 \
                        --remove-bpe  
```

Removing ```--with-amr``` or setting to False when the model does not support AMR.
