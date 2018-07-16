# Introduction
This is PointSQL, the source codes of [Natural Language to Structured Query Generation via Meta-Learning](https://arxiv.org/abs/1803.02400) 
and [Pointing Out SQL Queries From Text](https://www.microsoft.com/en-us/research/publication/pointing-sql-queries-text) from Microsoft Research.
We present the setup for the WikiSQL experiments.


# Training a New Model

## Data Pre-processing

- Download a preprocessed dataset [link](https://1drv.ms/u/s!AryzSDJYB5TxnDWZtpb3ZjL3xBny) to `input/`
- Untar the file `tar -xvjf input.tar.bz2`

#### Reproduce Preprocess Steps

1. Download data from [WikiSQL](https://github.com/salesforce/WikiSQL). 

```
$ cd wikisql_data
$ wget https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2
$ tar -xvjf data.tar.bz2
```
2. Put the [lib directory](https://github.com/salesforce/WikiSQL/tree/master/lib) under `wikisql_data/scripts/`
3. Run annotation using Stanza and preproces the dataset
```
$ cd wikisql_data/scripts/
$ python annotate.py
$ python prepare.py
```

4. Put the train/dev/test data into ``input/data`` for model training/testing. 
5. Use relevance function to prepare relevance files and put them under ``input/nl2prog_input_support_rank`` 
```
python wikisql_data/scripts/relevance.py
```
6. Download pretrained embeddings from [glove](https://nlp.stanford.edu/projects/glove/) and [character n-gram embeddings](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/) and put them under ``input/``



## Training
Meta + Sum loss training
```
$ OUTDIR=output/meta_sum
$ mkdir $OUTDIR
$ python run.py --input-dir ./input \
    --output-dir $OUTDIR \
    --config config/nl2prog.meta_2_0.001.rank.config  \
    --meta_learning_rate 0.001 --gradient_clip_norm 5 \
    --num_layers 3  --num_meta_example 2 \
    --meta_learning --production
```

## Evaluation
- Due to the preprocessing error, we ignore some development (see ``input/data/wikisql_err_dev.dat``) and test (see ``input/data/wikisql_err_test.dat``) set examples, we treat them as incorrect directly.
- Run evaluation as follows (replace ``model_zoo/meta_sum/table_nl_prog-40`` with ``$OUTDIR/table_nl_prog-??`` with the last checkpoint in the folder):

- Development set
```
$ mkdir -p ${OUTDIR}_dev
$ python run.py --input-dir ./input --output-dir ${OUTDIR}_dev \ 
    --config config/nl2prog.meta_2_0.001.rank.devconfig \
    --meta_learning --test-model model_zoo/meta_sum/table_nl_prog-40  --production
```
* Run execution for developement set as follows:
  ```
  $ cp ${OUTDIR}_dev/test_top_1.log dev_top_1.log
  $ python2 execute_dev.py 
    #Q2 (predition) result is wrong: 1254
    #Q1 or Q2 fail to parse: 0
    #Q1 (ground truth) exec to None: 20
    #Q1 (ground truth) failed to execute: 0
    Logical Form Accuracy: 0.631383269546
    Execute Accuracy: 0.68277747403
  ```  
- Test set
```
$ mkdir -p ${OUTDIR}_test
$ python run.py --input-dir ./input --output-dir ${OUTDIR}_test \ 
    --config config/nl2prog.meta_2_0.001.rank.testconfig \
    --meta_learning --test-model model_zoo/meta_sum/table_nl_prog-40  --production
```
* Run execution for test set as follows:
  ```
  $ cp ${OUTDIR}_test/test_top_1.log .
  $ python2 execute.py
    #Q2 (predition) result is wrong: 2556
    #Q1 or Q2 fail to parse: 0
    #Q1 (ground truth) exec to None: 48
    #Q1 (ground truth) failed to execute: 0
    Logical Form Accuracy: 0.628073829775
    Execute Accuracy: 0.680379563733
  ```

- Baseline model on test set
 ```
 $ OUTDIR=output/base_sum
 $ python run.py --input-dir ./input --output-dir ${OUTDIR}_test \
    --config config/nl2prog.testconfig --production  \
    --test-model model_zoo/base_sum/table_nl_prog-79 --production
 ```

* Run execution for the baseline model on test set as follows:
  ```
  $ cp ${OUTDIR}_test/test_top_1.log .
  $ python2 execute.py
    #Q2 (predition) result is wrong: 2636
    #Q1 or Q2 fail to parse: 0
    #Q1 (ground truth) exec to None: 48
    #Q1 (ground truth) failed to execute: 0
    Logical Form Accuracy: 0.614592374009
    Execute Accuracy: 0.668055314471
  ```


# Pre-trained Models
- Download [pretrained model checkpoints](https://1drv.ms/u/s!AryzSDJYB5TxnDR5I4rYjLi4HUYz) to ``model_zoo/`` 
- Run ``tar -xvjf model_zoo.tar.bz2`` to extract pretrain models.

  + Meta + Sum loss: `model_zoo/meta_sum`
  + Base Sum loss: `model_zoo/base_sum`


# Requirements 
- Tensorflow 1.4
- python 3.6
- [Stanza](https://github.com/stanfordnlp/stanza)


# Citation

If you use the code in your paper, then please cite it as:

```
@inproceedings{pshuang2018PT-MAML,
  author    = {Po{-}Sen Huang and
               Chenglong Wang and
               Rishabh Singh and
               Wen-tau Yih and
               Xiaodong He},
  title     = {Natural Language to Structured Query Generation via Meta-Learning},
  booktitle = {NAACL},
  year      = {2018},
}
```

and


```
@techreport{chenglong,
  author = {Wang, Chenglong and Brockschmidt, Marc and Singh, Rishabh},
  title = {Pointing Out {SQL} Queries From Text},
  number = {MSR-TR-2017-45},
  year = {2017},
  month = {November},
  url = {https://www.microsoft.com/en-us/research/publication/pointing-sql-queries-text/},
}
```



# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
