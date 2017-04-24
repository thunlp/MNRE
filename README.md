Neural Relation Extraction with Multi-lingual Attention (MNRE)
==========

Neural relation extraction aims to extract relations from plain text with neural models, which has been the state-of-the-art methods for relation extraction. In this project, we provide our implementations of CNN [Zeng et al., 2014] and PCNN [Zeng et al.,2015] and their extended version with multi-lingual sentence-level attention scheme [Lin et al., 2017] .

 
Data
==========

We provide the  dataset we used for the task relation extraction in  (https://pan.baidu.com/s/1dF26l93). We preprocess the original data to make it satisfy the input format of our codes. 

Pre-Trained English Word Vectors are learned from New York Times Annotated Corpus (LDC Data LDC2008T19), which should be obtained from LDC (https://catalog.ldc.upenn.edu/LDC2008T19).

Pre-Trained Chinese Word Vectors are learned from Chinese Baidu Baike (https://baike.baidu.com/).

To run our code, the dataset should be put in the folder data/ using the following format, containing six files

+ train_en.txt / train_zh.txt: training file, format (wikidata_qid_e1, wikidata_qid_e2, e1_name, e2_name, relation, sentence).

+ valid_en.txt / valid_zh.txt: validation file, same format as train.txt 

+ test_en.txt / test_zh.txt: test file, same format as train.txt.

+ entity2id.txt: all entities and corresponding ids, one per line.

+ relation2id.txt: all relations and corresponding ids, one per line.

+ vec_en.bin, vec_zh.bin: the pre-train word embedding file

Codes
==========

The source codes of various methods are put in the folders src/.

Compile 
==========

Just type "make" in the folder src/.

Train
==========

For training, you need to type the following command in each model folder:

./train

The training model file will be saved in folder out/ .

Test
==========

For testing, you need to type the following command in each model folder:

./test

The testing result which reports the precision/recall curve  will be shown in pr.txt.

Cite
==========

If you use the code, please cite the following paper:

[Lin et al., 2017] Yankai Lin, Zhiyuan Liu, and Maosong Sun. Neural Relation Extraction with Multi-lingual Attention. In Proceedings of ACL.[[pdf]](http://thunlp.org/~lyk/publications/acl2017_mnre.pdf)

Reference
==========
[Zeng et al., 2014] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. Relation classification via convolutional deep neural network. In Proceedings of COLING.

[Zeng et al.,2015] Daojian Zeng,Kang Liu,Yubo Chen,and Jun Zhao. Distant supervision for relation extraction via piecewise convolutional neural networks. In Proceedings of EMNLP.

[Lin et al., 2016] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. In Proceedings of ACL.[[pdf]](http://thunlp.org/~lyk/publications/acl2016_nre.pdf)
