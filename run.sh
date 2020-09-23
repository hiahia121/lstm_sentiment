#!/bin/bash

# 构造词的embeding
#cat data/data | python3 make_word_dict.py > wordscnt 

# 训练模型
python3 main.py --file1 wordscnt > sentence
