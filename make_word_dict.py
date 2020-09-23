# -*- coding: utf-8 -*-
import jieba
import sys
import time
import numpy as np


begin = time.time()

# 统计词频的词典
word_count = {}

for line in sys.stdin:
    line = line.strip()
    line_decode = line.encode('utf-8').decode("utf-8")
    seg_list = list(jieba.cut(line_decode, cut_all=True))
    for w in seg_list:
        if w in word_count:
            word_count[w] += 1
        else:
            word_count[w] = 1

sorted_words = sorted(word_count.items(), key = lambda x:x[1], reverse=True)

index = 1
for item in sorted_words:
    rand_seed = np.random.seed(0)
    rand_vec = str(list(np.random.rand(64)))
    ret = "%s\t%s\t%s\t%s" % (index, item[0], item[1], rand_vec)
    print(ret)
    index += 1

end = time.time()

#print("use time %s" %(end - begin))
