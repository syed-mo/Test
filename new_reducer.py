#!/usr/bin/env python
"""reducer.py"""

# python mapper.py < input.txt | sort | python reducer.py
# hadoop jar /usr/lib/hadoop/hadoop-streaming.jar -files mapper.py,reducer.py -mapper mapper.py -reducer reducer.py -input /user/inputs/inaugs.tar.gz -output /user/j_singh/inaugs

import collections
from collections import OrderedDict
import math
from operator import itemgetter
import sys
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent = 4, width = 80)

def main(argv):
    current_word = None
    current_fnam = None
    current_count = 0
    word = None
    wcss = {}

    # input comes from STDIN
    for line in sys.stdin:
        # remove leading and trailing whitespace
        line_ = line.strip()

        # parse the input we got from mapper.py
        _, word, fnam, count = line_.split('\t', 3)

        # convert count (currently a string) to int
        try:
            count = int(count)
            if fnam in wcss:
                wcss[fnam].update({word:count})
            else:
                wcss[fnam] = collections.Counter()
                wcss[fnam].update({word:count})

        except ValueError:
            # count was not a number, so silently
            # ignore/discard this line
            pass
    word_map = wcss
    
    tf_idf_details = calculate_tf_idf(word_map) # Implement this function
    for document in sorted(tf_idf_details, key=lambda x: x[0]):
        print(document)
        for word in OrderedDict(sorted(tf_idf_details[document].items(), key=lambda t: t[1], reverse=True)):
             print(word + " " + str(round(tf_idf_details[document][word], 5)))
            
            
def calculate_tf(values):
    total_words = sum(list(values.values()))
    tf_map = {}
    for word, occurrence in values.items():
        tf_map[word] = occurrence/total_words

    return tf_map


def normalize(values):
    sum_sq = 0
    for val in values:
        sum_sq += pow(val, 2)
    return math.sqrt(sum_sq)


def calculate_tf_idf(word_map):

    total_documents = len(word_map)
    idf_details = {}
    tf_details = {}
    tf_idf_details = {}

    for document, values in word_map.items():
        tf_idf_details[document] = {}
        tf_details[document] = calculate_tf(values)
        words_in_doc = list(values.keys())
        for word in words_in_doc:
            if word in idf_details:
                idf_details[word] += 1
            else:
                idf_details[word] = 1

    for word, no_of_documents in idf_details.items():
        idf_details[word] = math.log((total_documents+1)/(no_of_documents+1)) + 1

    for document, values in word_map.items():
        for word in values.keys():
            tf_idf_details[document][word] = tf_details[document][word] * idf_details[word]

    for document, values in tf_idf_details.items():
        normalize_value = normalize(list(values.values()))
        for key, value in values.items():
            tf_idf_details[document][key] = value/normalize_value

    return tf_idf_details

if __name__ == "__main__":
    main(sys.argv)
