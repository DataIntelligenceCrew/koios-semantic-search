from distutils.log import error
import os 
from os.path import isfile, join, isdir
import operator
import random 
import pickle 
import tqdm
DATA_LOC = '/localdisk2/cleansets70'




def jaccard(l1, l2):
    intersection = len([x for x in l1 if x in l2])
    union = (len(l1) + len(l2)) - intersection
    if union > 0:
        return float(intersection) / union
    else:
        return 0
    # return float(intersection) / union



def compute_query_index():
    '''
        For each query from the benchmark queries
            - generate the query_vocab
            - iterate over the query_vocab:
                - if token_stream doesn't exists then
                    - compare each word to the entire vocab to get the token stream
    '''
    # word : [(word, e1, sim), ....] such that the sim > alpha
    alpha = 0.8
    word_token_streams = dict()
    f = open('/localdisk2/silkmoth_queries_vocab.obj', 'rb')
    benchmark_vocab = pickle.load(f)
    f.close()
    f1 = open('/localdisk2/silkmoth_exp/opendata_ngrams_vocab.obj', 'rb')
    n_grams_vocab = pickle.load(f1)
    f1.close()
    # print(len(benchmark_vocab))
    progress = tqdm.tqdm(total=len(benchmark_vocab), desc='Words', position=0)
    for word in benchmark_vocab:
        # print(word)
        word_n_grams = generate_n_grams(word, 3)
        token_stream = list()
        if word not in word_token_streams:
            for key, value in n_grams_vocab.items():
                sim = jaccard(word_n_grams, value)
                if sim >= alpha:
                    token_stream.append((key, sim))
            
            word_token_streams[word] = token_stream.sort(key=operator.itemgetter(1))
        progress.update(1)
    
    print(len(word_token_streams))
    with open('/localdisk2/silkmoth_exp/opendata_word_token_streams08.obj', 'wb') as outfile:
        pickle.dump(word_token_streams, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    outfile.close()

    for word, stream in word_token_streams.items():
        file_name = "/localdisk2/silkmoth_exp/token_streams/0.8/{0}.txt".format(word)
        with open(file_name, 'w') as outfile:
            for w1 in stream:
                elem = w1[0]
                sim = w1[1]
                outfile.write("{0} : {1} : {2}\n".format(word, elem, sim))
        outfile.close()

def generate_n_grams_vocab():
    all_files = os.listdir(DATA_LOC)
    vocab = set()
    for file in all_files:
        set_file = open(join(DATA_LOC, file), 'r', encoding="utf8", errors='ignore')
        l = set_file.readlines()
        set_file.close()
        words = [w.strip() for w in l]
        for w1 in words:
            vocab.add(w1)

    print(len(vocab))
    n_grams_vocab = dict()
    for word in vocab:
        n_grams_vocab[word] = generate_n_grams(word, 3)
    
    with open('/localdisk2/silkmoth_exp/opendata_ngrams_vocab.obj', 'wb') as outfile:
        pickle.dump(n_grams_vocab, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    outfile.close()


def baseline_benchmark():
    segment_file_location = '/localdisk3/semantic_benchmark_tests/open_data/new_intervals'
    segment_file_intervals = os.listdir(segment_file_location)
    benchmark_queries = set()
    for interval_file in segment_file_intervals:
        f = open(join(segment_file_location, interval_file), 'r')
        l = f.readlines()
        f.close()
        q = [l1.strip() for l1 in l]
        q1 = random.choices(q, k=10)
        for q2 in q1:
            benchmark_queries.add(q2)
    
    print(benchmark_queries)
    print(len(benchmark_queries))

    with open('/localdisk2/silkmoth_queries.intervals', 'w') as outfile:
        for q3 in benchmark_queries:
            outfile.write(q3 + '\n')
    outfile.close()

    vocab = set()
    for q4 in benchmark_queries:
        try:
            set_file = open(join(DATA_LOC, q4), 'r')
            lines = set_file.readlines()
            set_file.close()
            words = [w.strip() for w in lines]
            for w1 in words:
                vocab.add(w1)
        except FileNotFoundError:
            print(q4)

    with open('/localdisk2/silkmoth_queries_vocab.obj', 'wb') as h:
        pickle.dump(vocab, h, protocol=pickle.HIGHEST_PROTOCOL)
    h.close()

    print(len(vocab))


def generate_n_grams(word, n):
    # print(word)
    n_grams_list = [word[i:j] for i in range(len(word)) for j in range(i + 1, len(word) + 1) if len(word[i:j]) == n]
    return n_grams_list

if __name__ == '__main__':
    # data  =get_vocab()
    # generate_n_grams(data)
    # compute_query_index(queries)
    # get_vocab()
    # open_data_stats()
    # baseline_benchmark()
    # res = generate_n_grams('attatched', 3)
    # print(res)
    # generate_n_grams_vocab()
    # l1 = generate_n_grams('Blaine', 3)
    # l2 = generate_n_grams('Blain', 3)
    # print(jaccard(l1, l2))
    compute_query_index()











