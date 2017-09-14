import collections
import os
import numpy as np


checkpoints_dir = os.path.abspath('./checkpoints/')
poetry_file ='data/poetry.txt'
model_prefix = 'poem'
start_token = '['
end_token = ']'

batch_size = 64

def preprocess_poems():
    # poem list
    poetrys = []
    with open(poetry_file, "r", encoding='utf-8',) as f:
        for line in f:
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ','')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poetrys.append(content)
            except Exception as e: 
                pass
     
    # sort poems by number of words contained
    poetrys = sorted(poetrys,key=lambda line: len(line))
    print('Total number of poems: ', len(poetrys))
     
    # count the occurence of each word
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
     
    # take the most used words
    #words = words[:len(words)] + (' ',)
    words = words[:2000] + (' ',)

    # map each word to an integer
    word_int_map = dict(zip(words, range(len(words))))
    # vectorize
    to_num = lambda word: word_int_map.get(word, len(words))
    poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
    return poetrys_vector,  word_int_map, words

def generate_batch(poetrys_vector, word_int_map):     

    n_chunk = len(poetrys_vector) // batch_size
    
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
     
        batches = poetrys_vector[start_index:end_index]
        # find the longest poem in this batch
        length = max(map(len,batches))
        # fill the empty space with " "
        xdata = np.full((batch_size,length), word_int_map[' '], np.int32)
        for row in range(batch_size):
            xdata[row,:len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        # y is equal to x one step ahead
        ydata[:,:-1] = xdata[:,1:]
        """
        xdata             ydata
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(xdata)
        y_batches.append(ydata)
        
    return x_batches, y_batches 
