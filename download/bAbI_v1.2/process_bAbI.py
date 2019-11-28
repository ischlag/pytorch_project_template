# This script merges all the bAbI tasks into one.
# The en-valid and en-valid-10k are treated separately. The stories are
# reshaped into one long sequence. The supporting labels are removed. All
# words are lowercased. The dots and question marks are treated as words
# and the question is part of the story. The request answer tag (<ra>) is
# added after every question mark. The target is supposed to be predicted
# at this point.
#
# The new format uses "\n" to separate samples. A sample is the story
# followed by the answer token. They are separated by "\t". The words of
# the story are separated by " ".

import os
import pickle
import random

DATASETS = ["en-valid", "en-valid-10k"]
PARTITIONS = ["train", "valid", "test"]
DATA_PATH = ".tmp/tasks_1-20_v1-2/{}/qa{}_{}.txt"
OUTPUT_PATH = "data/bAbI_v1.2/"

PAD = "<pad>"
RA = "<ra>"


def parse_file_to_samples(path, task, request_answer_token=RA):
    """ returns [story:[str], (task_id:int, answer:str)] """
    f = open(path, "r")
    samples = []
    story = []
    for line in f:
        tid, text = line.lower().rstrip('\n').split(' ', 1)
        if tid == "1":
            # new story begins
            story = []
        if text.endswith('.'):
            # non-question, append words and period to the story so far
            words = text[:-1].split(' ')
            story.extend(words)
            story.append(".")
        else:
            query, answer, _ = (x.strip() for x in text.split('\t'))
            query_words = query[:-1].split(' ')
            # copy the current story as it may continue for the next sample
            current_story = list(story)
            current_story.extend(query_words)
            current_story.append("?")
            current_story.append(request_answer_token)
            samples.append((list(current_story), answer, task))
    f.close()
    return samples


def read_task_files(dataset, partition):
    """ returns [story:[str], (task_id:int, answer:str)] """
    all_samples = []
    for task in list(range(1, 21)):
        s = parse_file_to_samples(
              path=DATA_PATH.format(dataset, task, partition),
              task=task)
        # print("taks {}: {} samples".format(task, len(s)))
        all_samples.extend(s)
    return all_samples


def load_all_data():
    """ returns {str:{str:[[str], str]}}"""
    data = {}
    for dataset in DATASETS:
        data[dataset] = {}
        for partition in PARTITIONS:
            data[dataset][partition] = read_task_files(dataset, partition)
    return data


def get_vocabulary(data):
    """ returns [] """
    unique_words = [PAD, RA]
    for d in DATASETS:
        for p in PARTITIONS:
            for sample in data[d][p]:
                for word in sample[0]:
                    if word in unique_words:
                        continue
                    else:
                        unique_words.append(word)
                if not sample[1] in unique_words:
                    unique_words.append(sample[1])

    word2idx = {w: i for i, w in enumerate(unique_words)}
    idx2word = {i: w for i, w in enumerate(unique_words)}
    return word2idx, idx2word


def write_files(data, path):
    for d in DATASETS:
        for p in PARTITIONS:
            random.shuffle(data[d][p])
            f = open(os.path.join(OUTPUT_PATH, "{}_{}.txt".format(d, p)), "w+")
            for sample in data[d][p]:
              story = " ".join(sample[0])
              answer = sample[1]
              task = sample[2]
              line = "{}\t{}\t{}\n".format(story, answer, task)
              f.write(line)
            f.close()


def write_vocab(word2idx, idx2word, path):
    f = open(os.path.join(OUTPUT_PATH, "vocab.pkl"), "wb")
    pickle.dump([word2idx, idx2word], f)
    f.close()


os.mkdir(OUTPUT_PATH)
# process joint data
all_data = load_all_data()
write_files(all_data, OUTPUT_PATH)
# vocab
word2idx, idx2word = get_vocabulary(all_data)
write_vocab(word2idx, idx2word, OUTPUT_PATH)
