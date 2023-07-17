import spacy
import random
from tqdm import tqdm
import multiprocessing as mp
import os
import re

nlp = spacy.load("en_core_web_sm")

def parse_line(line):
        tokens = [
            token for token in line.split(' ')
            if token not in ['', '']
        ]
        if len(tokens) == 0:
            return None
        spaces = [True for _ in range(len(tokens)-1)] + [False]
        assert len(tokens) == len(spaces), f"{len(tokens)} != {len(spaces)}"

        doc = spacy.tokens.doc.Doc(
            nlp.vocab, words=tokens, spaces=spaces)
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        return [str(sent) for sent in doc.sents]


def parse_data_file(data_file, max_sentences, pool,shuffle=False):
    data_file = data_file
    
    multiprocess = 20

    parsed = []
    with open(data_file, "r") as fd:
        lines = fd.readlines()
    if shuffle:
        random.seed(0xdead)
        random.shuffle(lines)
    
    
    
    window = 5
    lines2 = []
    for i in range(0,len(lines)-window):
        line = lines[i:i+window]
        lines2.append("".join(line))
    lines = lines2
    
    
    
    
    max_sentences = max_sentences
    # max_sentences = max_sentences
    
    if max_sentences > -1:
        line_it = pool.imap_unordered(parse_line, lines)
        sentence_pb = tqdm(total=max_sentences)
    else:
        line_it = pool.imap_unordered(parse_line, lines)

    for curr_sentences in line_it:
        if curr_sentences == None:
            continue
        if -1 < max_sentences:
            sentence_pb.update(len(curr_sentences))
        parsed.extend(curr_sentences)
        if -1 < max_sentences <= len(parsed):
            parsed = parsed[:max_sentences]
            # pool.terminate()
            break
    return parsed


parsed = []
def get_files(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


pool = mp.Pool(20)
files1 = get_files('/u/amo-d1/grad/mha361/work/Code-LMs/Data/Code/Python/')
files2 = get_files('/u/amo-d1/grad/mha361/work/Code-LMs/Data/Code/Java/')

files = files1[:500]
# files.extend(files2[:250])

random.shuffle(files)
for file in tqdm(files):
    p = parse_data_file(file,-1,pool)
    parsed.extend(p)
pool.terminate()
for p in parsed:
    pattern = r'(for|while)'

    matches = re.findall(pattern, p, flags=re.IGNORECASE)
    print(p)
    print(matches)
