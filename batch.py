from typing import List, Dict, Iterator, Any

from fairseq.data import data_utils, FairseqDataset, iterators
from fairseq.data import (
    AppendTokenDataset,
    data_utils,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
)
import numpy as np
import torch


import spacy
import random
from tqdm import tqdm
import multiprocessing as mp

nlp = spacy.load("en_core_web_sm")

def parse_line(line):
        tokens = [
            token for token in line.split(' ')
            if token not in ['', '\n']
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


def parse_data_file(data_file, max_sentences, shuffle=True):
    data_file = data_file
    
    multiprocess = 20

    parsed = []
    with open(data_file, "r") as fd:
        lines = fd.readlines()
    if shuffle:
        random.seed(0xdead)
        random.shuffle(lines)
    
    max_sentences = max_sentences
    
    pool = mp.Pool(20)
    if max_sentences > -1:
        line_it = pool.imap_unordered(parse_line, lines)
        sentence_pb = tqdm(total=max_sentences)
    else:
        line_it = tqdm(pool.imap_unordered(parse_line, lines), total=len(lines))

    for curr_sentences in line_it:
        if curr_sentences == None:
            continue
        if -1 < max_sentences:
            sentence_pb.update(len(curr_sentences))
        parsed.extend(curr_sentences)
        if -1 < max_sentences <= len(parsed):
            parsed = parsed[:max_sentences]
            pool.terminate()
            break
    return parsed


parsed = parse_data_file('/mounts/u-amo-d1/grad/mha361/work/ff-layers2/examples/language_model/wikitext-103/wiki.train.tokens',2000)


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.23.0")

tokenizer_polycoder = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
# model_polycoder = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")


tokenized_sentences = [tokenizer_polycoder.encode(sentence, return_tensors='pt') for sentence in tqdm(parsed)]


def get_batch_iterator(
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices,
                dataset,
                max_positions,
                raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices,
            dataset.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=0
        )
        return epoch_iter
    
    
    
def build_dataset_for_inference(src_tokens, src_lengths, **kwargs):
    """
    Generate batches for inference. We prepend an eos token to src_tokens
    (or bos if `--add-bos-token` is set) and we append a <pad> to target.
    This is convenient both for generation with a prefix and LM scoring.
    """
    dataset = StripTokenDataset(
        TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=None,  # ignored for "eos" break mode
            #pad=self.source_dictionary.pad(),
            pad = torch.Tensor(1),
            #eos=self.source_dictionary.eos(),
            eos = 2,
            break_mode="eos",
        ),
        # remove eos from (end of) target sequence
        # self.source_dictionary.eos(),
        2
    )
    src_dataset = PrependTokenDataset(
        dataset,
        token=(
            # self.source_dictionary.bos()
            0
            if False
            else 2
        ),
    )
    tgt_dataset = AppendTokenDataset(
        dataset,
        token=1
    )
    return NestedDictionaryDataset(
        {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": PadDataset(src_dataset, pad_idx=1, left_pad=False),
                "src_lengths": NumelDataset(src_dataset, reduce=False),
            },
            "target": PadDataset(tgt_dataset, pad_idx=1, left_pad=False),
        },
        sizes=[np.array(src_lengths)],
    )
    
def _build_batches(tokens: List[List[int]], skip_invalid_size_inputs: bool) -> Iterator[Dict[str, Any]]:
    
    lengths = torch.LongTensor([t.numel() for t in tokens])
    batch_iterator = get_batch_iterator(
        dataset=build_dataset_for_inference(tokens, lengths),
        max_tokens=3072,
        max_sentences=2000,
        max_positions=3072,
        ignore_invalid_inputs=skip_invalid_size_inputs,
    ).next_epoch_itr(shuffle=False)
    return batch_iterator
    
x = _build_batches(tokens=tokenized_sentences, skip_invalid_size_inputs=False)
next(x)

for batch in tqdm(_build_batches(tokens=tokenized_sentences, lengths = j, skip_invalid_size_inputs=False)):
    break



for batch in tqdm(_build_batches(tokens=tokenized_sentences, skip_invalid_size_inputs=False)):
    
    break
    