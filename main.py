import spacy
import random
from tqdm import tqdm
import multiprocessing as mp
import math
import os
import nethook
import heapq

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

files = files1[:5000]
# files.extend(files2[:250])

random.shuffle(files)
for file in tqdm(files):
    p = parse_data_file(file,-1,pool)
    parsed.extend(p)
pool.terminate()


# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from packaging import version
# assert version.parse(transformers.__version__) >= version.parse("4.23.0")

# tokenizer_polycoder = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
# model_polycoder = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")



# model_polycoder = model_polycoder.to("cuda")
# tokenizer_polycoder.pad_token = tokenizer_polycoder.eos_token


# tokenizer_polycoder.pad_token = tokenizer_polycoder.eos_token
# def _build_batches(parsed, batch_size):
    
#     for i in range(0,math.ceil(len(parsed)/batch_size)):
        
#         to = min(len(parsed),batch_size*(i+1))

#         yield tokenizer_polycoder(parsed[batch_size*i:to],padding=True, truncation=True,max_length=200, return_tensors="pt",)
        
        
        
# def _aggregate_layer_values(all_values, batch_index, start_idx, seq_len):
#     max_layer_vals = []
#     max_layer_pos = []
    
#     if seq_len > 0:
#         for layer_vals in all_values:
#             effective_layer_vals = layer_vals[batch_index][start_idx:start_idx + seq_len]
            
#             max_vals = effective_layer_vals.max(axis=0)

#             max_layer_vals.append(max_vals[0].cpu().detach().numpy().tolist())
#             max_layer_pos.append(max_vals[1].cpu().detach().numpy().tolist())

#             #For the sparsity experiment: we use a randomly chosen position for all layers
#     return max_layer_vals, max_layer_pos

# import torch

# def score(parsed1):
#     batch_size = 3
#     l = []
#     results = []
#     for i in range(0,32):
#         l.append('gpt_neox.layers.'+str(i)+'.mlp.dense_h_to_4h')
#     with nethook.TraceDict(model_polycoder, l) as ret:
#         for batch in _build_batches(parsed1,batch_size):#,total=math.ceil(len(parsed)/batch_size)):
#             torch.cuda.empty_cache()
#             # torch.cuda.synchronize(device="cuda")
#             batch = batch.to("cuda")
#             net_input = batch['input_ids']
#             out = model_polycoder(net_input)
            
#             fc1_vals = [
#                         ret[layer_fc1_vals].output#.transpose(0, 1)//works without transpose somehow
#                         for layer_fc1_vals in ret
#                     ]
#             # print(len(fc1_vals[0]))
            
            
            
#             bsz = len(fc1_vals[0])
#             hypos = []
#             start_idxs = 0
#             for i in range(0, bsz):
#                 max_layer_fc1_vals_i, max_pos_layer_fc1_vals_i  = _aggregate_layer_values(all_values = fc1_vals, batch_index = i, start_idx = start_idxs, seq_len = torch.count_nonzero(batch['input_ids'][0]))
#                 hypos.append({
#                             'max_fc1_vals': max_layer_fc1_vals_i,
#                             'max_pos_fc1_vals': max_pos_layer_fc1_vals_i,
#                         })
#             results.extend(hypos)#didnt sort on input ids
    
#     return results


# import json
# import numpy as np
# import pandas as pd

# def format_ffn_values(hypos, sentences, pos_neg, extract_mode, output_values_shape=False):
#     for i, hypo in enumerate(hypos):
#         if i == 0 and output_values_shape:
#             yield (len(hypo['max_fc1_vals']), len(hypo['max_fc1_vals'][0]))

#         if extract_mode == "layer-raw":
#             yield {
#                 'pos_neg': pos_neg,
#                 'text': str(sentences[i]),
#                 'max_fc1_vals': hypo['max_fc1_vals'],
#                 'max_pos_fc1_vals': hypo['max_pos_fc1_vals'],
#             }
#         elif extract_mode == "dim":
#             yield json.dumps({
#                 'pos_neg': pos_neg,
#                 'text': str(sentences[i]),
#                 'output_dist_vals': hypo['output_dist_vals'],
#                 'output_dist_conf': hypo['output_dist_conf'],
#                 'residual_ffn_output_rank': hypo['residual_ffn_output_rank'],
#                 'residual_ffn_output_prob': hypo['residual_ffn_output_prob'],
#                 'residual_ffn_argmax': hypo['residual_ffn_argmax'],
#                 'residual_ffn_argmax_prob': hypo['residual_ffn_argmax_prob'],
#                 'ffn_residual_output_rank': hypo['ffn_residual_output_rank'],
#                 'ffn_residual_output_prob': hypo['ffn_residual_output_prob'],
#                 'dim_pattern_preds': hypo['dim_pattern_preds'],
#                 'dim_pattern_output_rank': hypo['dim_pattern_output_rank'],
#                 'dim_pattern_ffn_output_rank': hypo['dim_pattern_ffn_output_rank'],
#                 'dim_pattern_ffn_output_prob': hypo['dim_pattern_ffn_output_prob'],
#                 'coeffs_vals': hypo['coeffs_vals'],
#                 'coeffs_l0': hypo['coeffs_l0'],
#                 'coeffs_residual_rank': hypo['coeffs_residual_rank'],
#                 'random_pos': hypo['random_pos'],
#             }) + '\n'
#         else:
#             assert extract_mode == "layer"
#             yield json.dumps({
#                 'pos_neg': pos_neg,
#                 'text': str(sentences[i]),
#                 'output_dist_vals': hypo['output_dist_vals'],
#                 'layer_output_argmax': hypo['layer_output_argmax'],
#                 'layer_output_argmax_prob': hypo['layer_output_argmax_prob'],
#                 'residual_argmax': hypo['residual_argmax'],
#                 'residual_argmax_prob': hypo['residual_argmax_prob'],
#                 'residual_argmax_change': hypo['residual_argmax_change'],
#                 'residual_output_rank': hypo['residual_output_rank'],
#                 'residual_output_prob': hypo['residual_output_prob'],
#                 'ffn_matching_dims_count': hypo['ffn_matching_dims_count'],
#                 'ffn_output_rank': hypo['ffn_output_rank'],
#                 'ffn_output_prob': hypo['ffn_output_prob'],
#                 'ffn_residual_output_rank': hypo['ffn_residual_output_rank'],
#                 'ffn_residual_output_prob': hypo['ffn_residual_output_prob'],
#                 'ffn_argmax': hypo['ffn_argmax'],
#                 'ffn_argmax_prob': hypo['ffn_argmax_prob'],
#                 'coeffs_l0': hypo['coeffs_l0'],
#                 'coeffs_residual_rank': hypo['coeffs_residual_rank'],
#                 'random_pos': hypo['random_pos'],
#             }) + '\n'


# def get_trigger_examples(all_ffn_values, dims_for_analysis, num_sentences, values_shape, output_file,
#                          top_k=5, apply_relu=True, num_layers=32):
#     values_key = 'max_fc1_vals'
#     position_key = 'max_pos_fc1_vals'
#     hidden_size = values_shape[1]  # shape: (num_layers, hidden_size)
#     # if args.dims_for_analysis is not None and len(args.dims_for_analysis) > 0:
#     #     assert len([
#     #         dim for dim in dims_for_analysis
#     #         if dim < 0 or dim >= hidden_size
#     #     ]) == 0
#     # else:
#     dims_for_analysis = list(range(hidden_size))
#     layers = list(range(num_layers))
#     num_dims = len(dims_for_analysis)

#     layer_vals = np.zeros((num_layers, top_k, num_dims))
#     min_layer_vals_i = np.zeros((num_layers, num_dims), dtype=int)
#     token_indices = np.zeros((num_layers, top_k, num_dims))  # token indices
#     sentence_indices = np.zeros((num_layers, top_k, num_dims), dtype=int)
#     all_ffn_vals = []
#     j = 0
#     for i, ffn_vals in enumerate(tqdm(all_ffn_values, total=num_sentences)):
#         j+=1
#         loaded_vals = ffn_vals
#         val = loaded_vals.pop(values_key)
#         val_pos = loaded_vals.pop(position_key)
#         for layer_index in layers:
#             for d_i, d in enumerate(dims_for_analysis):
#                 loc_ind = min_layer_vals_i[layer_index, d_i]
#                 if val[layer_index][d] > layer_vals[layer_index, loc_ind, d_i]:
#                     layer_vals[layer_index, loc_ind, d_i] = val[layer_index][d]
#                     token_indices[layer_index, loc_ind, d_i] = val_pos[layer_index][d]
#                     sentence_indices[layer_index, loc_ind, d_i] = i
#                     min_layer_vals_i[layer_index, d_i] = np.argmin(
#                         layer_vals[layer_index, :, d_i])
#         all_ffn_vals.append(loaded_vals)
#     top_vals_per_dim = []
#     for layer_index in layers:
#         if apply_relu:
#             layer_vals[layer_index] = np.maximum(layer_vals[layer_index], 0)
#         else:
#             layer_vals[layer_index] = layer_vals[layer_index]
#         top_vals_per_dim.append(np.argsort(layer_vals[layer_index], axis=0)[-top_k::][::-1, :].T)

#     # write output
#     with open(output_file, "w") as fd:
#         for dim_i, dim in enumerate(dims_for_analysis):
#             dim_outputs = []
#             for layer_index in layers:
#                 layer_output = []
#                 for rank, i in enumerate(top_vals_per_dim[layer_index][dim_i]):
#                     layer_output.append({
#                         "rank": rank, 'layer_index': layer_index,
#                         'token_indice': token_indices[layer_index][i][dim_i],
#                         "fc1_value": layer_vals[layer_index][i][dim_i],
#                         "text": all_ffn_vals[sentence_indices[layer_index][i][dim_i]]['text']
#                     })
#                 dim_outputs.append(layer_output)
#             fd.write(json.dumps({"dim": dim, "top_values": dim_outputs}) + '\n')


# def normalize_sublists(matrix):
#     normalized_matrix = []
#     for sublist in matrix:
#         normalized_sublist = sublist / np.max(sublist)
#         normalized_matrix.append(normalized_sublist)
#     return normalized_matrix

# def get_top_k(matrix, k):
#     matrix = normalize_sublists(matrix)
#     flattened_list = [item for sublist in matrix for item in sublist]  # Flatten the 2D list
#     top_k_indices_flattened = heapq.nlargest(k, range(len(flattened_list)), key=flattened_list.__getitem__)
#     row_length = len(matrix[0])
#     top_k_elements = [(index // row_length, index % row_length, flattened_list[index]) for index in top_k_indices_flattened]
#     return top_k_elements

# #my func()
# def get_trigger_keys(all_ffn_values,num_sentences):
#     for i, ffn_vals in enumerate(tqdm(all_ffn_values, total=num_sentences)):
#         loaded_vals = ffn_vals
#         print(loaded_vals['text'])
#         print(get_top_k(loaded_vals['max_fc1_vals'],50))



# def get_hypos():
#         for batch_i in tqdm(list(range(0, len(parsed), 1000))):
#             for hypo_parsed in score(
#                     parsed[batch_i:min(len(parsed), batch_i + 1000)]
#             ):
#                 yield hypo_parsed

# all_ffn_values = format_ffn_values(hypos=get_hypos(),
#                                    sentences=parsed,
#                                    pos_neg=1,
#                                    extract_mode="layer-raw",
#                                    output_values_shape=True)
# values_shape = next(all_ffn_values)

# # get_trigger_keys(all_ffn_values=all_ffn_values, num_sentences=len(parsed))

# get_trigger_examples(all_ffn_values,
#                     dims_for_analysis=22,
#                     num_sentences=len(parsed),
#                     values_shape=values_shape,
#                     output_file="top50_java_polycoder_5_line_5000files.jsonl",
#                     top_k=50)

