import torch
import torch.nn as nn
import numpy as np

def adaptive_decoding(model, tokenizer, prefix, max_len, epsilon):

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else eos_token_id

    device = model.device
    tokens = tokenizer.tokenize(prefix)
    prefix_id_list = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(prefix_id_list).to(device).repeat(1, 1)

    # prepare configuration to adaptive decoding
    vocabulary = len(tokenizer)
    up_bound = -np.log2(1 / vocabulary)

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device) if eos_token_id is not None else None
    for _ in range(max_len):
        greedy_output = model.generate(input_ids, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=eos_token_id)
        logit = greedy_output['scores'][0]
        
        # adaptive determination of candidace size
        x = nn.functional.softmax(logit, dim=-1)
        sorted_values, sorted_indices = torch.sort(x, descending=True)
        tensor = torch.arange(1, vocabulary + 1).repeat(input_ids.shape[0], 1).to('cuda')
        cumulative_sum = torch.cumsum(sorted_values, dim=1).to('cuda')
        A = sorted_values * torch.log(sorted_values * (vocabulary - tensor) / (1.0 - cumulative_sum))
        C = (1 - cumulative_sum) / (vocabulary - tensor)
        D = (1 - cumulative_sum + sorted_values) / (vocabulary + 1 - tensor)
        B = torch.log(C / D)
        
        delta_conf = (A + (1 - cumulative_sum + sorted_values) * B) / up_bound
        delta_conf[torch.isnan(delta_conf)] = 0 
        
        sorted_indices_to_remove = delta_conf < epsilon
        sorted_indices_to_remove[..., -1 :] = 0
        # do the truncation
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        
        logit = logit.masked_fill(indices_to_remove, -float("Inf"))
        x = nn.functional.softmax(logit, dim=-1)
        
        next_id = torch.multinomial(x, num_samples=1).squeeze(1)
        next_id = next_id * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        unfinished_sequences = unfinished_sequences.mul(
                        next_id.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )
    
        input_ids = torch.cat([input_ids, next_id[:, None]], dim=-1)
        if unfinished_sequences.max() == 0:
            break

    generated_results = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return generated_results