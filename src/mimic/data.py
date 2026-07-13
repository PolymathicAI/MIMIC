"""Batch preparation for MIMIC.

Builds model-ready modality dicts from (possibly partial) per-sample inputs:
allocate empty entries for every modality, fill in the ones the user provided,
then pad/collate within summation groups. These back ``MIMIC.input()`` and are
the natural seed for a future training ``Dataset``/dataloader.
"""
import torch
from torch.utils.data._utils.collate import default_collate


def make_empty_mod_dict(modality_info):
    """Allocate an empty (fully-masked) entry for every modality in `modality_info`."""
    empty_mod_dicts = {}

    for mod_name, mod_info in modality_info.items():
        empty_mod = {}

        # Tensor (all MIMIC modalities are discrete token sequences)
        max_tokens = mod_info['max_tokens']
        empty_mod['tensor'] = torch.zeros((max_tokens), dtype=torch.int32)

        # Input and target masks (True == masked / absent)
        empty_mod['input_mask'] = torch.ones((max_tokens), dtype=torch.bool)
        empty_mod['target_mask'] = torch.ones((max_tokens), dtype=torch.bool)

        # Decoder attention mask
        empty_mod['decoder_attention_mask'] = torch.zeros((max_tokens), dtype=torch.int32)

        empty_mod['len'] = 0

        empty_mod_dicts[mod_name] = empty_mod

    return empty_mod_dicts


def fill_incomplete_mod_dict(data, modality_info):
    """Fill a full modality dict from the provided (subset of) modalities in `data`."""
    mod_dict = make_empty_mod_dict(modality_info)
    for mod in mod_dict:
        if mod in data:
            mod_dict[mod].update(data[mod])
            mod_dict[mod]['len'] = data[mod]['tensor'].shape[0]
    return mod_dict


def pad_and_collate(batch, modality_info, sum_modality_groups):
    """
    Pads each key of individual samples to the maximum length of that key in the batch.
    The padding token is the pad_token of the respective modality.

    batch: List of individual sample dicts with modalities as keys. Each modality has
        'tensor', 'input_mask', 'target_mask', and 'decoder_attention_mask'.
    modality_info: Dictionary containing information about the modalities.
    """

    # Get the maximum length of each key in the batch
    max_len = {k: max([sample[k]['len'] for sample in batch]) for k in batch[0].keys() if k != "class_idx"}

    # get the group max and min lengths
    if sum_modality_groups:
        all_groups = set([modality_info[mod]['summation_group'] for mod in modality_info])
        group_max_len = {g: max([max_len[mod] for mod, data in modality_info.items() if data['summation_group'] == g]) for g in all_groups}

        # overwrite the max and min lengths with the group max and min lengths
        max_len = {k: group_max_len[modality_info[k]['summation_group']] for k in max_len.keys()}

    # iterate over the different modalities
    sample_lens = {}
    for mod in batch[0].keys():
        if mod == "class_idx":
            continue
        # Get the pad token for the modality
        try:
            pad_token = modality_info[mod]['pad_token_id']
        except KeyError:
            raise KeyError(f"Unequal lengths detected, but no pad token specified for modality {mod}. Please specify a pad token in the modality_info dictionary.")
        # Pad each tensor to the maximum length
        sample_lens[mod] = []
        for sample in batch:
            sample_lens[mod].append(sample[mod]['len'])
            if len(sample[mod]['tensor']) >= max_len[mod]:
                sample[mod]['tensor'] = sample[mod]['tensor'][:max_len[mod]]
                sample[mod]['input_mask'] = sample[mod]['input_mask'][:max_len[mod]]
                sample[mod]['target_mask'] = sample[mod]['target_mask'][:max_len[mod]]
                sample[mod]['decoder_attention_mask'] = sample[mod]['decoder_attention_mask'][:max_len[mod]]
            else:
                sample[mod]['tensor'] = torch.cat([sample[mod]['tensor'], torch.full((max_len[mod] - sample[mod]['tensor'].shape[0],), pad_token, dtype=sample[mod]['tensor'].dtype)])
                sample[mod]['input_mask'] = torch.cat([sample[mod]['input_mask'], torch.ones((max_len[mod] - sample[mod]['input_mask'].shape[0]), dtype=sample[mod]['input_mask'].dtype)])
                sample[mod]['target_mask'] = torch.cat([sample[mod]['target_mask'], torch.ones((max_len[mod] - sample[mod]['target_mask'].shape[0]), dtype=sample[mod]['target_mask'].dtype)])
                sample[mod]['decoder_attention_mask'] = torch.cat([sample[mod]['decoder_attention_mask'], torch.zeros((max_len[mod] - sample[mod]['decoder_attention_mask'].shape[0]), dtype=sample[mod]['decoder_attention_mask'].dtype)])

    collated_batch = default_collate(batch)

    return collated_batch
