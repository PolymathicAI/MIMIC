from abc import ABC, abstractmethod
from tqdm import tqdm
from contextlib import nullcontext
from typing import Generic, Dict, TypedDict, TypeVar, Set, List, Callable, Literal
import torch
from dataclasses import dataclass, field
from loguru import logger
import numpy as np
from collections import defaultdict
import copy


def make_decoder_attention_mask(target_mask: torch.Tensor, mod_type: str, is_target_autoregr: bool) -> torch.Tensor:
    """Build the per-sample decoder attention mask for one modality's target tokens.

    (Inlined from the training-time masking module, its only consumer.) The first
    target token of a chain modality carries the target length, which
    `MIMIC.adapt_decoder_attention_mask` expands into the right block structure.
    """
    if is_target_autoregr:
        # Strict autoregressive behavior regardless of mod_type.
        target_length = len(target_mask)
        decoder_attention_mask = torch.ones(target_length, dtype=torch.int, device=target_mask.device)
    else:
        if mod_type in ["chain_token"]:
            target_length = len(target_mask)
            decoder_attention_mask = torch.zeros(target_length, dtype=torch.int, device=target_mask.device)
            first_mask_token = torch.argmin(target_mask + torch.arange(target_length, device=target_mask.device) * 1e-6)
            decoder_attention_mask[first_mask_token] = (~target_mask).sum()  # actual number of target tokens
        elif mod_type in ["text_token_all_targets"]:
            length = len(target_mask)
            decoder_attention_mask = torch.ones(length, dtype=torch.int, device=target_mask.device)
        else:
            raise ValueError(f"Invalid modality type for decoder attention mask: {mod_type}")

    return decoder_attention_mask

class GenerationOutputDict(TypedDict):
    tokens: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor
    sampling_probs: torch.Tensor
    idx: List[int]

@dataclass
class ModStatus:
    idx_req: Set[int]
    idx_done: Set[int] = field(default_factory=set)
    idx_to_gen: List[int] = field(default_factory=list)
    gen_logits: torch.Tensor = field(default_factory=torch.Tensor)
    gen_probs: torch.Tensor = field(default_factory=torch.Tensor)
    gen_sampling_probs: torch.Tensor = field(default_factory=torch.Tensor)
    idx_to_fill: List[int] = field(default_factory=list)
    tokens_to_fill: torch.Tensor = field(default_factory=torch.Tensor)
    all_logits: Dict[int, torch.Tensor] = field(default_factory=dict)
    all_probs: Dict[int, torch.Tensor] = field(default_factory=dict)
    all_sampling_probs: Dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class BaseGenerationState:
    mod_dict: Dict
    target_dict: Dict[str, ModStatus]
    remaining_tokens: int

StateType = TypeVar("StateType", bound=BaseGenerationState)
class BaseGenerationStrategy(ABC, Generic[StateType]):
    """Base class for generation strategies.
    Any new generation strategy should inherit from this class and implement
    the select_generation_targets and process_logits methods.

    The selection of generation targets function should set the idx_to_gen
    attribute of the ModStatus objects in the target_dict of the state.

    The process_logits function should set the gen_probs, gen_sampling_probs,
    tokens_to_fill, and idx_to_fill attributes of the ModStatus objects in the
    target_dict of the state.
    """

    @torch.no_grad()
    def generate(self,
                 model_forward:Callable,
                 mod_dict:dict,
                 modality_info:Dict[str, dict],
                 target_set:Set[str],
                 max_model_target_tokens:int,
                 temperature:float,
                 is_target_autoregr:bool,
                 verbose:bool=True)->Dict[str, GenerationOutputDict]:
        """
        This generation script generates tokens for the specified target modalities.
        It is assumed that the mod_dict contains the target modalities with placeholder
        tensors of the correct shape.
        Args:
            model_forward (Callable): Function to perform the model forward pass.
            mod_dict (dict): Dictionary containing the modality tensors and masks.
            modality_info (Dict[str, dict]): Dictionary containing modality information.
            target_set (Set[str]): Set of modality names to generate tokens for.
            max_model_target_tokens (int): Maximum number of tokens the model can generate
                                           in a single forward pass for a modality.
            temperature (float): Temperature for sampling. Higher values result in more
                                 random samples, while lower values make the sampling
                                 more greedy.
            verbose (bool, optional): Whether to display a progress bar. Defaults to True.
        Returns:
            Dict[str, GenerationOutputDict]: Dictionary containing the generated tokens,
                                             logits, probabilities, sampling probabilities,
                                             and indices for each target modality.
        """

        if temperature == 0:
            logger.warning("Temperature 0 is ill-defined. Setting it to 1E-8 for sampling.")
            temperature = 1E-8

        state = self._initialize_state(mod_dict, target_set)
        total_target_tokens = state.remaining_tokens
        if verbose:
            pbar = tqdm(total=total_target_tokens, desc=f"Generating tokens", unit="token")

        with pbar if verbose else nullcontext():

            while state.remaining_tokens != 0:

                state = self.select_generation_targets(state, max_model_target_tokens)

                state = self._set_generation_masks(state, modality_info, is_target_autoregr)

                state = self._forward_pass(state, model_forward)

                state = self.process_logits(state, temperature)

                state = self._fill_mod_dict(state)

                if verbose:
                    pbar.n = total_target_tokens - state.remaining_tokens
                    pbar.refresh()

        output = self._make_output(state)

        return output

    def _set_generation_masks(self, state: StateType, modality_info: Dict[str, dict], is_target_autoregr: bool) -> StateType:

        for mod, mod_status in state.target_dict.items():
            # make sure the idx_to_gen is sorted
            mod_status.idx_to_gen = sorted(mod_status.idx_to_gen)
            # reset the target_mask to be all True
            state.mod_dict[mod]['target_mask'][0] = True
            state.mod_dict[mod]['target_mask'][0][mod_status.idx_to_gen] = False
            state.mod_dict[mod]['decoder_attention_mask'] = make_decoder_attention_mask(
                state.mod_dict[mod]['target_mask'][0], mod_type = modality_info[mod]['type'],
                is_target_autoregr = is_target_autoregr
            ).unsqueeze(0)

        return state

    def _forward_pass(self, state: StateType, model_forward) -> StateType:
        output = model_forward(state.mod_dict,
                       return_encoder_output=False,
                       return_logits=True,
                       return_loss=False)
        for mod, mod_status in state.target_dict.items():
            if mod_status.idx_to_gen:
                mod_status.gen_logits = output['mod_logits'][mod].data
        return state

    def _calc_remaining_tokens(self, target_dict: Dict[str, ModStatus]) -> int:
        remaining_tokens = sum(len(mod.idx_req - mod.idx_done) for mod in target_dict.values())
        return remaining_tokens

    # Overwrite if using a different state class
    def _initialize_state(self, mod_dict: dict, target_set: Set[str]) -> StateType:
        """
        Initialize the generation state.
        This function creates the initial state for generation, including
        the mod_dict, target_dict, and remaining_tokens.
        """

        target_dict = {}
        for mod in target_set:
            idx_req = set(torch.where(mod_dict[mod]['input_mask'][0])[0].tolist())
            target_dict[mod] = ModStatus(idx_req=idx_req)

        state = BaseGenerationState(
            mod_dict=mod_dict,
            target_dict={mod: mod_status for mod, mod_status in target_dict.items()},
            remaining_tokens=self._calc_remaining_tokens(target_dict)
        )

        return state

    def _fill_mod_dict(self, state: StateType) -> StateType:

        for mod, mod_status in state.target_dict.items():
            if mod_status.idx_to_fill:
                tensor = state.mod_dict[mod]['tensor']
                tensor[0, mod_status.idx_to_fill] = mod_status.tokens_to_fill.type_as(tensor)
                state.mod_dict[mod]['input_mask'][0, mod_status.idx_to_fill] = False
                mod_status.idx_done.update(mod_status.idx_to_fill)
                mod_status.all_logits.update({idx: logit for idx, logit in zip(mod_status.idx_to_fill, mod_status.gen_logits)})
                mod_status.all_probs.update({idx: prob for idx, prob in zip(mod_status.idx_to_fill, mod_status.gen_probs)})
                mod_status.all_sampling_probs.update({idx: s_prob for idx, s_prob in zip(mod_status.idx_to_fill, mod_status.gen_sampling_probs)})
                mod_status.idx_to_gen = []
                mod_status.gen_logits = torch.tensor([])
                mod_status.idx_to_fill = []
                mod_status.tokens_to_fill = torch.tensor([])

        state.remaining_tokens = self._calc_remaining_tokens(state.target_dict)

        return state


    def _make_output(self, state: StateType) -> Dict[str, GenerationOutputDict]:
        output = {}
        for mod, mod_status in state.target_dict.items():
            req_idx = sorted(list(mod_status.idx_req))
            all_logits = torch.stack([mod_status.all_logits[idx] for idx in req_idx], dim=0)
            all_probs = torch.stack([mod_status.all_probs[idx] for idx in req_idx], dim=0)
            all_sampling_probs = torch.stack([mod_status.all_sampling_probs[idx] for idx in req_idx], dim=0)
            output[mod] = GenerationOutputDict(
                tokens=state.mod_dict[mod]['tensor'][0][req_idx],
                logits=all_logits,
                probs=all_probs,
                sampling_probs=all_sampling_probs,
                idx=req_idx
            )
        return output

    @abstractmethod
    def select_generation_targets(self, state: StateType, max_model_target_tokens: int) -> StateType:
        """
        Select the target tokens for generation from the state.
        This function should set the idx_to_gen attribute of the ModStatus
        objects in the target_dict of the state.
        """
        return state

    @abstractmethod
    def process_logits(self, state: StateType, temperature: float) -> StateType:
        """
        Process the logits from the model forward pass and sample tokens.
        This function should set the gen_probs, gen_sampling_probs,
        tokens_to_fill, and idx_to_fill attributes of the ModStatus
        objects in the target_dict of the state.
        """
        return state



class OneShotGenerationStrategy(BaseGenerationStrategy[BaseGenerationState]):
    """
    One-shot generation strategy.
    This strategy generates all required tokens for the target modalities
    in a single forward pass.
    """

    def select_generation_targets(self, state: BaseGenerationState, max_model_target_tokens: int) -> BaseGenerationState:
        for mod in state.target_dict.keys():
            assert len(state.target_dict[mod].idx_req) <=  max_model_target_tokens, \
                f"Number of tokens to generate for modality {mod} exceeds max that can be generated in one pass."
            state.target_dict[mod].idx_to_gen = list(state.target_dict[mod].idx_req)

        return state

    def process_logits(self, state: BaseGenerationState, temperature: float) -> BaseGenerationState:

        for mod_status in state.target_dict.values():
            mod_status.gen_probs = torch.softmax(mod_status.gen_logits, dim=-1)
            mod_status.gen_sampling_probs = torch.softmax(mod_status.gen_logits / temperature, dim=-1)
            mod_status.tokens_to_fill = torch.multinomial(mod_status.gen_sampling_probs, num_samples=1)[:, 0]
            mod_status.idx_to_fill = mod_status.idx_to_gen

        return state
    
class BatchedOneShotGenerationStrategy(OneShotGenerationStrategy):
    """
    Batched version of One-shot generation strategy.
    Accepts a list of mod_dicts (one per sample) and processes them in a batch.
    """

    


class RandomSubsetGenerationStrategy(OneShotGenerationStrategy):
    """
    Random subset generation strategy.
    This strategy randomly selects a subset of tokens to generate from the
    available tokens in the state.
    """

    def __init__(self, num_tokens_to_generate: int):
        """
        Args:
            num_tokens_to_generate (int): Number of tokens to generate in one pass.
        """
        self.num_tokens_to_generate = num_tokens_to_generate

    def _initialize_state(self, mod_dict: dict, target_set: Set[str]) -> BaseGenerationState:
        state = super()._initialize_state(mod_dict, target_set)
        for mod_status in state.target_dict.values():
            idx_req_tens = torch.tensor(list(mod_status.idx_req))
            mod_status.idx_req = set(idx_req_tens[torch.randperm(idx_req_tens.numel())[:self.num_tokens_to_generate]].tolist())

        state.remaining_tokens = self._calc_remaining_tokens(state.target_dict)
        return state


class ARGenerationStrategy(BaseGenerationStrategy[BaseGenerationState]):
    """
    Auto-regressive generation strategy.
    This strategy generates a fixed number of tokens per step, either
    for all target modalities at once, or one modality at a time.
    The tokens to generate can be selected randomly or sequentially.
    Optionally, a subset of the tokens to generate can be selected based
    on the model's confidence (top-n probabilities).
    """

    def __init__(self, num_tokens_per_step:int,
                 mode:str = 'all_at_once',
                 sampling:str = 'random',
                 pick_top_n_probs:int = None,
                 top_n_mode:str = 'min',
                 prob_threshold: float = None):
        """
        Args:
            num_tokens_per_step (int): Number of tokens to generate per step.
            mode (str, optional): Generation mode. One of 'all_at_once', 'index_first', 'mod_first'.
                                  'all_at_once': Generate tokens for all target modalities at once.
                                  'index_first': Generate tokens for one index across all modalities first.
                                  'mod_first': Generate all tokens for one modality first.
                                  Defaults to 'all_at_once'.
            sampling (str, optional): Sampling strategy for selecting tokens to generate.
                                      One of 'random', 'sequential'. Defaults to 'random'.
            pick_top_n_probs (int, optional): If specified, only the top-n tokens with
                                              highest (or lowest) probabilities will be considered
                                              for generation. Must be <= num_tokens_per_step.
                                              Defaults to None.
            top_n_mode (str, optional): If pick_top_n_probs is specified, this determines
                                        whether to pick the 'min' (lowest), 'max' (highest),
                                        'random' probability tokens, the 'first' tokens by
                                        smallest index, or the 'last' tokens by largest index.
                                        Defaults to 'min'.
            prob_threshold (float, optional): If specified, and ALL generated tokens have
                                              confidence > this threshold, pick_top_n_probs
                                              is ignored and all tokens are filled.
                                              NOTE: This threshold is for probabilities calculated with 
                                              temperature set to 1.
                                              Defaults to None.
        """

        self.num_tokens_per_step = num_tokens_per_step
        self.prob_threshold = prob_threshold

        assert mode in ['all_at_once', 'index_first', 'mod_first'], f"Mode {mode} not recognized."
        logger.warning("ARGenerationStrategy only implemented for one target modality.")
        self.mode = mode

        assert sampling in ['random', 'sequential'], f"Sampling {sampling} not recognized."
        self.sampling = sampling

        assert pick_top_n_probs is None or pick_top_n_probs <= num_tokens_per_step, \
            f"pick_top_n_probs {pick_top_n_probs} must be smaller than or equal to num_tokens_per_step {num_tokens_per_step}."
        self.pick_top_n_probs = pick_top_n_probs

        assert top_n_mode in ['min', 'max', 'random', 'first', 'last'], \
            f"top_n_mode {top_n_mode} not recognized."
        self.top_n_mode = top_n_mode

    def select_generation_targets(self, state: BaseGenerationState, max_model_target_tokens: int) -> BaseGenerationState:

        assert self.num_tokens_per_step <= max_model_target_tokens, \
            f"num_tokens_per_step {self.num_tokens_per_step} exceeds max_model_target_tokens {max_model_target_tokens}."

        for mod in state.target_dict.keys():
            not_done_idx = list(state.target_dict[mod].idx_req - state.target_dict[mod].idx_done)
            if self.sampling == 'sequential':
                idx_to_gen = sorted(not_done_idx)[:self.num_tokens_per_step]
            elif self.sampling == 'random':
                idx_to_gen = [not_done_idx[i] for i in torch.randperm(len(not_done_idx))[:self.num_tokens_per_step]]
            else:
                raise NotImplementedError
            state.target_dict[mod].idx_to_gen = idx_to_gen

        return state

    def process_logits(self, state: BaseGenerationState, temperature: float) -> BaseGenerationState:

        for mod_status in state.target_dict.values():

            gen_probs = torch.softmax(mod_status.gen_logits, dim=-1)
            ext_prob = gen_probs.max(1)[0]
            
            if (self.pick_top_n_probs is not None
                and self.pick_top_n_probs <= gen_probs.size(0)
                and (self.prob_threshold is None or ext_prob.min() <= self.prob_threshold)):
                
                if self.top_n_mode == 'random':
                    top_idx = sorted(
                        torch.randperm(gen_probs.size(0))[:self.pick_top_n_probs].tolist()
                    )
                elif self.top_n_mode == 'first':
                    top_idx = sorted(
                        range(len(mod_status.idx_to_gen)),
                        key=lambda i: mod_status.idx_to_gen[i],
                    )[:self.pick_top_n_probs]
                elif self.top_n_mode == 'last':
                    top_idx = sorted(
                        range(len(mod_status.idx_to_gen)),
                        key=lambda i: mod_status.idx_to_gen[i],
                        reverse=True,
                    )[:self.pick_top_n_probs]
                else:
                    if self.top_n_mode == 'min':
                        ext_prob = - ext_prob
                    top_idx = sorted(torch.topk(ext_prob, self.pick_top_n_probs, dim=-1)[1].tolist())
                gen_probs = gen_probs[top_idx]
                mod_status.gen_logits = mod_status.gen_logits[top_idx]
                mod_status.idx_to_fill = [mod_status.idx_to_gen[i] for i in top_idx]
            else:
                mod_status.idx_to_fill = mod_status.idx_to_gen
            mod_status.gen_probs = gen_probs
            mod_status.gen_sampling_probs = torch.softmax(mod_status.gen_logits / temperature, dim=-1)
            mod_status.tokens_to_fill = torch.multinomial(mod_status.gen_sampling_probs, num_samples=1)[:, 0]

        return state

@dataclass
class EnsembleModStatus(ModStatus):
    """
    Extended status to handle multiple passes and aggregation.
    """
    # We store lists of tensors because each index will be generated multiple times.
    ensemble_logits: Dict[int, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    ensemble_probs: Dict[int, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    ensemble_tokens: Dict[int, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    
    # Scheduling: A list of partitions (lists of indices) to process sequentially
    partitions: List[List[int]] = field(default_factory=list)
    partition_ptr: int = 0

class EnsembleGenerationOutputDict(GenerationOutputDict):
    """
    Extended output dictionary for Ensemble Generation.
    
    Attributes:
        individual_probs (torch.Tensor): A tensor containing the probabilities 
            for every pass for every token. 
            Shape: [num_tokens, max_passes, vocab_size].
            Note: Since 'generate_partitions' creates a randomized schedule, 
            some tokens may be visited more often than others. Entries are 
            padded with 0.0 where no data exists.
    """
    individual_probs: torch.Tensor


class EnsembleGenerationStrategy(BaseGenerationStrategy[BaseGenerationState]):
    """
    Ensemble generation strategy.
    
    This strategy generates tokens in random batches (partitions) such that every
    token is covered at least `num_passes` times. The input context is never 
    updated with generated tokens (frozen context).
    
    Final results are aggregated via:
      - 'hard': Majority vote on sampled tokens.
      - 'soft': Average of probabilities across passes.
    """

    def __init__(self, num_tokens_per_step: int, num_passes: int, agg_mode: str = 'soft'):
        """
        Args:
            num_tokens_per_step (int): Size of random batch (p).
            num_passes (int): Minimum coverage per token (k).
            agg_mode (str): 'soft' (average probs) or 'hard' (majority vote).
        """
        self.p = num_tokens_per_step
        self.k = num_passes
        self.agg_mode = agg_mode.lower()
        
        if self.agg_mode not in ['soft', 'hard']:
            raise ValueError(f"agg_mode must be 'soft' or 'hard', got {self.agg_mode}")

    def generate_partitions(self, tokens: List[int], p: int, k: int) -> List[List[int]]:
        """
        Generates random partitions of size p where each token appears at least k times.
        (Your provided implementation)
        """
        n = len(tokens)
        if p > n:
            raise ValueError(f"Partition size p={p} cannot be larger than number of tokens N={n}.")
            
        indices = np.arange(n)
        counts = np.zeros(n, dtype=int)
        partitions = []
        deck = list(np.random.permutation(indices))
        
        while np.min(counts) < k:
            current_partition_indices = set()
            while len(current_partition_indices) < p:
                if not deck:
                    deck = list(np.random.permutation(indices))
                
                selected_index_in_deck = -1
                for i, candidate in enumerate(deck):
                    if candidate not in current_partition_indices:
                        selected_index_in_deck = i
                        break
                
                if selected_index_in_deck != -1:
                    token_idx = deck.pop(selected_index_in_deck)
                    current_partition_indices.add(token_idx)
                    counts[token_idx] += 1
                else:
                    raise RuntimeError("Cannot fill partition: Unique tokens exhausted.")
            
            partition = [tokens[i] for i in current_partition_indices]
            partitions.append(partition)
            
        return partitions

    def _initialize_state(self, mod_dict: dict, target_set: Set[str]) -> BaseGenerationState:
        # Initialize standard state
        target_dict = {}
        for mod in target_set:
            # Identify all indices that need generation
            idx_req = set(torch.where(mod_dict[mod]['input_mask'][0])[0].tolist())
            idx_req_list = sorted(list(idx_req))
            
            # Generate the schedule (partitions)
            # Handle edge case where p > total_tokens (though validation exists in generate_partitions)
            p_adj = min(self.p, len(idx_req_list))
            partitions = self.generate_partitions(idx_req_list, p_adj, self.k)
            
            # Use our custom ModStatus
            target_dict[mod] = EnsembleModStatus(
                idx_req=idx_req,
                partitions=partitions,
                partition_ptr=0
            )

        state = BaseGenerationState(
            mod_dict=mod_dict,
            target_dict=target_dict,
            remaining_tokens=0 # Will be calculated by _calc_remaining_tokens
        )
        state.remaining_tokens = self._calc_remaining_tokens(state.target_dict)
        return state

    def _calc_remaining_tokens(self, target_dict: Dict[str, EnsembleModStatus]) -> int:
        """
        Override: Remaining work is defined by the number of unprocessed partitions,
        not the number of unfilled tokens.
        """
        remaining = 0
        for mod_status in target_dict.values():
            remaining += (len(mod_status.partitions) - mod_status.partition_ptr)
        return remaining

    def select_generation_targets(self, state: BaseGenerationState, max_model_target_tokens: int) -> BaseGenerationState:
        for mod, mod_status in state.target_dict.items():
            if mod_status.partition_ptr < len(mod_status.partitions):
                # Select the next pre-calculated partition
                # We do NOT use max_model_target_tokens here because we enforced `p` earlier
                # or we assume `p` <= max_model_target_tokens
                next_batch = mod_status.partitions[mod_status.partition_ptr]
                mod_status.idx_to_gen = next_batch
                
                # Advance pointer
                mod_status.partition_ptr += 1
            else:
                mod_status.idx_to_gen = []
                
        return state

    def process_logits(self, state: BaseGenerationState, temperature: float) -> BaseGenerationState:
        for mod_status in state.target_dict.values():
            if not mod_status.idx_to_gen:
                continue
                
            # Standard calculation
            mod_status.gen_probs = torch.softmax(mod_status.gen_logits, dim=-1)
            mod_status.gen_sampling_probs = torch.softmax(mod_status.gen_logits / temperature, dim=-1)
            
            # We sample tokens now. 
            # Even for Soft voting, we might want the sampled token for other metrics, 
            # but specifically for Hard voting, this sampled token is the "vote".
            mod_status.tokens_to_fill = torch.multinomial(mod_status.gen_sampling_probs, num_samples=1)[:, 0]
            mod_status.idx_to_fill = mod_status.idx_to_gen

        return state

    def _fill_mod_dict(self, state: BaseGenerationState) -> BaseGenerationState:
        """
        CRITICAL OVERRIDE: 
        1. We do NOT update state.mod_dict[mod]['tensor'] (Input remains masked).
        2. We append results to our lists instead of overwriting.
        """
        for mod, mod_status in state.target_dict.items():
            if mod_status.idx_to_fill:
                # Do NOT fill the input tensor
                # tensor = state.mod_dict[mod]['tensor']
                # tensor[0, mod_status.idx_to_fill] = ... (Skipped)
                
                # Do NOT mark idx as "done" in the traditional sense,
                # as we might revisit them. We track progress via partition_ptr.
                # mod_status.idx_done.update(...) (Skipped)

                # Store the results
                # We zip idx, logit, prob, token and append to the respective lists
                for idx, logit, prob, token in zip(
                    mod_status.idx_to_fill, 
                    mod_status.gen_logits, 
                    mod_status.gen_probs,
                    mod_status.tokens_to_fill
                ):
                    mod_status.ensemble_logits[idx].append(logit.detach().cpu())
                    mod_status.ensemble_probs[idx].append(prob.detach().cpu())
                    mod_status.ensemble_tokens[idx].append(token.detach().cpu())

                # Cleanup for next step
                mod_status.idx_to_gen = []
                mod_status.gen_logits = torch.tensor([])
                mod_status.idx_to_fill = []
                mod_status.tokens_to_fill = torch.tensor([])

        state.remaining_tokens = self._calc_remaining_tokens(state.target_dict)
        return state

    def _make_output(self, state: BaseGenerationState) -> Dict[str, EnsembleGenerationOutputDict]:
        output = {}
        for mod, mod_status in state.target_dict.items():
            req_idx = sorted(list(mod_status.idx_req))
            
            final_tokens_list = []
            final_logits_list = []
            final_probs_list = []
            final_sampling_probs_list = [] 
            
            # For exporting individual probs, we need to handle ragged lists (uneven counts)
            # 1. Determine maximum passes any token received
            max_passes = 0
            if req_idx:
                max_passes = max(len(mod_status.ensemble_probs[idx]) for idx in req_idx)
            
            padded_individual_probs_list = []

            for idx in req_idx:
                # Retrieve collected data
                # Shapes: [k_i, vocab_size]
                idx_probs = torch.stack(mod_status.ensemble_probs[idx]) 
                idx_logits = torch.stack(mod_status.ensemble_logits[idx])
                idx_tokens = torch.stack(mod_status.ensemble_tokens[idx])
                
                # --- Padding for Individual Export ---
                # We need [max_passes, vocab_size]
                current_k, vocab_size = idx_probs.shape
                if current_k < max_passes:
                    # Pad with -1
                    padding = torch.full((max_passes - current_k, vocab_size), -1, dtype=idx_probs.dtype)
                    padded_probs = torch.cat([idx_probs, padding], dim=0)
                else:
                    padded_probs = idx_probs
                padded_individual_probs_list.append(padded_probs)

                # --- Aggregation Logic ---
                if self.agg_mode == 'soft':
                    # Average Probability
                    avg_probs = torch.mean(idx_probs, dim=0)
                    best_token = torch.argmax(avg_probs)
                    
                    final_tokens_list.append(best_token)
                    final_probs_list.append(avg_probs)
                    final_logits_list.append(torch.log(avg_probs + 1e-10)) 
                    final_sampling_probs_list.append(avg_probs) 

                elif self.agg_mode == 'hard':
                    # Majority Vote (Mode)
                    mode_token = torch.mode(idx_tokens, dim=0).values
                    
                    # Calculate vote frequency
                    vote_counts = torch.bincount(idx_tokens, minlength=vocab_size).float()
                    vote_probs = vote_counts / idx_tokens.size(0)
                    
                    final_tokens_list.append(mode_token)
                    final_probs_list.append(vote_probs)
                    final_logits_list.append(torch.log(vote_probs + 1e-10))
                    final_sampling_probs_list.append(vote_probs)

            # Stack final results
            # individual_probs shape: [num_tokens, max_passes, vocab_size]
            all_individual_probs = torch.stack(padded_individual_probs_list) if padded_individual_probs_list else torch.tensor([])

            output[mod] = EnsembleGenerationOutputDict(
                tokens=torch.stack(final_tokens_list) if final_tokens_list else torch.tensor([]),
                logits=torch.stack(final_logits_list) if final_logits_list else torch.tensor([]),
                probs=torch.stack(final_probs_list) if final_probs_list else torch.tensor([]),
                sampling_probs=torch.stack(final_sampling_probs_list) if final_sampling_probs_list else torch.tensor([]),
                idx=req_idx,
                individual_probs=all_individual_probs
            )
            
        return output
    

class MultiPassGenerationStrategy(BaseGenerationStrategy):
    """
    Wrapper strategy that executes an inner generation strategy multiple times
    (multi-pass) and aggregates the results using Soft or Hard voting.
    
    This is preferred over inheritance for AR strategies because it ensures 
    a perfectly clean state reset for every pass without complex rollback logic.
    """

    def __init__(self, 
                 base_strategy: BaseGenerationStrategy, 
                 num_passes: int, 
                 agg_mode: str = 'soft'):
        """
        Args:
            base_strategy (BaseGenerationStrategy): The initialized strategy to run (e.g., ARGenerationStrategy).
            num_passes (int): Number of complete generation passes to perform.
            agg_mode (str): 'soft' (average probs) or 'hard' (majority vote).
        """
        self.base_strategy = base_strategy
        self.num_passes = num_passes
        self.agg_mode = agg_mode.lower()

        if self.agg_mode not in ['soft', 'hard']:
            raise ValueError(f"agg_mode must be 'soft' or 'hard', got {self.agg_mode}")

    @torch.no_grad()
    def generate(self,
                 model_forward: Callable,
                 mod_dict: dict,
                 modality_info: Dict[str, dict],
                 target_set: Set[str],
                 max_model_target_tokens: int,
                 temperature: float,
                 is_target_autoregr: bool,
                 verbose: bool = True) -> Dict[str, EnsembleGenerationOutputDict]:
        
        # 1. Container for results from all passes
        # Structure: {mod_name: [OutputDict_pass_1, OutputDict_pass_2, ...]}
        pass_outputs = defaultdict(list)

        if verbose:
            # make a tqdm pbar
            pbar = tqdm(total=self.num_passes, desc=f"Ensemble Generation Passes", unit="pass")

        # 2. Execution Loop
        with pbar if verbose else nullcontext():
            for i in range(self.num_passes):

                # Deep copy mod_dict to ensure the next pass starts with a clean input state.
                # Essential for AR strategies that modify tensors in-place.
                current_mod_dict = self._deep_copy_mod_dict(mod_dict)

                # Execute the base strategy
                output = self.base_strategy.generate(
                    model_forward=model_forward,
                    mod_dict=current_mod_dict,
                    modality_info=modality_info,
                    target_set=target_set,
                    max_model_target_tokens=max_model_target_tokens,
                    temperature=temperature,
                    is_target_autoregr=is_target_autoregr,
                    verbose=False
                )

                # Collect results
                for mod, out_dict in output.items():
                    pass_outputs[mod].append(out_dict)

                if verbose:
                    pbar.update(1)

        # 3. Aggregation
        return self._aggregate_outputs(pass_outputs)

    def _deep_copy_mod_dict(self, mod_dict: dict) -> dict:
        """
        Creates a deep copy of the mod_dict, cloning tensors to prevent 
        in-place modifications from affecting subsequent passes.
        """
        new_dict = {}
        for key, value in mod_dict.items():
            if isinstance(value, dict):
                new_dict[key] = self._deep_copy_mod_dict(value)
            elif isinstance(value, torch.Tensor):
                new_dict[key] = value.clone()
            else:
                new_dict[key] = copy.deepcopy(value)
        return new_dict

    def _aggregate_outputs(self, pass_outputs: Dict[str, List[GenerationOutputDict]]) -> Dict[str, EnsembleGenerationOutputDict]:
        final_output = {}

        for mod, outputs_list in pass_outputs.items():
            if not outputs_list:
                continue

            # Stack data from all passes
            # individual_probs shape: [num_tokens, num_passes, vocab_size]
            # Note: We permute because EnsembleOutput expects [num_tokens, passes, vocab]
            # but usually we get [passes, num_tokens, vocab] naturally from stacking lists.
            
            # 1. Stack Probabilities: [num_passes, num_tokens, vocab]
            all_probs_stacked = torch.stack([out['probs'] for out in outputs_list], dim=0)
            # 2. Stack Tokens: [num_passes, num_tokens]
            all_tokens_stacked = torch.stack([out['tokens'] for out in outputs_list], dim=0)
            
            # Get consistent metadata from the first pass
            req_idx = outputs_list[0]['idx']
            vocab_size = all_probs_stacked.shape[-1]
            
            final_tokens = None
            final_probs = None
            final_logits = None
            final_sampling_probs = None

            if self.agg_mode == 'soft':
                # Average Probabilities across passes
                avg_probs = torch.mean(all_probs_stacked, dim=0) # [num_tokens, vocab]
                best_tokens = torch.argmax(avg_probs, dim=-1)

                final_tokens = best_tokens
                final_probs = avg_probs
                final_logits = torch.log(avg_probs + 1e-10)
                final_sampling_probs = avg_probs

            elif self.agg_mode == 'hard':
                # Majority Vote on Tokens
                # mode.values contains the most frequent token
                mode_result = torch.mode(all_tokens_stacked, dim=0)
                final_tokens = mode_result.values

                # Calculate vote frequency for probabilities
                # [num_tokens, vocab]
                vote_probs = torch.zeros_like(all_probs_stacked[0])
                for t_idx in range(all_tokens_stacked.size(1)):
                    tokens_at_idx = all_tokens_stacked[:, t_idx]
                    counts = torch.bincount(tokens_at_idx, minlength=vocab_size).float()
                    vote_probs[t_idx] = counts / self.num_passes

                final_probs = vote_probs
                final_logits = torch.log(vote_probs + 1e-10)
                final_sampling_probs = vote_probs

            # Prepare individual_probs in format: [num_tokens, num_passes, vocab_size]
            individual_probs = all_probs_stacked.permute(1, 0, 2)

            final_output[mod] = EnsembleGenerationOutputDict(
                tokens=final_tokens,
                logits=final_logits,
                probs=final_probs,
                sampling_probs=final_sampling_probs,
                idx=req_idx,
                individual_probs=individual_probs
            )

        return final_output

    # The following abstract methods must be implemented to satisfy ABC, 
    # but they are not used because 'generate' is overridden completely.
    def select_generation_targets(self, state, max_tokens): return state
    def process_logits(self, state, temp): return state
    def _initialize_state(self, mod_dict, target_set): return None