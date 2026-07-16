from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
import json
from huggingface_hub import hf_hub_download
import torch
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F

class NoraForActionPrediction(Qwen2_5_VLForConditionalGeneration):
    # Define action token range and normalization bounds as class attributes
    # These are specific to the model's vocabulary and task
    _ACTION_TOKEN_MIN = 151665
    _ACTION_TOKEN_MAX = 153712

    def __init__(self, config) -> None:
        super().__init__(config)

        self.eval() 
        repo_id = "declare-lab/nora"
        filename = "norm_stats.json"

        file_path = hf_hub_download(repo_id=repo_id, filename=filename)

        with open(file_path, "r") as f:
            norm_stats = json.load(f)
        if config.name_or_path is not None:
            file_path = f"{config.name_or_path}/norm_stats.json"
            with open(file_path, "r") as f:
                norm_stats = json.load(f)
        self.norm_stats = norm_stats

        

    def evaluate_action(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.FloatTensor,
            image_grid_thw: Optional[Tuple[int, int, int]],
            labels: torch.LongTensor,         
            action: torch.LongTensor,
            #unnorm_key: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #action_len = self.get_action_dim(unnorm_key)

        # check last token is `</s>`
        #assert torch.all(input_ids[:, -1] == 2)
        # check last 7 tokens are action tokens (32000 - 256)
        #assert torch.all(input_ids[:, -action_len - 1: -1] >= 32000 - 256)
        # check the last -9 token is ` `
        #assert torch.all(input_ids[:, -action_len - 2] == 29871)
        # check valid attention mask
        #assert torch.all(attention_mask[:, -action_len - 2:] == 1)
        # check input_ids and labels
        #assert torch.allclose(input_ids, labels)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,  # output hidden_states
            return_dict=True,  # output dict
            image_grid_thw=image_grid_thw
        )

        last_hidden_state = outputs.hidden_states[-1]  # [B, L, hidden_dim]

        # find first valid token index
        #  IG  IG
        #  -   -  a0  a6  eos ?
        #  |   |   |   |   |  |
        # bos img img a0  a6 eos

        

        # logits
        #action_len = labels.shape[1]  
        action_len = action.shape[1]
        logits_tensor = outputs.logits[:, -action_len-2:-2]  # the last two are '\n' and predicton, and we dont care about
        logits_tensor = logits_tensor[:, :, self._ACTION_TOKEN_MIN : self._ACTION_TOKEN_MAX+1] # [B, action_len, 256]
        logprobs_tensor = F.log_softmax(logits_tensor, dim=-1)  # [B, action_len, 256]

        # idxes = labels[:, -action_len - 1: -1].unsqueeze(-1) - (32000 - 256)  # [B, action_len, 1]
        idxes = labels[:, -action_len - 1: -1].unsqueeze(-1) - self._ACTION_TOKEN_MIN   #for label. the last one is "\n"
        mask = (idxes >= 0) & (idxes < logprobs_tensor.size(2)) #to get the problog we want
        mask = mask.to(logprobs_tensor.device)
        idxes_safe = idxes.clamp(0, logprobs_tensor.size(2)-1)
        idxes_safe = idxes_safe.to(logprobs_tensor.device)
        idxes = idxes.to(logprobs_tensor.device) # [B, action_len, 1]
        #logprobs = torch.gather(logprobs_tensor, 2, idxes).squeeze(-1)  # [B, action_len]
        gathered = torch.gather(logprobs_tensor, 2, idxes_safe) 
        gathered = gathered * mask 
        gathered = gathered.squeeze(-1)     #only get the valid logprobs
        logprobs = gathered.sum(dim=1, keepdim=True)  # [B, 1]

        # entropy
        #probs_tensor = F.softmax(logits_tensor, dim=-1) # [B, action_len, 256]
        #entropy = -(probs_tensor * logprobs_tensor).sum(dim=-1)  # [B, action_len]
        #entropy = entropy.mean(dim=-1, keepdim=True) # [B, 1]

        values = torch.zeros_like(logprobs).to(logprobs.device)
        entropy = torch.zeros_like(logprobs).to(logprobs.device)

        return logprobs, entropy, values


    def predict_action_batch(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.FloatTensor,
            #unnorm_key: str,
            #do_sample: bool = True,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = input_ids.shape[0]
        #action_len = self.get_action_dim(unnorm_key)

        # assert first token is 1
        #assert torch.all(input_ids[:, 0] == 151644)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        #assert torch.all(input_ids[:, -1] == 198)
        assert torch.all(attention_mask[:, -1] == 1)

        # Run VLA inference
        output = self.generate(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            #max_new_tokens=action_len,
            return_dict_in_generate=True,
            #output_hidden_states=True,
            output_logits=True,
            #logits_processor=LogitsProcessorList([AllowedTokensLogitsProcessor()]),
            #do_sample=do_sample,
            **kwargs
        )
        generated_ids = output.sequences
        start_idx = (self._ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= self._ACTION_TOKEN_MAX)
        start_idx = torch.where(start_idx)[0]
        if len(start_idx) > 7:
            start_idx = start_idx[:7]  # make sure only keep first 7 action tokens
        generated_ids = generated_ids[:, start_idx]
        #print(f"len:{len(start_idx)}, start_idx{start_idx}")
        

        # get logits index for action tokens
        logits_idx = start_idx - input_ids.shape[1]

        # logits
        logits_tensor = torch.stack(output.logits, dim=1) # [B, action_len, vocab_size]
        logits_tensor = logits_tensor[:, logits_idx, self._ACTION_TOKEN_MIN : self._ACTION_TOKEN_MAX+1] # _ACTION_TOKEN_MAX+1？# [B, action_len, 256]
        logprobs_tensor = F.log_softmax(logits_tensor, dim=-1) # [B, action_len, 256]

        idxes = generated_ids.unsqueeze(-1) - self._ACTION_TOKEN_MIN # [B, action_len, 1]
        logprobs = torch.gather(logprobs_tensor, 2, idxes).squeeze(-1) # [B, action_len]
        logprobs = logprobs.sum(dim=1, keepdim=True) # [B, 1]

        #create values with zeros, shape is same as logprobs
        values = torch.zeros_like(logprobs).to(logprobs.device)    #don't use value.



        return values, generated_ids, logprobs  

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]