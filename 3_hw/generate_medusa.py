import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        
        generated_tokens = []
        device = input_ids.device

        for _ in range(self.max_output_len):
            outputs = self.model.base_model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]

            next_token = torch.argmax(logits, dim=-1)

            if next_token.item() == self.eos_token_id:
                break
            generated_tokens.append(next_token.item())
            next_tensor = next_token.unsqueeze(-1).unsqueeze(-1).to(device)
            input_ids = torch.cat([input_ids, next_tensor], dim=-1)

        return torch.tensor(generated_tokens)

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    

        current_ids = input_ids
        device = input_ids.device
        num_tokens_generated = 5 
        # Some single head tokens to provide a good start.

        for _ in range(num_tokens_generated):
            outputs = self.model.base_model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]

            next_token = torch.argmax(logits, dim=-1)

            if next_token.item() == self.eos_token_id:
                break

            next_tensor = next_token.unsqueeze(-1).unsqueeze(-1).to(device)
            current_ids = torch.cat([current_ids, next_tensor], dim=-1)
        
        
        # The main thing: loop over medusa generation till max output size
        while (num_tokens_generated + self.no_heads <= self.max_output_len):
            medusa_logits, _, logits = self.model(
                input_ids=current_ids,
                output_orig=True,
                medusa_forward=True
            )

            candidates = [current_ids]
            eos_candidates = []
            eos_scores = []
            scores = [0.0]

            for s in range(self.no_heads):
                required_logits = None
                if (s == 0):
                    required_logits = logits[0, -1, :]
                else:
                    required_logits = medusa_logits[s - 1, 0,  -1, :]
                log_probs = nn.functional.log_softmax(required_logits)

                new_candidates = []
                new_scores = []

                for c in range (len(candidates)):
                    for y_hat in torch.topk(log_probs, 3 * self.beam_width).indices:
                        # 3 * self.beam_width -- heuristic, for more sequences without EOS tokens

                        new_score = scores[c] + log_probs[y_hat]
                        next_token = torch.Tensor([y_hat]).unsqueeze(-1).to(device)
                        new_candidate = torch.cat([candidates[c], next_token], dim=-1)

                        if (y_hat == self.eos_token_id):
                            eos_scores.append(new_score * 5)
                            # negative score * 5 -- penalized early terminating (EOS) sequences
                            eos_candidates.append(new_candidate)
                        
                        else:
                            if (y_hat == candidates[c][0, -1]):
                                # penalizing repeated tokens: negative score * 3
                                new_score *= 3
                            elif (y_hat == candidates[c][0, -2]):
                                new_score *= 2
                            elif (y_hat == candidates[c][0, -3]):
                                new_score *= 2
                            
                            # With the above, we penalize repeated values
                            new_scores.append(new_score)
                            new_candidates.append(new_candidate)

                k_val = min(self.beam_width, len(new_scores))
                new_scores_tensor = torch.Tensor(new_scores)
                topk_scores = torch.topk(new_scores_tensor, k_val).indices

                scores = [new_scores[i] for i in topk_scores]
                candidates = [new_candidates[i] for i in topk_scores]
            
            scores_tensor = torch.Tensor(scores)
            index = torch.argmax(scores_tensor).item()
            current_ids = candidates[index]

            if (eos_candidates != [] and eos_scores != []):
                eos_scores_tensor = torch.Tensor(eos_scores)
                eos_index = torch.argmax(eos_scores_tensor).item()

                if (eos_scores[eos_index] > scores[index]):
                    current_ids = eos_candidates[eos_index]
                    break
            
            num_tokens_generated += self.no_heads
            current_ids = current_ids.int()
            
        
        current_ids_integers = current_ids.int()
        current_ids_linear = current_ids_integers.reshape(-1)
        generated_tokens = current_ids_linear[input_ids.shape[1]:]

        return generated_tokens
        
        
            
