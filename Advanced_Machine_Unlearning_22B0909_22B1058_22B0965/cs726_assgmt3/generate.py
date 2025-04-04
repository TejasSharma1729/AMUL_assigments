import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
        '''
            Implement Greedy decoding technique. (refer assignment document for more details)

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
        # TODO:
        generated_tokens = []
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]

            next_token = torch.argmax(logits, dim=-1)

            if next_token.item() == self.eos_token_id:
                break
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

        return torch.tensor(generated_tokens)
        # END TODO
        
    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Random sampling technique. (refer assignment document for more details)

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
        # TODO:
        generated_tokens = []
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]

            probs = nn.functional.softmax(logits, dim=-1)
            adjusted_probs = probs.pow(1 / self.tau)  # Apply temperature scaling correctly
            adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)  # Normalize

            next_token = torch.multinomial(adjusted_probs, num_samples=1)  # Keep it 2D

            if next_token.item() == self.eos_token_id:
                break
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # next_token is now 2D

        return torch.tensor(generated_tokens)
        # END TODO
    
    
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

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
        # TODO:
        generated_tokens = []
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for last generated token
            
            probs = nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, self.k, dim=-1)  # Get top-k values & indices
            
            # Normalize only the top-k probabilities
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            
            # Sample from the restricted top-k probabilities
            sampled_index = torch.multinomial(topk_probs.squeeze(0), num_samples=1)
            next_token = topk_indices[0, sampled_index]  # Convert back to actual token ID
            
            if next_token.item() == self.eos_token_id:
                break
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)  # Ensure 2D shape
        
        return torch.tensor(generated_tokens)
        
        # END TODO
    
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

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
        # TODO:
        generated_tokens = []
        
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for last generated token
            
            probs = nn.functional.softmax(logits, dim=-1)
            
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Compute cumulative sum
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Get the indices of tokens within the nucleus
            nucleus_mask = cumsum_probs < self.p  # Boolean mask
            
            # Ensure we always include at least one token (the highest probability one)
            nucleus_mask[..., 0] = True  
            
            # Apply mask
            topk_probs = sorted_probs[nucleus_mask]
            topk_indices = sorted_indices[nucleus_mask]
            
            # Normalize only the selected top-p probabilities
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            
            # Sample from the nucleus
            sampled_index = torch.multinomial(topk_probs, num_samples=1)  # 1D tensor
            
            next_token = topk_indices[sampled_index]  # Correct indexing
            
            if next_token.item() == self.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        return torch.tensor(generated_tokens)
        # END TODO
    