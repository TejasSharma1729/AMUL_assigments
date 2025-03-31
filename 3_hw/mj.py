import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.all_tokens = set()

    def insert(self, tokenized_word):
        node = self.root
        for token in tokenized_word:
            self.all_tokens.add(token)
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True

    def get_valid_next_tokens(self, prefix_tokens):
        node = self.root
        for token in prefix_tokens:
            if token in node.children:
                node = node.children[token]
            else:
                return set()
        return set(node.children.keys())
    
    def is_valid_word(self, token_sequence):
        node = self.root
        for token in token_sequence:
            if token in node.children:
                node = node.children[token]
            else:
                return False
        return node.is_end

def tokenize_words(tokenizer, word_list):
    return [tokenizer.encode(word, add_special_tokens=False) for word in word_list]

class ConstrainedTextGenerator:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, eos_id: int, max_output_len: int = 10) -> None:
        self.model = model
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tokenizer = tokenizer

    def __call__(self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list) -> Int[torch.Tensor, "batch out_seq_len"]:
        tokenized_words = [self.tokenizer.encode(word, add_special_tokens=False) for word in word_list if word]
        if not tokenized_words:
            return torch.tensor([], dtype=torch.long)
        
        trie = Trie()
        for tokens in tokenized_words:
            trie.insert(tokens)
        
        generated_tokens = []
        current_sequence = []
        used_words = set()
        word_usage_count = {tuple(tokens): 0 for tokens in tokenized_words}  # Track usage count
        device = input_ids.device
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            mask = torch.full_like(logits, -float('inf'))
            
            if current_sequence:
                valid_next = trie.get_valid_next_tokens(current_sequence)
                mask[:, list(valid_next)] = logits[:, list(valid_next)]
            else:
                word_start_tokens = set(trie.root.children.keys())
                mask[:, list(word_start_tokens)] = logits[:, list(word_start_tokens)]
            
            next_token = None
            for _ in range(5):
                if mask.isneginf().all():
                    break
                candidate = torch.argmax(mask, dim=-1).item()
                test_sequence = current_sequence + [candidate]
                if (current_sequence and candidate in trie.get_valid_next_tokens(current_sequence)) or (not current_sequence and candidate in trie.root.children):
                    next_token = candidate
                    break
                else:
                    mask[0, candidate] = -float('inf')
            
            if next_token is None:
                break
            
            if next_token == self.eos_token_id:
                if not current_sequence:
                    break
                else:
                    continue
            
            generated_tokens.append(next_token)
            current_sequence.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=-1)
            
            if trie.is_valid_word(current_sequence):
                used_words.add(tuple(current_sequence))
                word_usage_count[tuple(current_sequence)] += 1
                current_sequence = []
                
                if all(count > 0 for count in word_usage_count.values()):
                    break  # Ensure all words are used at least once
                
        print(f"Word bag: {word_list}")
        
        if generated_tokens:
            generated_text = self.tokenizer.decode(generated_tokens)
            print(f"Generated sentence: {generated_text}")
        
        return torch.tensor(generated_tokens, device=device)