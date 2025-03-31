import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TrieNode:
    """ Helper class to build a trie for word-constrained decoding. """
    def __init__(self):
        self.children = {}
        self.is_end = False
class Trie:
    """ Trie for storing valid token sequences from the word list. """
    def __init__(self):
        self.root = TrieNode()
        self.all_tokens = set()  # Track all valid tokens in the word list

    def insert(self, tokenized_word):
        """ Insert a tokenized word (list of token IDs) into the trie. """
        node = self.root
        for token in tokenized_word:
            self.all_tokens.add(token)
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True  # Mark the end of a valid word

    def get_valid_next_tokens(self, prefix_tokens):
        """ Return valid next tokens given the current prefix. """
        node = self.root
        for token in prefix_tokens:
            if token in node.children:
                node = node.children[token]
            else:
                return set()  # If prefix is invalid, no valid next tokens
        return set(node.children.keys())  # Return valid next tokens
    
    def is_valid_word(self, token_sequence):
        """ Check if a sequence of tokens forms a complete valid word. """
        node = self.root
        for token in token_sequence:
            if token in node.children:
                node = node.children[token]
            else:
                return False
        return node.is_end

def tokenize_words(tokenizer, word_list):
    """ Tokenize each word in the word list using the model's tokenizer. """
    return [tokenizer.encode(word, add_special_tokens=False) for word in word_list]

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_id
        self.max_output_len = max_output_len
        self.alpha = 0.8  # Blending weight between model probability and constraints

    def __call__(self, input_ids, word_list):
        tokenized_words = tokenize_words(self.tokenizer, word_list)
        if not tokenized_words:
            return torch.tensor([], dtype=torch.long)
        
        trie = Trie()
        all_words = set()
        terminal_words = set()
        for tokens in tokenized_words:
            trie.insert(tokens)
            all_words.add(tuple(tokens))
            if tokens and tokens[-1] == self.tokenizer.encode(".", add_special_tokens=False)[-1]:
                terminal_words.add(tuple(tokens))
        
        words_used = set()
        generated_tokens = []
        current_sequence = []
        device = input_ids.device
        prev_token = None
        
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
                unused_words = all_words - words_used
                valid_starts = {tokens[0] for tokens in unused_words}
                mask[:, list(valid_starts)] = logits[:, list(valid_starts)]
            
            # Soft blending instead of hard masking
            blended_logits = self.alpha * logits + (1 - self.alpha) * mask
            blended_logits = torch.where(mask.isneginf(), mask, blended_logits)
            
            next_token = None
            for _ in range(5):
                if blended_logits.isneginf().all():
                    break
                candidate = torch.argmax(blended_logits, dim=-1).item()
                if candidate == prev_token:
                    blended_logits[0, candidate] = -float('inf')
                    continue
                test_sequence = current_sequence + [candidate]
                if (current_sequence and candidate in trie.get_valid_next_tokens(current_sequence)) or (not current_sequence and candidate in trie.root.children):
                    next_token = candidate
                    break
                else:
                    blended_logits[0, candidate] = -float('inf')
            
            if next_token is None:
                break
            
            if next_token == self.eos_token_id:
                if words_used == all_words and terminal_words.issubset(words_used):
                    break
                else:
                    continue
            
            generated_tokens.append(next_token)
            current_sequence.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=-1)
            prev_token = next_token
            
            if trie.is_valid_word(current_sequence):
                words_used.add(tuple(current_sequence))
                current_sequence = []
        
        print(f"Word bag: {word_list}")
        if generated_tokens:
            print(f"Generated sentence: {self.tokenizer.decode(generated_tokens)}")
        
        return torch.tensor(generated_tokens, device=device)