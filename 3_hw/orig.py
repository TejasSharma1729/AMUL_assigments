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
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

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
        
        tokenized_words = []
        for word in word_list:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if tokens:  
                tokenized_words.append(tokens)
        
        if not tokenized_words:
            return torch.tensor([], dtype=torch.long)
        
        trie = Trie()
        for tokens in tokenized_words:
            trie.insert(tokens)
        
        generated_tokens = []
        current_sequence = []
        device = input_ids.device
        
        for _ in range(self.max_output_len):
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Create mask for valid tokens
            mask = torch.full_like(logits, -float('inf'))
            
            # Get valid next tokens based on current sequence
            if current_sequence:
                # We're in the middle of forming a word - only allow tokens that continue valid words
                valid_next = trie.get_valid_next_tokens(current_sequence)
                mask[:, list(valid_next)] = logits[:, list(valid_next)]
            else:
                # Need to start a new word - only allow tokens that start valid words
                word_start_tokens = set(trie.root.children.keys())
                mask[:, list(word_start_tokens)] = logits[:, list(word_start_tokens)]
            
            # Try multiple candidates if first choice is invalid
            next_token = None
            for _ in range(5):  # Try top candidates
                if mask.isneginf().all():  # No valid tokens left
                    break
                
                candidate = torch.argmax(mask, dim=-1).item()
                
                # Check if token is valid
                test_sequence = current_sequence + [candidate]
                if (current_sequence and candidate in trie.get_valid_next_tokens(current_sequence)) or (not current_sequence and candidate in trie.root.children):
                    next_token = candidate
                    break
                else:
                    # Mask this invalid token and try next best option
                    mask[0, candidate] = -float('inf')
            
            if next_token is None:
                break  # Couldn't find a valid token
                
            # Handle EOS token
            if next_token == self.eos_token_id:
                # Verify current sequence is empty (not in middle of word)
                if not current_sequence:
                    break
                else:
                    continue  # Don't allow EOS in middle of word
                
            # Update state
            generated_tokens.append(next_token)
            current_sequence.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=-1)
            
            # Check if current sequence completes a valid word
            if trie.is_valid_word(current_sequence):
                current_sequence = []  # Reset for next word
                
        #print the word_bag:
        print(f"Word bag: {word_list}")
        
        # Print the generated sentence
        generated_text = ""
        if generated_tokens:
            generated_text = self.tokenizer.decode(generated_tokens)
            print(f"Generated sentence: {generated_text}")
        
        return torch.tensor(generated_tokens, device=device)
        