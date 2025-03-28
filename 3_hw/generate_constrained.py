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

    def insert(self, tokenized_word):
        """ Insert a tokenized word (list of token IDs) into the trie. """
        node = self.root
        for token in tokenized_word:
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
        # TODO:
        
        # Tokenize words and build Trie
        tokenized_words = tokenize_words(self.tokenizer, word_list)
        trie = Trie()
        for tokens in tokenized_words:
            trie.insert(tokens)

        generated_tokens = []
        current_word_tokens = []  # Tracks tokens for current word being formed

        for _ in range(self.max_output_len):
            # Get model predictions
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Get valid next tokens based on current word formation
            valid_next_tokens = trie.get_valid_next_tokens(current_word_tokens)
            
            # If we're in the middle of forming a word, we must constrain to valid continuations
            if current_word_tokens:
                if not valid_next_tokens:  # No valid continuations
                    # Fallback: choose most probable token (could also use other strategies)
                    next_token = torch.argmax(logits, dim=-1)
                else:
                    # Mask invalid tokens by setting their probabilities to -inf
                    mask = torch.ones_like(logits) * float('-inf')
                    mask[:, list(valid_next_tokens)] = logits[:, list(valid_next_tokens)]
                    next_token = torch.argmax(mask, dim=-1)
            else:
                # Not currently forming a word - can start a new word or use any token
                # Get all tokens that start any word in the Trie
                word_start_tokens = trie.get_valid_next_tokens([])
                if word_start_tokens:
                    # Prefer tokens that start valid words
                    mask = torch.ones_like(logits) * float('-inf')
                    mask[:, list(word_start_tokens)] = logits[:, list(word_start_tokens)]
                    next_token = torch.argmax(mask, dim=-1)
                else:
                    # No words left to constrain - use regular decoding
                    next_token = torch.argmax(logits, dim=-1)

            # Check for EOS token
            if next_token.item() == self.eos_token_id:
                break

            # Update tracking
            generated_tokens.append(next_token.item())
            current_word_tokens.append(next_token.item())
            
            # Check if current token sequence completes a valid word
            node = trie.root
            for token in current_word_tokens:
                if token in node.children:
                    node = node.children[token]
                else:
                    break
            if node.is_end:
                current_word_tokens = []  # Reset for next word

            # Update input for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        return torch.tensor(generated_tokens)
        