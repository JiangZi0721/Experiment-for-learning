import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Vocabulary definition (13 characters)
# '0'-'9', '+', ' ', '_'
CHARS = [' ', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

class AdditionDataset(Dataset):
    def __init__(self, questions, answers, reverse=False):
        """
        questions: list of 7-char strings, e.g. '16+75  '
        answers: list of 5-char strings, e.g. '_91  '
        reverse: whether to reverse the input question sequence
        """
        self.reverse = reverse
        self.raw_questions = questions
        self.raw_answers = answers
        
        # Convert to numpy id arrays
        q_ids = []
        for q in questions:
            chars = list(q)
            if self.reverse:
                chars = chars[::-1]
            q_ids.append([CHAR_TO_ID[c] for c in chars])
            
        a_ids = []
        for a in answers:
            a_ids.append([CHAR_TO_ID[c] for c in a])
            
        self.x = torch.tensor(q_ids, dtype=torch.long)
        self.y = torch.tensor(a_ids, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def generate_addition_dataset(num_samples=50000, seed=42):
    """
    Generate unique addition pairs: A + B = C
    A, B in [0, 999]
    Question format: '{A}+{B}'.ljust(7)
    Answer format: '_{A+B}'.ljust(5)
    """
    random.seed(seed)
    seen = set()
    questions = []
    answers = []
    
    # We want num_samples unique questions
    while len(questions) < num_samples:
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        
        q_str = f"{a}+{b}".ljust(7)
        ans_val = a + b
        a_str = f"_{ans_val}".ljust(5)
        
        questions.append(q_str)
        answers.append(a_str)
        
    return questions, answers


def get_dataloaders(num_samples=50000, train_ratio=0.9, batch_size=256, reverse=False, seed=42):
    """
    Returns (train_loader, test_loader, train_questions, test_questions, train_answers, test_answers)
    """
    questions, answers = generate_addition_dataset(num_samples=num_samples, seed=seed)
    
    # Shuffle and split
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    train_size = int(num_samples * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    train_q = [questions[i] for i in train_idx]
    train_a = [answers[i] for i in train_idx]
    test_q = [questions[i] for i in test_idx]
    test_a = [answers[i] for i in test_idx]
    
    train_dataset = AdditionDataset(train_q, train_a, reverse=reverse)
    test_dataset = AdditionDataset(test_q, test_a, reverse=reverse)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, (train_q, train_a), (test_q, test_a)


if __name__ == '__main__':
    train_l, test_l, tr, te = get_dataloaders(num_samples=1000, reverse=True)
    for bx, by in train_l:
        print('bx shape:', bx.shape, 'by shape:', by.shape)
        q_chars = ''.join([ID_TO_CHAR[i.item()] for i in bx[0]])
        a_chars = ''.join([ID_TO_CHAR[i.item()] for i in by[0]])
        print(f'Reversed Question: {repr(q_chars)} -> Answer: {repr(a_chars)}')
        break
