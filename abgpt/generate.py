import argparse
import os
import torch
import random
import math
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

CHAIN_LENGTHS = {
    "light": (100, 120),
    "heavy": (110, 140)
}
MODEL_REPO = "deskk/AbGPT"

def parse_args():
    parser = argparse.ArgumentParser(description='BCR Sequence Generation')
    parser.add_argument('--chain_type', type=str, choices=['light', 'heavy'], required=True, help='Chain type: light or heavy')
    parser.add_argument('--starting_residue', type=str, help='Starting residue for sequence generation')
    parser.add_argument('--num_seqs', type=int, help='Number of sequences to generate')
    parser.add_argument('--num_seqs_each_starting_residue', type=int, help='Number of sequences to generate for each starting residue')
    args = parser.parse_args()
    return args

def generate_bcr_sequences(num_sequences=100, chain_type="light", starting_residue=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    min_length = 20 if chain_type == "light" else 28
    starting_prompt = f"<|endoftext|>{starting_residue}"
    abgpt_pipeline = pipeline('text-generation', model=MODEL_REPO, tokenizer=MODEL_REPO, device=0 if device == "cuda" else -1)
    generated_sequences = abgpt_pipeline(
        starting_prompt,
        min_length=min_length,
        do_sample=True,
        top_k=950,
        repetition_penalty=1.2,
        num_return_sequences=num_sequences,
        eos_token_id=0
    )
    return [seq['generated_text'].replace("<|endoftext|>", "").strip() for seq in generated_sequences]

def save_sequences(filename, sequences):
    with open(filename, 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n\n")

def calculate_perplexity(sequence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return math.exp(loss.item())

def preprocess_sequence(sequence):
    return "<|endoftext|>" + '\n'.join([sequence[i:i+60] for i in range(0, len(sequence), 60)]) + "<|endoftext|>"

def save_best_sequences(filename, sequences):
    with open(filename, 'w') as file:
        for seq, ppl in sequences:
            cleaned_seq = seq.replace("<|endoftext|>", "").strip()
            formatted_seq = "\n".join([cleaned_seq[i:i+60] for i in range(0, len(cleaned_seq), 60)])
            file.write(f"Sequence: {formatted_seq}\nPerplexity: {ppl}\n\n")

def process_sequences_from_file(file_path, model, tokenizer, device):
    with open(file_path, 'r') as file:
        sequences = file.read().split("\n\n")
    filtered_sequences = []
    for seq_block in tqdm(sequences, desc=f"Calculating sequences for {file_path}"):
        seq = seq_block.strip()
        if seq:
            concatenated_seq = ''.join(seq.splitlines())
            preprocessed_seq = preprocess_sequence(concatenated_seq)
            ppl = calculate_perplexity(preprocessed_seq, model, tokenizer, device)
            if ppl < 13.0:
                filtered_sequences.append((concatenated_seq, ppl))
    return sorted(filtered_sequences, key=lambda x: x[1])

def read_and_filter_sequences(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    chunks = text.split('Sequence: ')[1:]
    sequences = []
    for chunk in chunks:
        sequence = chunk.split('Perplexity:')[0].strip().replace('\n', '')
        if 'X' not in sequence and 'B' not in sequence:
            sequences.append(sequence)
    return sequences

def format_sequence(sequence):
    return '\n'.join(sequence[i:i+60] for i in range(0, len(sequence), 60))

def save_to_fasta(sequences, output_file, sequence_counter, seen_sequences):
    with open(output_file, 'a') as f:
        for sequence in sequences:
            if sequence not in seen_sequences:
                seen_sequences.add(sequence)
                sequence_counter += 1
                formatted_sequence = format_sequence(sequence)
                f.write(f'>Sequence_{sequence_counter}\n')
                f.write(f'{formatted_sequence}\n')
    return sequence_counter, seen_sequences

def process_directory(directory_path, output_file):
    sequence_counter = 0
    seen_sequences = set()
    with open(output_file, 'w') as f:
        pass
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            sequences = read_and_filter_sequences(file_path)
            sequence_counter, seen_sequences = save_to_fasta(sequences, output_file, sequence_counter, seen_sequences)

def generate_specific_sequences(args):
    sequences = generate_bcr_sequences(
        num_sequences=args.num_seqs,
        chain_type=args.chain_type,
        starting_residue=args.starting_residue
    )
    output_dir = 'bcr_design'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{args.chain_type}_{args.starting_residue}.txt')
    save_sequences(filename, sequences)

def generate_bcr_library(args):
    output_dir = 'bcr_library'
    os.makedirs(output_dir, exist_ok=True)
    combined_filename = os.path.join(output_dir, f'{args.chain_type}_BCR_library.txt')
    all_sequences = []
    residues = args.starting_residue.split(',')
    for starting_residue in residues:
        sequences = generate_bcr_sequences(
            num_sequences=args.num_seqs_each_starting_residue,
            chain_type=args.chain_type,
            starting_residue=starting_residue
        )
        all_sequences.extend(sequences)
    save_sequences(combined_filename, all_sequences)

def main():
    args = parse_args()
    if args.starting_residue and args.num_seqs:
        generate_specific_sequences(args)
    elif args.starting_residue and args.num_seqs_each_starting_residue:
        generate_bcr_library(args)
    else:
        print("Please specify either --starting_residue and --num_seqs or --starting_residue and --num_seqs_each_starting_residue for library generation.")
