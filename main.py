import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import random

def create_random_string(str_len):
    return ''.join(random.choices(['A','C','G','T'], k=100))

def kmers(sequence: str, k: int = 6):
    """Convert DNA sequence into k-mer format."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def encode_string(genome_sequence: str):
    """Convert genome sequence into DNABERT embedding."""
    kmer_sequence = kmers(genome_sequence)
    tokens = tokenizer(kmer_sequence, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        embedding = model(**tokens).last_hidden_state.mean(dim=1)  # Average pooling
    return embedding.numpy()

# Function to search for similar genomes
def search_similar_genomes(query_sequence: str, top_k: int = 10):
    """Search for the top_k most similar reference genomes."""
    query_embedding = encode_string(query_sequence).astype("float32")
    D, I = index.search(query_embedding, top_k)
    return {"closest_genomes": I.tolist(), "distances": D.tolist()}


def mutate_kmer(dna: str, hamming_distance: int) -> str:
    """
    Generate a DNA string that is a given Hamming distance away from the input DNA string.
    
    :param dna: The original DNA string.
    :param hamming_distance: The number of positions to mutate.
    :return: A new DNA string with mutations.
    """
    dna_list = list(dna)
    nucleotides = {'A', 'C', 'G', 'T'}
    
    # Select random unique positions to mutate
    mutation_positions = random.sample(range(len(dna)), min(hamming_distance, len(dna)))
    
    for pos in mutation_positions:
        original_nucleotide = dna_list[pos]
        possible_mutations = nucleotides - {original_nucleotide}  # Exclude the original nucleotide
        dna_list[pos] = random.choice(list(possible_mutations))
    
    return ''.join(dna_list)



# Example usage
if __name__ == "__main__":
    
    # Load DNABERT model and tokenizer
    MODEL_NAME = "zhihan1996/DNA_bert_6"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    
    k = 21
    ref_kmers = []

    # Step 1: create 100 random kmers
    for i in range(100):
        kmer = create_random_string(k)
        ref_kmers.append(kmer)

    # Convert them to embeddings
    reference_embeddings = np.vstack([encode_string(seq) for seq in ref_kmers])

    print(reference_embeddings.shape)
    print("References converted to embeddings.")
    
    # Create FAISS index
    embedding_dim = reference_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(reference_embeddings)
    faiss.write_index(index, "genome_index.faiss")

    print("FAISS index successfully stored")
    
    num_simulations = 5
    
    distances = {1:[], 2:[], 3:[]}
    for hd in [1,2,3]:
        for i in range(num_simulations):
            kmer = ref_kmers[random.randint(0,len(ref_kmers)-1)]
            mut_kmer = mutate_kmer(kmer, hd)
            results = search_similar_genomes(mut_kmer, top_k=1)
            distances[hd].append(results["distances"][0])
            
    print(distances)
        
            