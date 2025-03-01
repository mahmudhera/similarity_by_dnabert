import numpy as np
import faiss
import random

faiss.omp_set_num_threads(4)


char_to_bits = {
    'A' : np.array([0,0,0,1], dtype=bool),
    'C' : np.array([0,0,1,0], dtype=bool),
    'G' : np.array([0,1,0,0], dtype=bool),
    'T' : np.array([1,0,0,0], dtype=bool)
}

def create_random_string(str_len):
    return ''.join(random.choices(['A','C','G','T'], k=str_len))


def kmers(sequence: str, k: int = 6):
    """Convert DNA sequence into k-mer format."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


def encode_kmer(kmer: str):
    """Convert kmer into bitwise embedding."""
    kmer_list = list(kmer)
    kmer_bits = [char_to_bits[char] for char in kmer_list]
    kmer_bits_flat = np.array(kmer_bits).flatten()
    
    return kmer_bits_flat


# Function to search for similar kmers
def search_similar_kmers(index, query_embedding, top_k):
    """Search for the top_k most similar reference genomes."""
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
    
    k = 22
    ref_kmers = []

    # Step 1: create 100 random kmers
    for i in range(10):
        kmer = create_random_string(k)
        ref_kmers.append(kmer)

    # Convert them to embeddings
    reference_embeddings = np.vstack([encode_kmer(seq) for seq in ref_kmers])
    print(reference_embeddings.shape)
    reference_embeddings = reference_embeddings.astype('uint8')
    reference_embeddings = np.packbits(reference_embeddings, axis=1)
    print(reference_embeddings.shape)
    
 
    # Create FAISS index
    nlist = 4096  # Number of clusters
    embedding_dim = reference_embeddings.shape[1]
    index = faiss.IndexBinaryFlat(k*4)
    index.add(reference_embeddings)

    print("FAISS index successfully stored")
    
    num_simulations = 5
    
    distances = {1:[], 2:[], 3:[]}
    for hd in [1,2,3]:
        for i in range(num_simulations):
            kmer = ref_kmers[random.randint(0,len(ref_kmers)-1)]
            mut_kmer = mutate_kmer(kmer, hd)
            mut_kmer_embedding = np.packbits(encode_kmer(mut_kmer).astype('uint8'), axis=1)
            results = search_similar_kmers(index, mut_kmer_embedding, top_k=2)
            distances[hd].append(results["distances"][0])
            
    print(distances)