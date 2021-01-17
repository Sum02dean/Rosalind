import re
import sys
import os
import itertools
import numpy as np
import pandas as pd
import pydot
from collections import Counter

"""This class is to store all of the code required to complete the entire stack of Rosalind problems http://rosalind.info/problems/list-view/
    Feel free to fork this repository and begin building your own bioinformatics package.
    
    Author: Dean Sumner 
    License: Open source license MIT
    """
class Rosalind:
    def __init__(self):
        self.compliment = {'A': 'T',
                           'C': 'G',
                           'T': 'A',
                           'G': 'C',
                           'U': 'A'}
        self.rna_codex = {
            'A': ['GCU', 'GCC', 'GCA', 'GCG'],
            'C': ['UGU', 'UGC'],
            'D': ['GAU', 'GAC'],
            'E': ['GAA', 'GAG'],
            'F': ['UUU', 'UUC'],
            'G': ['GGU', 'GGC', 'GGA', 'GGG'],
            'H': ['CAU', 'CAC'],
            'I': ['AUU', 'AUC', 'AUA'],
            'K': ['AAA', 'AAG'],
            'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
            'M': ['AUG'],
            'N': ['AAU', 'AAC'],
            'P': ['CCU', 'CCC', 'CCA', 'CCG'],
            'Q': ['CAA', 'CAG'],
            'R': ['AGA', 'AGG', 'CGU', 'CGC', 'CGA', 'CGG'],
            'S': ['AGU', 'AGC', 'UCU', 'UCC', 'UCA', 'UCG'],
            'T': ['ACU', 'ACC', 'ACA', 'ACG'],
            'V': ['GUU', 'GUC', 'GUA', 'GUG'],
            'W': ['UGG'],
            'Y': ['UAU', 'UAC'],
            '': ['UGA', 'UAA', 'UAG']}

    def count_nucleotides(self, seq):
        """Returns the number of nucleotides, listed in alphabetical order.

        :param seq: DNA sequence
        :type seq: str
        :return: number of each nucleotide in the order [A, C, G, T]
        :rtype: list
        """

        # Rosalind expects alphabeticallt sorted outputs
        nucleotides = {'A': [], 'C': [], 'G': [], 'T': []}
        seq = seq.upper()
        for key, _ in nucleotides.items():
            nucleotides.update(
                {key: len([x for x in seq if x == key])})
        counts = list(nucleotides.values())
        return " ".join([str(x) for x in counts])

    def transcribe_dna(self, seq):
        """Transcribes a string of DNA into its RNA equivalent e.g
           'AACTCAG'  --> 'UUGAGUC'.

        :param seq: DNA sequences
        :type seq: str
        :return: RNA conversation of input DNA sequence
        :rtype: str
        """
        seq = seq.upper()
        mrna = seq.replace('T', 'U')
        return mrna

    def get_compliment(self, seq):
        """Computes the reverse compliment of a sequence of DNA

        :param seq: DNA sequence
        :type seq: str
        :return: Reverse strand to the DNA input sequences
        :rtype: str
        """
        seq = seq.upper()
        seq_reversed = seq[::-1]
        seq_compliment = "".join([self.compliment[x]
                                  for x in seq_reversed])

        return seq_compliment

    def rabbits(self, n, k):
        """Wascally Wabbits: Returns the total number of rabbits produced ater n generations,
           when given a reproduction factor of k. 

        :param n: number of generations
        :type n: int
        :param k: reproduction factor
        :type k: int
        :return: Total number of rabbits after n generations
        :rtype: int
        """

        if n == 1 or n == 2:
            return 1

        term1 = self.rabbits(n-1, k)
        term2 = self.rabbits(n-2, k)
        term3 = term1 + (k * term2)
        return term3

    def parse_fasta(self, filename):
        """Parses FASTA files and returns a sequence record. Keys: Roslaind IDs, values: genomic sequence.

        :param filename: name of text file containing FASTA data
        :type filename: str
        :return: Record object where keys are FASTA labels and values are FASTA sequences
        :rtype: dict
        """
        with open(filename, 'r') as f:
            # Remove newline characters
            txt = f.read().replace('\n', '')
            txt = re.split(">", txt)
            # Extract Rosalind sequences
            seqs = [re.findall(r'Rosalind_\d\d\d\d(.*)', x)
                    for x in txt if x != '']

            # Extract Rosalind ids
            ids = [re.match(r'(.*)Rosalind_\d\d\d\d', x)
                   for x in txt if x != '']
            id_matches = [ids[x][0] for x in range(0, len(ids))]
            # Compile as a record (dict)
            record = dict(zip(id_matches, [x[0] for x in seqs]))
        return record

    def __compute_gc_contents(self, seq):
        """Computes the GC content of a single nucleotide sequence.

        :param seq: sequence to compute GC score on
        :type seq: str
        :return: gc percentage of sequence
        :rtype: int
        """
        seq = "".join(seq)
        desired_nucleotides = ['G', 'C']
        gcs = [x for x in seq if x in desired_nucleotides]
        gc_content = (len(gcs)/len(seq)) * 100
        return gc_content

    def get_gc_contents(self, record):
        """Reutrns the GC score of all sequences in a fasta record object.

        :param record: keys are sequence labels, values are the respective sequences.
        :type record: dict
        :raises TypeError: if type(record) not dictionary   
        :return: collection containing {labels:GC-percentage} as
                                well as top the label and score with highest the GC percentage.
        :rtype: dict
        """

        # Check the type of the input variable
        if isinstance(record, dict) == False:
            raise TypeError(
                "input variable 'record' should be a dict object but got {} instead".format(type(record)))

        gc_scores = {}
        for key, value in record.items():
            # Compute GC content
            gc_scores.update({key: self.__compute_gc_contents(value)})

        # Sort scores based on GC content (reverse ordered)
        max_gc_scores = {k: v for k, v in sorted(
            gc_scores.items(), key=lambda item: item[1])}

        # Return the highest score and its associated label
        max_label, max_score = list(
            max_gc_scores.keys())[-1], list(max_gc_scores.values())[-1]

        # Print to console
        print(max_label)
        print(max_score)
        return gc_scores, [max_label, max_score]

    def compute_punnet(self, alleles_1, alleles_2, get_dominance_ratio=False):
        """Returns the set of genotype ratios for the new generation along with a punnet square of the parental cross.

        :param alleles_1: parent 1 allele set
        :type alleles_1: str
        :param alleles_2: parent 2 allele set
        :type alleles_2: str
        :return: genotype ratio in the new generation, punnet crosses
        :rtype: dict, list
        """
        # Options should be  ['H','H'], ['H','h'], or ['h','h']
        if isinstance(alleles_1, list) == False:
            alleles_1 = [alleles_1]

        if isinstance(alleles_2, list) == False:
            alleles_2 = [alleles_2]

        # Check inputs
        acceptable_alleles = [['H', 'H'], ['H', 'h'], ['h', 'h']]
        if alleles_1 not in acceptable_alleles:
            return ValueError("ValueError: Alleles should be one of the forms: ['H','H'], ['H','h'], or ['h','h']. Got {}".format(alleles_1))

        if alleles_2 not in acceptable_alleles:
            return ValueError("ValueError: Alleles should be one of the forms: ['H','H'], ['H','h'], or ['h','h']. Got {}".format(alleles_2))

        # Create punnet cross
        punnet = []
        for alleles in itertools.product(alleles_1, alleles_2):
            punnet += [alleles[0] + alleles[1]]

        # Return genotype ratios after punnet cross
        genotypes = {}
        for geneset in ['HH', 'Hh', 'hH', 'hh']:
            genotypes.update(
                {geneset: len([x for x in punnet if x == geneset]) / len(punnet)})

        # Compute the percentage of domant genotypes in the cross
        dominance = [v for k, v in genotypes.items() if 'H' in k]
        if get_dominance_ratio:
            return genotypes, punnet, sum(dominance)
        else:
            return genotypes, punnet, None

    def count_point_mutations(self, seq1, seq2):
        """Counts the number of point mutations when comparing two sequences of the same length.

        :param seq1: first sequence to compare
        :type seq1: string
        :param seq2: second sequence to compare
        :type seq2: string
        :return: the number of mismatched nucleotides
        :rtype: int
        """
        length_seq = len(seq1)
        counter = 0
        for i in range(0, length_seq):
            if seq1[i] != seq2[i]:
                counter += 1
        return counter

    def get_genepool_stats(self, HH=2, Hh=2, hh=2, genepool=None):
        """Computes the statistics of genotypes in a genepool.

        :param HH: number of homozyhous dominant individuals, defaults to 2
        :type HH: int, optional
        :param Hh: number of heterozygous individuals, defaults to 2
        :type Hh: int, optional
        :param hh: number of homozygous recessive indivoduals, defaults to 2
        :type hh: int, optional
        :return: percentage of each genotype within a genepool
        :rtype: dict
        """
        # Compute genepool statistics
        if genepool == None:
            genepool = ['HH'] * HH + ['Hh'] * Hh + ['hh'] * hh

        stats = {}
        for geneset in ['HH', 'Hh', 'hh']:
            stats.update(
                {geneset: len([x for x in genepool if x == geneset]) / len(genepool)})
        return stats, genepool

    def get_mendelian_probas(self, HH=2, Hh=2, hh=2):
        """Computes the mendelian probability of an individual inheriting a domninat allele, 
           assuming any two randomly chosen individual can mate from a know genegool. 

        :param HH: number of homozygous individuals, defaults to 2
        :type HH: int, optional
        :param Hh: number of heterozygous individuals, defaults to 2
        :type Hh: int, optional
        :param hh: number of homozygous recessive individuals, defaults to 2
        :type hh: int, optional
        :return: the probability of an individual containing dominant allele after random parental mating.
        :rtype: float
        """
        # Initialize empty probability
        probas = []

        # Compute possible permutations of the parental genotypes
        parent_genotypes = ['HH', 'Hh', 'hh']
        combinations = list(
            itertools.combinations_with_replacement(parent_genotypes, 2))

        for combination in combinations:
            # Collect each parents genotype and split on alleles
            parent_a = [x for x in combination[0]]
            parent_b = [x for x in combination[1]]

            # Collect genepool statistics for selection 1
            genepool_stats, gp = self.get_genepool_stats(
                HH, Hh, hh)

            # Drop current selection from genepool
            gp.remove(combination[0])

            # Collect genepool statistics for selection 2 (without replacment)
            genepool_stats_2, _ = self.get_genepool_stats(genepool=gp)

            # Compute punnet square statistics, determine dominance factor
            _, _, dominance_ratio = self.compute_punnet(
                parent_a, parent_b, True)

            # Account for sequence variant selection and final probabilities for dominant offspring
            if len(np.unique(combination)) > 1:
                # (ratio_allele_selection_1/n_pop X ratio_allele_selection_1/n_pop-1 * 2) * dominance_ratio
                proba = (genepool_stats[combination[0]] *
                         genepool_stats_2[combination[1]] * 2) * dominance_ratio
            else:
                # Same for invariant selection
                proba = (genepool_stats[combination[0]] *
                         genepool_stats_2[combination[1]] * 1) * dominance_ratio

            # update probability list
            probas.append(proba)
        # Sum all of the conditional probabilities
        sum_proba = sum(probas)
        return sum_proba

    def __codon_generator(self, seq, n=3):
        """Generator object which yields n-sized chunked subsequences from seq 
           e.g. 'AAACCCGGG' --> ['AAA','CCC,'GGG].

        :param seq: nucleotide sequences    
        :type seq: str  
        :param n: chunking size, defaults to 3
        :type n: int, optional
        :yield: list of n-sized substring chunks
        :rtype: list
        """
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    def seq_to_codons(self, seq):
        """Calls codon_generator()."""
        return list(self.__codon_generator(seq))

    def transalte_mrna(self, seq):
        """Parses rna-seq string and returns the translated protein sequence.

        :param seq: rna sequence
        :type seq: str
        :return: amino-acid/protein sequence
        :rtype: str
        """
        amino_acids = []
        # Stagger the rna code by 3 units
        for codon in (self.seq_to_codons(seq)):
            # Query codon-protein lookup dictionary
            for key, values in self.rna_codex.items():
                if codon in values:
                    amino_acids.append(key)
        # Merge list to char array
        amino_acid_str = "".join(amino_acids)
        return amino_acid_str

    def find_motif(self, seq, subseq, strip_comma=True):
        """Return the seq indices of the first character in the matched subseq.

        :param seq: dna sequence
        :type seq: str
        :param subseq: dna sub-sequence 
        :type subseq: str
        :return: list of indicies
        :rtype: lsit
        """
        idxs = [m.start() + 1 for m in re.finditer(
            '(?={0})'.format(re.escape(subseq)), seq)]

        # Format as expected for Roslaind challenge
        if strip_comma:
            idxs = ' '.join([str(x) for x in idxs])
            return idxs
        else:
            return idxs

    def convert_fasta_to_pandas(self, record):
        """Converts a fasta record object to pandas DataFrame. 
           Columns represent nucleotides, rows individual sequences.     

        :param record: fasta record object
        :type record: dict
        :return: Formatted DataFrame    
        :rtype: pandas DataFrame
        """
        seqs = list(record.values())

        for i, seq in enumerate(seqs):
            char_seq = [x[0] for x in seq]
            df_seq = pd.DataFrame(char_seq).T

            if i == 0:
                df = df_seq
            else:
                df = pd.concat([df, df_seq])

        df.index = list(record.keys())
        return df

    def get_consensus(self, record, print_report=True):
        """Computes the consensus sequence and consensus matrix of a fasta record.

        :param record: fasta record
        :type record: dict
        :param print_report: set to true to print out Rosalind format, defaults to True
        :type print_report: bool, optional
        :return: consnsus sequence, consensus matrix
        :rtype: str, np.array
        """
        # Convert fasta to pandas dataframe
        df = self.convert_fasta_to_pandas(record)
        x = np.array(df)
        print(x.shape)

        # Generate naive consensus matrix
        z = np.zeros((4, np.shape(x)[1]))
        nucleotides = ['A', 'C', 'G', 'T']

        # Populate consensus matrix
        for row in range(0, len(nucleotides)):
            nucleotide = nucleotides[row]

            # Count the occurance of nucleotides across columns
            for col in range(0, np.shape(z)[1]):
                col_nucleotides = "".join(x[:, col])
                num_nucleotide = col_nucleotides.count(nucleotides[row])
                z[row, col] = num_nucleotide

        # Print out rosalind-format report
        cmat = {'A': [], 'T': [], 'C': [], 'G': []}
        for i, nucleotide in enumerate(nucleotides):
            cmat[nucleotide] = [x for x in z[i, :]]
            counts = [str(int(x)) for x in z[i, :]]
            print('{}: {}'.format(nucleotide, " ".join(counts)))

        # Get consensus sequence
        df2 = pd.DataFrame(cmat).T
        cseq = []
        for i in range(0, df2.shape[1]):
            cseq.append(df2[i].idxmax())
        cseq = "".join(cseq)
        return cmat, cseq


# Instantiate object
R = Rosalind()

# Call method
n = R.count_nucleotides("ACCGGGTTTT")
print(n)