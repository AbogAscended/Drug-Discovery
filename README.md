# SMILES Molecule Generation

A research project exploring deep generative models to generate valid and novel molecules represented as SMILES (Simplified Molecular Input Line Entry System) strings.

## Goal
Explore how generative models can be used to generate valid, novel, and diverse chemical structures for potential applications in drug discovery and material science.

## Dataset
We used the [MOSES dataset](https://github.com/molecularsets/moses), which contains 1.6 million training and 176k testing molecules. The data is preprocessed into tokenized SMILES strings with chemical constraints.

## Models Implemented
We implemented and compared three types of generative models:

- **CharRNN**: Character-level GRU models, including a Flow-based variant with latent space regularization and added entropy-based penalties.
- **GAN**: A GAN framework operating in latent space, built on an autoencoder architecture to better learn SMILES representations.
- **VAE**: A variational autoencoder trained to reconstruct SMILES and sample new ones, enhanced with KL annealing, repetition penalties, and output sanitization.

## Evaluation Metrics
We evaluated models using:
- **Validity**: Chemically valid SMILES strings.
- **Uniqueness**: Percentage of unique valid outputs.
- **Novelty**: Percentage not seen during training.
- **Distinctness**: Diversity among generated outputs.

## Results Snapshot
- **CharRNN**: Validity up to 99.97%, but low uniqueness.
- **GAN**: Validity ~16%, high uniqueness, moderate novelty.
- **VAE**: 100% valid and novel outputs; ~20% distinctness.

## Key Takeaways
- VAE produced the most chemically valid outputs, though many were simplistic or unrealistic.
- Additional penalties (e.g., for repetitions, invalid characters) helped improve quality.
- All models showed trade-offs between validity, diversity, and novelty.

## Team
Jacob Levine, Abigail Pinkus, Alex Kazmirchuk, Ben Tran  
Cal Poly Pomona – Departments of Mathematics, Computer Science, and Biological Science
