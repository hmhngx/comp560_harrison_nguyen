##############################################
# Configuration file for onboarding experiment
##############################################

# Output directory for results
out_dir = "out"
# Input data directory
data_dir = "data"

# Random seed for reproducibility
seed = 54321

# Number of samples per batch during training
batch_size = 64
# Number of training epochs (complete passes through the dataset)
epochs = 30
# Dimension of embedding vectors (an embedding vector is a numerical representation of what the model is currently "thinking")
embedding_dim = 128
# Number of attention heads in transformer
n_heads = 4
# Number of transformer layers
n_layers = 4