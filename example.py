# This script imports the sequential_design and vis_sequential_design functions from the boundary module in the src.seq_design package.
# It then calls the sequential_design function with k=10, alpha=0.1, and beta=0.2 to generate a sequential design boundary.
# Finally, it calls the vis_sequential_design function to visualize the boundary.
from src.seq_design.boundary import sequential_design, vis_sequential_design

bound = sequential_design(k=10, alpha=0.1, beta=0.2)

vis_sequential_design(bound)