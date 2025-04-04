# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here
class Algo1_Sampler:
    
class Algo2_Sampler:
    
# --- Main Execution ---
if __name__ == "__main__":
    
    