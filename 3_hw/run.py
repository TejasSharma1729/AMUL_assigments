import subprocess

# Define the different decoding strategies and their parameters
decoding_strategies = [
    # {"strategy": "greedy", "args": ""},
    {"strategy": "random", "args": "--tau 0.5"},
    {"strategy": "random", "args": "--tau 0.9"},
    {"strategy": "topk", "args": "--k 5"},
    {"strategy": "topk", "args": "--k 10"},
    {"strategy": "nucleus", "args": "--p 0.5"},
    {"strategy": "nucleus", "args": "--p 0.9"},
]

output_file = "out.txt"

token = "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"

with open(output_file, "w") as f:
    f.write("Decoding Experiments Output:\n\n")

print(f"{token}")

for strategy in decoding_strategies:
    command = (
        f"CUDA_VISIBLE_DEVICES=3 python task0.py --hf-token \"{token}\""
        f" --decoding-strategy \"{strategy['strategy']}\" {strategy['args']}"
    )
    
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Append results to the output file
    with open(output_file, "a") as f:
        f.write(f"Command: {command}\n")
        f.write("Output:\n")
        f.write(process.stdout + "\n")
        f.write("Error (if any):\n")
        f.write(process.stderr + "\n")
        f.write("-" * 80 + "\n")

print(f"Decoding experiments completed. Results saved in {output_file}")
