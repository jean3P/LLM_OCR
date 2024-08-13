from transformers import AutoTokenizer

# Load the tokenizer (use the tokenizer suitable for your model)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Sample text with approximately 50 words
sample_text = ("Corrected the Kong to the Stores, Vc. and adjusted the phrase. The confidence level indicates some recognition errors.")

# Tokenize the sample text
tokens = tokenizer.tokenize(sample_text)

# Print the number of tokens
num_tokens = len(tokens)
print(f"Number of tokens: {num_tokens}")

# If the number of tokens is close to 50, print the sample text
if num_tokens <= 50:
    print(f"Sample text with {num_tokens} tokens:\n{sample_text}")
else:
    # If the sample text is more than 50 tokens, find the point where we reach 50 tokens
    truncated_text = tokenizer.convert_tokens_to_string(tokens[:50])
    print(f"Text that corresponds to 50 tokens:\n{truncated_text}")
