from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt, model_name="gpt2", max_length=50, temperature=0.7):
    """
    Generates text using a pre-trained GPT model.

    Args:
        prompt (str): Input text to start generation.
        model_name (str): Name of the pre-trained model (default is GPT-2).
        max_length (int): Maximum length of the generated text.
        temperature (float): Randomness in text generation.

    Returns:
        str: Generated text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], 
                             max_length=max_length, 
                             temperature=temperature, 
                             num_return_sequences=1)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print("Welcome to TextMagic!")
    user_prompt = input("Enter a prompt: ")
    generated_text = generate_text(user_prompt)
    print("\nGenerated Text:")
    print(generated_text)
