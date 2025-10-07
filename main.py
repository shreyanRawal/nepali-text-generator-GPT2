from transformers import AutoTokenizer, AutoModelForCausalLM

# Loading a model that supports nepali language, here we are using Helsinki-NLP/opus-mt-en-ne
model_name = "Sakonii/distilgpt2-nepali"

tokenizer = AutoTokenizer.from_pretrained(model_name) # getting auto tokenizer for this model
model =AutoModelForCausalLM.from_pretrained(model_name) # getting auto model to perform seq to seq for this model

#input prompt
prompt = "नेपालका धेरैजसो चाडपर्वहरूमध्ये,"



#Tokenizing
inputs = tokenizer(prompt, return_tensors = "pt")


outputs= model.generate(**inputs,max_length=50,       # limit length of generated text
    do_sample=True,      # enable randomness
    top_k=50,            # sampling parameter
    top_p=0.95,          # nucleus sampling
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Chatbot: ", response)


