from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sarvamai/sarvam-m" #~40 GB

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# prepare the model input
prompt = "Tell me about Fractal AI India and their latest model?"

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=8192)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
output_text = tokenizer.decode(output_ids)

if "</think>" in output_text:
    reasoning_content = output_text.split("</think>")[0].rstrip("\n")
    content = output_text.split("</think>")[-1].lstrip("\n").rstrip("</s>")
else:
    reasoning_content = ""
    content = output_text.rstrip("</s>")

print("reasoning content:", reasoning_content)
print("content:", content)




# sarvam AI focuses on building AI for India, with an emphasis on Indian languages and voice-first applications. They have released:
# Sarvam-1: A 2-billion parameter LLM optimized for 10 Indian languages.
# Sarvam-M: A 24-billion parameter hybrid model built on Mistral Small, also focused on Indian languages.

# Fractal AI is a global AI company that helps enterprises make better decisions. They have:
# Fathom-R1-14B: A 14-billion parameter reasoning model derived from Deepseek-R1-Distilled-14B. They also propose building larger models (up to 70 billion parameters) under the IndiaAI Mission