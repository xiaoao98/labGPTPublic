from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os

model_name = "Qwen/Qwen3-32B"
classifyContent = "You are a lab assistant AI, respond with the category name (safety, member information, protocol for experiment) for all the information you need to answer this question. If it contain several categories, respond them all split by comma. If it does not need all of them, respond with general. **Question**: "

def run_chat() -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # chat_model = ChatModel()
    messages = []
    print("Welcome to chat with Metis!")
    # print("Cryorecovery, how should I do this experiment?")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break
        # if query.strip() == "clear":
        #     messages = []
        #     print("History has been removed.")
        #     continue

        # messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)
        
        # for new_text in chat_model.stream_chat(messages):
        #     print(new_text, end="", flush=True)
        #     response += new_text
        # print("database_response: " + database_response)
        query = classifyContent + query
        print_model_response(tokenizer, model, query)        
        # print(chat_output)
        # messages.append({"role": "assistant", "content": response})

# load the tokenizer and the model
def print_model_response(tokenizer, model, input_content):
    messages = [
        {"role": "user", "content": input_content}
    ]
    # question = "I have three types of question: memberinfo, safety, biology experiment protocol. Please answer me the type of the following question (you can only choose one):  What does 2 Gallon sharps containers accepts. The answer should only be the type, no more explanation"

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    try:
        generated_ids = model.generate(
            **model_inputs,
            # streamer=streamer,
            max_new_tokens=3276
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print(content)
    except Exception as e:
        print(f"\nError during model generation: {e}")

if __name__ == "__main__":
    run_chat()



