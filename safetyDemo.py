from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os
import json

import labgpt_config as cfg

model_name = cfg.MODEL_NAME


def load_safety_content():
    """Load the safety Q&A corpus and flatten it into one text block.

    This concatenates the whole corpus so it can be injected into a prompt wholesale.
    That is the prompt-stuffing pattern documented in the README, and it is what the
    retrieval work replaces. Kept as-is here so that swapping the data source is a
    separate change from swapping the architecture.
    """
    with open(cfg.require(cfg.SAFETY_PATH), encoding="utf-8") as handle:
        entries = json.load(handle)
    return " ".join(f"{e['instruction']}? {e['output']}" for e in entries)


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
        query = "read this safety text: " + load_safety_content() + "Then answer this question: " + query
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
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    try:
        model.generate(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=3276
        )
    except Exception as e:
        print(f"\nError during model generation: {e}")

if __name__ == "__main__":
    run_chat()



