import os
from reagentsSearch import read_reagents_by_experiment

database_path = 'reagents.db'  

def run_chat() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # chat_model = ChatModel()
    messages = []
    print("Welcome to the Reagent search application, please input the experiment name, use `exit` to exit the application.")

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

        response = read_reagents_by_experiment(database_path, query)
        # for new_text in chat_model.stream_chat(messages):
        #     print(new_text, end="", flush=True)
        #     response += new_text
        print(response)
        print()
        # messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    run_chat()
