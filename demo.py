from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os
from protocolDemo import read_protocol_by_experiment, read_reagent_by_experiment
from safetyDemo import load_safety_content
from memberInfoDemo import load_member_info
from search.getDynamicPage import fetch_all_webpage_content
from search.duckducksearch import get_useful_link
from paperDemo import fetch_paper_data
from rich import print as rprint
import json
import labgpt_config as cfg
import labgpt_metrics as metrics

model_name = cfg.MODEL_NAME

# Generation caps. Every one of these was 32768 or 65536, for answers measured in words.
# The cap is what the caller can actually use, plus headroom.
MAX_NEW_TOKENS_YESNO = 8      # "yes" / "no"
MAX_NEW_TOKENS_CLASSIFY = 32  # "safety, experiment reagent and protocol" is ~12 tokens
MAX_NEW_TOKENS_IDS = 2048     # thinking budget plus "1, 5, 12, 22, 30"
MAX_NEW_TOKENS_ANSWER = 4096  # the user-facing answer
ifSearchContent = "Search is NEEDED (true) for: 1. Real-time information: Weather, stock prices, traffic, sports scores. 2. Recent events: Anything that happened after your late 2023 knowledge cutoff. 3. Specific people or entities: Questions about non-famous individuals or specific company news. 4. Niche or local information: Store hours, specific product recommendations, local regulations." + '\n'
ifSearchContent += "Does this query need to do the web search (Answer can only be one word: yes or no):  "
classifyContent = "You are a lab assistant AI, respond with the category name (safety, experiment reagent and protocol) for all the information you need to answer this question. If it contain several categories, respond them all split by comma. If it does not need any of them, respond with general. **Question**: "
ifSearchPapersContent = """
    You are a query classifier for a specialized biology and cancer research paper database.
    
    Your task is to determine if the user's input requires retrieving specific scientific literature or data.

    Criteria to output "yes":
    - The user asks about specific cancer types or genes (e.g., TP53, KRAS), or drug mechanisms.
    - The user asks for citations, papers, or recent studies.
    - The user asks for statistical data or experimental results.

    Criteria to output "no":
    - The user asks for coding help, creative writing, or general conversation.
    - The user asks a question unrelated to biology or cancer.

    IMPORTANT: You must respond with ONLY the word "yes" or "no". Do not provide punctuation or explanations.
    """
ifSearchPapersContent += "Does this query need to retrieve specific scientific literature or data from the biology and cancer research paper database (Answer can only be one word: yes or no):  "
database_path = cfg.DATABASE_PATH
paper_json_path = cfg.PAPER_CONTENT_PATH


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
    rprint("[green]Welcome to chat with " + cfg.ASSISTANT_NAME + "![/green]")
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
        rprint("[green]" + cfg.ASSISTANT_NAME + ": [/green]", end="", flush=True)
        
        # for new_text in chat_model.stream_chat(messages):
        #     print(new_text, end="", flush=True)
        #     response += new_text
        # print("database_response: " + database_response)
        ifSearchQuery = ifSearchContent + query
        ifSearch = getAnswer(tokenizer, model, ifSearchQuery,
                             MAX_NEW_TOKENS_YESNO, label="classify:web").lower()
        # print(ifSearch)
        ifSearchPaperQuery = ifSearchPapersContent + query
        ifSearchPaper = getAnswer(tokenizer, model, ifSearchPaperQuery,
                                  MAX_NEW_TOKENS_YESNO, label="classify:paper").lower()
        # print("here " + ifSearchPaper)
        classifyQuery = classifyContent + query
        dataClassArray = getAnswer(tokenizer, model, classifyQuery,
                                   MAX_NEW_TOKENS_CLASSIFY, label="classify:domain").split(",")
        # print(dataClassArray)
        finalQuery = "You are a lab assistant AI." + '\n'
        if "safety" in dataClassArray:
            rprint('\n' + "[blue]Searching safety measurement database: [/blue]")
            finalQuery += "Read this safety document: " + load_safety_content() + '\n'
            rprint('\n' + "Done")
        if "experiment reagent and protocol" in dataClassArray:
            rprint("[blue]Begin searching protocol database: [/blue]")
            database_reagent_response = read_reagent_by_experiment(database_path, query, tokenizer, model)
            database_protocol_response = read_protocol_by_experiment(database_path, query, tokenizer, model)
            if database_reagent_response != "not found":
                finalQuery += "Read those reagents documents: " + database_reagent_response + '\n' 
            if database_protocol_response != "not found":   
                finalQuery += "Read those protocol documents: " + database_protocol_response + '\n'
        if "yes" in ifSearch and "experiment reagent and protocol" not in dataClassArray:
            print()
            rprint("[blue]Begin searching internet: [/blue]")
            searchQuery = "There is the information fetched from the internet to answer the question: " + query
            searchQuery += "Summarize those information fetched from the internet (Try to be precise and contain useful link): "
            links, snippets = get_useful_link(query)
            all_page_contents = fetch_all_webpage_content(links)
            for i, item in enumerate(all_page_contents):
                    searchQuery += f"\n--- Content from Link {i+1} ---"
                    searchQuery += f"URL: {item['url']}"
                    searchQuery += f"Snippet: {snippets[i]}"
                    searchQuery += f"Content: {item['content']}" 
                    searchQuery += "-" * 30
            rprint("[blue]Summary of the search: [/blue]")
            searchAnswer = stream_and_return_response(tokenizer, model, searchQuery, False) 
            # print("#########################" + searchAnswer)
            finalQuery += "Read those information from the website: " + searchAnswer + '\n'
        if "yes" in ifSearchPaper:
            rprint("[blue]Begin to search the paper from our lab to fetch related paper: [/blue]")
            IDs = getIDs(tokenizer, model, query)
            print(IDs)
            titles, details = fetch_paper_data(paper_json_path, IDs)
            print("Related Papers from our lab: "+ '\n')
            print(titles)
            # print(IDs)
            finalQuery += "Read method and results part from those papers: " + details + '\n'
        # print(finalQuery)
        finalQuery += "Read this document for our lab members to see if someone can help: " + load_member_info() + '\n'
        finalQuery += "Now answer this question (if possible always tell who in our lab can help): " + query
        rprint("[blue]Begin to think before answering the question: [/blue]")
        print_model_response(tokenizer, model, finalQuery, True)  
        


def getAnswer(tokenizer, model, input_content, max_new_tokens=MAX_NEW_TOKENS_YESNO,
              label="classify") -> str:
    """Short, deterministic classification calls.

    `max_new_tokens` used to be 32768 here, for answers that are a single word. Three of
    these run before any answering begins, so the budget was 98K tokens of generation to
    decide routing. The cap now matches what the caller can actually receive; see the
    MAX_NEW_TOKENS_* constants.

    Sampling is off. Routing decisions should be reproducible, otherwise the same question
    can take different paths on different runs and the evaluation set means nothing.
    """
    messages = [
        {"role": "user", "content": input_content}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    try:
        with metrics.timer() as elapsed:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        metrics.record(
            label,
            prompt_tokens=model_inputs.input_ids.shape[1],
            generated_tokens=len(output_ids),
            elapsed_s=elapsed[0],
            cap=max_new_tokens,
        )
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
    except Exception as e:
        print(f"\nError during model generation: {e}")
        return ""

def getIDs(tokenizer, model, query):
    json_file_path = cfg.PAPERS_PATH
    # A missing or malformed corpus file is not recoverable here. The previous version
    # printed and returned None, which then surfaced as an AttributeError inside
    # fetch_paper_data three frames later. Fail where the problem is.
    with open(cfg.require(json_file_path), 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Convert the JSON object to a string so the model can read it
    paperInfoText = json.dumps(data, indent=2)
    full_instructional_prompt = (
            "### DATASET\n"
            f"{paperInfoText}\n\n"
            "### USER QUERY\n"
            f"{query}\n\n"
            "### TASK\n"
            "Identify the top 5 relevant papers. Provide ONLY their numerical IDs as a comma-separated list.\n"
            "STRICT RESTRICTIONS:\n"
            "- No titles.\n"
            "- No introductory text or conversational filler.\n"
            "- Output ONLY the numbers (e.g., 1, 5, 12, 22, 30).\n\n"
            "### RELEVANT IDs:\n"
        )
    IDs = get_answers_only_for_thinking(tokenizer, model, full_instructional_prompt)
    return IDs

def get_answers_only_for_thinking(tokenizer, model, input_content,
                                  max_new_tokens=MAX_NEW_TOKENS_IDS,
                                  label="paper_ids"):
    """Generate with thinking enabled and return only the post-thinking answer.

    Two changes from the original. The cap was 65536 to produce a comma-separated list of
    five integers; it is now a budget that covers the reasoning plus the answer.

    More importantly, this used to return None implicitly whenever "</think>" was absent
    from the response, which happens whenever the token budget runs out mid-thought. The
    caller then passed None into fetch_paper_data, which called None.split(',') and
    crashed. Now a truncated response falls back to the raw text and says so.
    """
    messages = [
        {"role": "user", "content": input_content}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    try:
        with metrics.timer() as elapsed:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        metrics.record(
            label,
            prompt_tokens=model_inputs.input_ids.shape[1],
            generated_tokens=len(new_tokens),
            elapsed_s=elapsed[0],
            cap=max_new_tokens,
            truncated="</think>" not in response_text,
        )

        if "</think>" in response_text:
            return response_text.split("</think>")[-1].strip()

        # Budget ran out before the model finished thinking. Return what there is rather
        # than None, so the caller fails on bad content instead of on a NoneType.
        print(
            f"\nWarning: generation hit the {max_new_tokens} token cap before finishing. "
            "Raise max_new_tokens if this recurs."
        )
        return response_text.strip()
    except Exception as e:
        print(f"\nError during model generation: {e}")

# load the tokenizer and the model
def print_model_response(tokenizer, model, input_content, ifThinking):
    messages = [
        {"role": "user", "content": input_content}
    ]
    # question = "I have three types of question: memberinfo, safety, biology experiment protocol. Please answer me the type of the following question (you can only choose one):  What does 2 Gallon sharps containers accepts. The answer should only be the type, no more explanation"

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ifThinking # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    try:
        with metrics.timer() as elapsed:
            generated_ids = model.generate(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=MAX_NEW_TOKENS_ANSWER
            )
        # This is the call that carries the assembled prompt, so its prompt_tokens is the
        # number the retrieval work is trying to bring down.
        metrics.record(
            "answer",
            prompt_tokens=model_inputs.input_ids.shape[1],
            generated_tokens=len(generated_ids[0]) - model_inputs.input_ids.shape[1],
            elapsed_s=elapsed[0],
            cap=MAX_NEW_TOKENS_ANSWER,
            thinking=bool(ifThinking),
        )
    except Exception as e:
        print(f"\nError during model generation: {e}")

class TextCaptureStreamer(TextStreamer):
    """A custom streamer that captures the generated text while also printing it."""
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.captured_text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Call the parent's method to print the text to the console
        super().on_finalized_text(text, stream_end)
        # Append the text to our internal capture
        self.captured_text += text

def stream_and_return_response(tokenizer, model, input_content, ifThinking):
    """
    Generates a model response, streams it to the console, and returns the full text.
    """
    messages = [
        {"role": "user", "content": input_content}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ifThinking
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Use our custom streamer
    streamer = TextCaptureStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    try:
        with metrics.timer() as elapsed:
            generated_ids = model.generate(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=MAX_NEW_TOKENS_ANSWER
            )
        metrics.record(
            "web_summary",
            prompt_tokens=model_inputs.input_ids.shape[1],
            generated_tokens=len(generated_ids[0]) - model_inputs.input_ids.shape[1],
            elapsed_s=elapsed[0],
            cap=MAX_NEW_TOKENS_ANSWER,
        )
    except Exception as e:
        print(f"\nError during model generation: {e}")
        return None

    # Return the text captured by the streamer
    return streamer.captured_text

# Example usage:
# input_prompt = "What are the main benefits of using Python for data science?"
# response = stream_and_return_response(tokenizer, model, input_prompt, ifThinking=False)
# print("\n--- Function returned: ---")
# print(response)

if __name__ == "__main__":
    run_chat()



