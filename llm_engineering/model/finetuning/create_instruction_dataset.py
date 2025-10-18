import concurrent.futures
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from datasets import Dataset
import google.generativeai as genai
from tqdm.auto import tqdm
from llm_engineering.settings import settings
from huggingface_hub import HfApi

import time
from threading import Lock
import queue

class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 10):
        self.max_requests = max_requests_per_minute
        self.requests = queue.Queue()
        self.lock = Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
           
            while not self.requests.empty():
                if now - self.requests.queue[0] > 60:
                    self.requests.get()
                else:
                    break
            
            if self.requests.qsize() >= self.max_requests:
                sleep_time = 60 - (now - self.requests.queue[0]) + 1
                time.sleep(sleep_time)
            
            self.requests.put(time.time())

rate_limiter = RateLimiter(15)


def load_articles_from_json(file_path: str) -> Dataset:
    with open(file_path, "r") as file:
        data = json.load(file)
    
    if isinstance(data, list):
        articles = data
    elif isinstance(data, dict) and "artifact_data" in data:
        articles = data["artifact_data"]
    else:
        raise ValueError("Unsupported JSON structure. Expected a list or dict with 'artifact_data' key.")
    
    if articles and isinstance(articles[0], dict) and "payload" in articles[0]:
        sample_keys = articles[0]["payload"].keys()
        print(f"Available keys in JSON payload: {list(sample_keys)}")
        
        dataset_dict = {}
        
        dataset_dict["id"] = [item.get("id", f"item_{i}") for i, item in enumerate(articles)]
        
        dataset_dict["content"] = [item["payload"].get("content", "") for item in articles]
        dataset_dict["platform"] = [item["payload"].get("platform", "unknown") for item in articles]
        dataset_dict["author_id"] = [item["payload"].get("author_id", "unknown") for item in articles]
        dataset_dict["author_full_name"] = [item["payload"].get("author_full_name", "unknown") for item in articles]
        dataset_dict["link"] = [item["payload"].get("link", "") for item in articles]
        
        return Dataset.from_dict(dataset_dict)
    else:
        raise ValueError("Invalid JSON structure: expected list of dictionaries with 'payload' field.")

def clean_text(text):
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_substrings(dataset: Dataset, min_length: int = 1000, max_length: int = 2000) -> List[str]:
    extracts = []
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

    for article in dataset["content"]:
        cleaned_article = clean_text(article)
        sentences = re.split(sentence_pattern, cleaned_article)

        current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())

            current_chunk = sentence + " "

        if len(current_chunk) >= min_length:
            extracts.append(current_chunk.strip())

    return extracts


class InstructionAnswerSet:
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    @classmethod
    def from_json(cls, json_str: str) -> 'InstructionAnswerSet':
        data = json.loads(json_str)
        pairs = [(pair['instruction'], pair['answer'])
            for pair in data['instruction_answer_pairs']]
        
        return cls(pairs)
    
    def __iter__(self):
        return iter(self.pairs)


def generate_instruction_answer_pairs(extract: str, model: genai.GenerativeModel) -> List[Tuple[str, str]]:
    rate_limiter.acquire()

    prompt = f"""Based on the following extract, generate five
    instruction-answer pairs. Each instruction \
    must ask to write about a specific topic contained in the context.
    each answer \
    must provide a relevant paragraph based on the information found in
    the \
    context. Only use concepts from the context to generate the
    instructions. \
    Instructions must never explicitly mention a context, a system, a
    course, or an extract. \
    Instructions must be self-contained and general. \
    Answers must imitate the writing style of the context. \
    Example instruction: Explain the concept of an LLM Twin. \
    Example answer: An LLM Twin is essentially an AI character that
    mimics your writing style, personality, and voice. \
    It's designed to write just like you by incorporating these elements
    into a language model. \
    The idea is to create a digital replica of your writing habits using
    advanced AI techniques. \
    Provide your response in JSON format with the following structure:
    {{
        "instruction_answer_pairs": [
            {{"instruction": "...", "answer": "..."}},
            ...
        ]
    }}

    Extract:
    {extract}
    """

    try:
        response = model.generate_content(prompt)
        
        # Extract JSON from response text
        response_text = response.text
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = response_text
        
        # Parse the structured output
        result = InstructionAnswerSet.from_json(json_str)
        
        # Convert to list of tuples
        return result.pairs
        
    except Exception as e:
        print(f"Error generating instruction-answer pairs: {e}")
        return []

def create_instruction_dataset(dataset: Dataset, client: genai.GenerativeModel, num_workers: int = 2) -> Dataset:
    extracts = extract_substrings(dataset)
    instruction_answer_pairs = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_instruction_answer_pairs, extract, client) for extract in extracts]   

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            instruction_answer_pairs.extend(future.result())

    instructions, answers = zip(*instruction_answer_pairs)

    return Dataset.from_dict(
        {"instruction": list(instructions), "output": list(answers)}
    )


def check_repo_exists(hf_api, repo_id):
    """Helper function to check if repository exists"""
    try:
        hf_api.repo_info(repo_id=repo_id, repo_type="dataset")
        return True
    except Exception:
        return False

def main(dataset_id: str, api_key: str = None) -> Dataset:
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # You can set the API key as an environment variable: GOOGLE_API_KEY
        genai.configure()

    hf_api = HfApi(token=settings.HF_TOKEN)
    repo_id = f"{settings.HF_USERNAME}/llmtwin"


    # Initialize Gemini model
    model = genai.GenerativeModel(settings.GOOGLE_GEMINI_MODEL)  # or 'gemini-1.5-pro' for better quality

    # 1. Load the raw data
    raw_dataset = load_articles_from_json("cleaned_documents.json")
    print("Raw dataset:")
    print(raw_dataset.to_pandas())

    # 2. Create instructiondataset
    instruction_dataset = create_instruction_dataset(raw_dataset, model)
    print("Instruction dataset:")
    print(instruction_dataset.to_pandas())

    # 3. Train/test split and export
    filtered_dataset = instruction_dataset.train_test_split(test_size=0.1)

    repo_exists = check_repo_exists(hf_api, repo_id)
    
    if repo_exists:
        print(f"Repository {repo_id} already exists")
    else:
        print(f"Creating new repository {repo_id}")
        hf_api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )

    filtered_dataset.push_to_hub(
        repo_id,
        token=settings.HF_TOKEN,
        private=False,
        commit_message="Updated instruction dataset with train/test split"
    )

    return filtered_dataset

if __name__ == "__main__":
    main("llmtwin", settings.GOOGLE_API_KEY)