import concurrent.futures
import json
import re
from typing import List, Tuple
from datasets import Dataset
from openai import OpenAI
from tqdm.auto import tqdm
import google.generativeai as genai
from llm_engineering.settings import settings
from huggingface_hub import HfApi
import time
from threading import Lock
import queue

class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 15):
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

rate_limiter = RateLimiter(10)

class PreferenceSet:
    def __init__(self, triples: List[Tuple[str, str, str]]):
        self.triples = triples

    @classmethod
    def from_json(cls, json_str: str) -> 'PreferenceSet':
        data = json.loads(json_str)
        triples = [(triple['instruction'], triple['generated_answer'], triple['extracted_answer'])
        for triple in data['preference_triples']]
        return cls(triples)
    
    def __iter__(self):
        return iter(self.triples)
    
def load_articles_from_json(file_path: str) -> Dataset:
    with open(file_path, "r") as file:
        data = json.load(file)
        data = data[:50]
    
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

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_substrings(dataset: Dataset, min_length: int = 10, max_length: int = 200) -> List[str]:
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

def generate_preference_triples(extract: str, model: genai.GenerativeModel) -> List[Tuple[str, str, str]]:
    # Apply rate limiting
    rate_limiter.acquire()

    prompt = f"""Based on the following extract, generate five
instruction-answer triples. Each triple should consist of:
1. An instruction asking about a specific topic in the context.
2. A generated answer that attempts to answer the instruction based
on the context.
3. An extracted answer that is a relevant excerpt directly from the
given context.

Instructions must be self-contained and general, without explicitly
mentioning a context, system, course, or extract.

Important:
- Ensure that the extracted answer is a verbatim copy from the
context, including all punctuation and apostrophes.
- Do not add any ellipsis (...) or [...]
in the extracted answer.
- If the relevant text is not continuous, use two separate sentences
from the context instead of skipping text.

Provide your response in JSON format with the following structure:
{{
"preference_triples": [
{{
"instruction": "...",
"generated_answer": "...",
"extracted_answer": "..."
}},
...
]
}}

Extract:
{extract}
"""

    try:
        # Generate content using Gemini
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                candidate_count=1,
                max_output_tokens=2000,
            )
        )
        
        # Extract JSON from response text
        response_text = response.text
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = response_text
        
        # Parse the structured output
        result = PreferenceSet.from_json(json_str)
        
        # Convert to list of tuples
        return result.triples
        
    except Exception as e:
        print(f"Error generating preference triples: {str(e)}")
        return []

def filter_short_answers(dataset: Dataset, min_length: int = 10) -> Dataset:
    def is_long_enough(example):
        return len(example['chosen']) >= min_length
    return dataset.filter(is_long_enough)

def filter_answer_format(dataset: Dataset) -> Dataset:
    def is_valid_format(example):
        chosen = example['chosen']
        return (len(chosen) > 0 and chosen[0].isupper() and chosen[-1] in ('.', '!', '?'))
    
    return dataset.filter(is_valid_format)

def create_preference_dataset(dataset: Dataset, model: genai.GenerativeModel, num_workers: int = 4) -> Dataset:
    extracts = extract_substrings(dataset)
    preference_triples = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_preference_triples, extract, model)
        for extract in extracts
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            preference_triples.extend(future.result())

    instructions, generated_answers, extracted_answers = zip(*preference_triples)
    
    return Dataset.from_dict(
        {
            "prompt": list(instructions),
            "rejected": list(generated_answers),
            "chosen": list(extracted_answers)
        }
    )

def main(dataset_id: str, api_key: str = None) -> Dataset:
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # You can set the API key as an environment variable: GOOGLE_API_KEY
        genai.configure()

    hf_api = HfApi(token=settings.HF_TOKEN)
    repo_id = f"{settings.HF_USERNAME}/llmtwin"


    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # 1. Load the raw data
    raw_dataset = load_articles_from_json("cleaned_documents.json")
    print("Raw dataset:")
    print(raw_dataset.to_pandas())

    # 2. Create preference dataset
    dataset = create_preference_dataset(raw_dataset, model)
    print("Preference dataset:")
    print(dataset.to_pandas())

    # 3. Filter out samples with short answers
    dataset = filter_short_answers(dataset)

    # 4. Filter answers based on format
    dataset = filter_answer_format(dataset)

    # 5. Export
    dataset.push_to_hub(dataset_id, token = settings.HF_TOKEN, private=False)
    return dataset

if __name__ == "__main__":
    main("llmtwin-preference", settings.GOOGLE_API_KEY)