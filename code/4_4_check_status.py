import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
client = OpenAI()

batch_id = {
  "file_id": "file-1kVnWo4YX6ZKE9Qs1ZD4tm",
  "batch_id": "batch_690ae971d7048190a982f22185698051",
  "status": "validating"
}

batch = client.batches.retrieve(batch_id["batch_id"])
print(json.dumps(batch.model_dump(), indent=2))