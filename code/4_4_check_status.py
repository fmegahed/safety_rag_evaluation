import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
client = OpenAI()

BATCH_ID = os.environ.get("OPENAI_BATCH_ID") 

batch = client.batches.retrieve(BATCH_ID)
print(json.dumps(batch.model_dump(), indent=2))