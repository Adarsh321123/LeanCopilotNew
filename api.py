from fastapi import FastAPI
from pydantic import BaseModel
import requests
import uvicorn

app = FastAPI()

HUGGINGFACE_API_URL = "https://huggingface.co/api/models"
USER = "AK123321"

class Request(BaseModel):
    url: str

class Response(BaseModel):
    output: str

class ModelInfo(BaseModel):
    completed: bool
    message: str
    ct2_model_name: str
    ct2_url: str
    emb_name: str
    emb_url: str
    last_modified: str

unique_urls = set()
# TODO: remove after testing
url1 = "https://github.com/leanprover-community/mathlib4.git"
url2 = "https://github.com/teorth/pfr.git"
unique_urls.add(url1)
unique_urls.add(url2)

@app.post("/train/")
async def add_model_to_train(request: Request):
    print(f"Adding model to train: {request.url}")
    url = request.url
    unique_urls.add(url)
    print(f"all unique urls: {unique_urls}")
    return Response(output=f"Model added to train: {url}")

# curl -X 'POST' \
#   'https://leancopilotapi.onrender.com/train/' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{"url": "https://github.com/Adarsh321123/new-version-test.git"}'

@app.get("/get_urls/")
async def get_urls():
    print(f"Getting all unique urls: {unique_urls}")
    return {"urls": list(unique_urls)}

# curl -X 'GET' \
#   'https://leancopilotapi.onrender.com/get_urls/' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json'

@app.get("/latest_model/")
async def get_latest_model():
    # TODO: later, the compute server can tell the endpoint what the latest is so we can avoid many HF calls, at that piont we can also delete existing urls
    # TODO: use api key if needed
    # This endpoint handles errors gracefully by always returning a successful response
    try:
        response = requests.get(f"{HUGGINGFACE_API_URL}?author={USER}", timeout=10)
        models = response.json()

        if not models:
            print("No models found for user")
            return ModelInfo(
                completed=False,
                message="No models found for user",
                ct2_model_name=None,
                ct2_url=None,
                emb_name=None,
                emb_url=None,
                last_modified=None
            )

        sorted_models = sorted(models, key=lambda x: x['createdAt'], reverse=True)
        # TODO: don't set to None since if no model is found, it will return None
        latest_model = None
        for model in sorted_models:
            if "ct2" in model['modelId']:
                latest_model = model
                break

        latest_emb = None
        for model in sorted_models:
            if "emb" in model['modelId']:
                latest_emb = model
                break

        print(f"latest_model id is: {latest_model['modelId']}")
        print(f"latest_emb id is: {latest_emb['modelId']}")
        return ModelInfo(
            completed=True,
            message="Latest model retrieved successfully",
            ct2_model_name=latest_model['modelId'],
            ct2_url=f"https://huggingface.co/{latest_model['modelId']}",
            emb_name=latest_emb['modelId'],
            emb_url=f"https://huggingface.co/{latest_emb['modelId']}",
            last_modified=latest_model['createdAt']
        )
    except requests.RequestException as e:
        print(f"Error: {e}")
        return ModelInfo(
            completed=False,
            message=f"Unable to fetch latest model due to a network issue. Please try again later.",
            ct2_model_name=None,
            ct2_url=None,
            emb_name=None,
            emb_url=None,
            last_modified=None
        )
    
# curl -X 'GET' \
#   'https://leancopilotapi.onrender.com/latest_model/' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json'

def main():
    import ipdb; ipdb.set_trace()
    # TODO: use api key if needed
    # This endpoint handles errors gracefully by always returning a successful response
    try:
        response = requests.get(f"{HUGGINGFACE_API_URL}?author={USER}", timeout=10)
        models = response.json()

        if not models:        
            return ModelInfo(
                completed=False,
                message="No models found for user",
                ct2_model_name=None,
                ct2_url=None,
                emb_name=None,
                emb_url=None,
                last_modified=None
            )

        sorted_models = sorted(models, key=lambda x: x['createdAt'], reverse=True)
        latest_model = None
        for model in sorted_models:
            if "ct2" in model['modelId']:
                latest_model = model
                break

        latest_emb = None
        for model in sorted_models:
            if "emb" in model['modelId']:
                latest_emb = model
                break

        return ModelInfo(
            completed=True,
            message="Latest model retrieved successfully",
            ct2_model_name=latest_model['modelId'],
            ct2_url=f"https://huggingface.co/{latest_model['modelId']}",
            emb_name=latest_emb['modelId'],
            emb_url=f"https://huggingface.co/{latest_emb['modelId']}",
            last_modified=latest_model['createdAt']
        )
    except requests.RequestException as e:
        return ModelInfo(
            completed=False,
            message=f"Unable to fetch latest model due to a network issue. Please try again later.",
            ct2_model_name=None,
            ct2_url=None,
            emb_name=None,
            emb_url=None,
            last_modified=None
        )

if __name__ == "__main__":
    # main()
    uvicorn.run(app, host="127.0.0.1", port=8000)