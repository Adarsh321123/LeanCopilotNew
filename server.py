from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    text: str

class Response(BaseModel):
    output: str

class StatusResponse(BaseModel):
    completed: bool

@app.post("/reverse/")
async def reverse_string(request: Request):
    reversed_text = request.text[::-1]
    print(f"Reversing text: {request.text} -> {reversed_text}")
    return Response(output=reversed_text)

# curl -X 'POST' \
#   'http://127.0.0.1:8000/reverse/' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{"text": "hello"}'

@app.get("/check-status/")
async def check_status():
    from random import choice
    status = choice([True, False])
    print(f"Checking status: {status}")
    return StatusResponse(completed=status)

# curl -X 'GET' \
#   'http://127.0.0.1:8000/check-status/' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
