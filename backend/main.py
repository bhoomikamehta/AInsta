from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Content Moderation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Content Moderation API is running!"}

@app.post("/api/analyze-comment")
async def analyze_comment(request: dict):
    """
    Analyze a comment for toxicity and provide rephrasing suggestions
    For now, this is just a placeholder
    """
    text = request.get("text", "")
    
    return {
        "original_text": text,
        "toxicity": None,
        "rephrases": []
    }

@app.post("/api/analyze-image")
async def analyze_image():
    """
    Analyze an image for privacy concerns
    For now, this is just a placeholder
    """
    return {
        "message": "Image analysis coming soon!",
        "privacy_issues": [],
        "ocr_text": ""
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)