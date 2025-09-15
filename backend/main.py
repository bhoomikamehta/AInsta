from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging

# Import our new toxicity detection components
try:
    from perspective import PerspectiveAPI
    from llama import LlamaRephrase
    TOXICITY_DETECTION_AVAILABLE = True
except ImportError:
    print("Warning: Toxicity detection modules not found. Install requirements first.")
    TOXICITY_DETECTION_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Content Moderation API", version="2.0.0")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize toxicity detection services (if available)
perspective_api = None
llama_rephraser = None

if TOXICITY_DETECTION_AVAILABLE:
    try:
        perspective_api = PerspectiveAPI()
        llama_rephraser = LlamaRephrase()
        logger.info("Toxicity detection services initialized")
    except Exception as e:
        logger.error(f"Failed to initialize toxicity detection: {e}")

# Request models
class CommentAnalysisRequest(BaseModel):
    text: str
    toxicity_threshold: Optional[float] = 0.7
    include_rephrase: Optional[bool] = True

@app.get("/")
async def root():
    return {
        "message": "Content Moderation API is running!",
        "version": "2.0.0",
        "features": {
            "toxicity_detection": TOXICITY_DETECTION_AVAILABLE and perspective_api is not None,
            "comment_rephrasing": TOXICITY_DETECTION_AVAILABLE and llama_rephraser is not None,
            "image_analysis": "coming_soon"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check including toxicity detection services"""
    status = {
        "api": "healthy",
        "toxicity_detection": "not_available",
        "rephrasing_service": "not_available"
    }
    
    if perspective_api:
        try:
            test_result = perspective_api.analyze_comment("Hello world")
            status["toxicity_detection"] = "healthy" if test_result['success'] else "error"
        except Exception as e:
            status["toxicity_detection"] = f"error: {str(e)}"
    
    if llama_rephraser:
        is_connected, message = llama_rephraser.test_connection()
        status["rephrasing_service"] = "healthy" if is_connected else f"error: {message}"
    
    return status

@app.post("/api/analyze-comment")
async def analyze_comment(request: CommentAnalysisRequest):
    """
    Enhanced analyze comment endpoint with actual toxicity detection and rephrasing
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = {
        "original_text": text,
        "toxicity": None,
        "rephrases": [],
        "analysis_performed": False,
        "error": None
    }
    
    # If toxicity detection is not available, return basic response
    if not TOXICITY_DETECTION_AVAILABLE or not perspective_api:
        result["error"] = "Toxicity detection not configured. Set GOOGLE_API_KEY environment variable."
        result["toxicity"] = {
            "score": 0.0,
            "is_toxic": False,
            "confidence": 0.0,
            "note": "Analysis not performed - service unavailable"
        }
        return result
    
    try:
        # Step 1: Analyze toxicity
        logger.info(f"Analyzing comment: {text[:50]}...")
        
        is_toxic, toxicity_score = perspective_api.is_toxic(
            text, 
            threshold=request.toxicity_threshold
        )
        
        # Get detailed analysis
        detailed_analysis = perspective_api.analyze_comment(text)
        confidence = 0.0
        if detailed_analysis['success'] and 'toxicity' in detailed_analysis['results']:
            confidence = detailed_analysis['results']['toxicity'].get('confidence', 0.0)
        
        result["toxicity"] = {
            "score": toxicity_score,
            "is_toxic": is_toxic,
            "confidence": confidence,
            "threshold_used": request.toxicity_threshold
        }
        result["analysis_performed"] = True
        
        # Step 2: Generate rephrases if toxic and requested
        if is_toxic and request.include_rephrase and llama_rephraser:
            logger.info(f"Generating rephrases for toxic comment (score: {toxicity_score:.3f})")
            
            rephrase_result = llama_rephraser.rephrase_comment(text, toxicity_score)
            
            if rephrase_result['success']:
                # Verify the rephrase is actually less toxic
                rephrase_text = rephrase_result['rephrased_text']
                _, rephrase_score = perspective_api.is_toxic(rephrase_text)
                
                result["rephrases"] = [{
                    "text": rephrase_text,
                    "toxicity_score": rephrase_score,
                    "improvement": toxicity_score - rephrase_score,
                    "method": "llama_rephrase"
                }]
            else:
                result["error"] = f"Rephrasing failed: {rephrase_result['error']}"
        
        elif is_toxic and request.include_rephrase and not llama_rephraser:
            result["error"] = "Rephrasing requested but LLaMA service not available"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_comment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# Keep the existing image analysis endpoint for future development
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

# New endpoint for testing just the rephrasing functionality
@app.post("/api/rephrase")
async def rephrase_text(request: dict):
    """
    Rephrase text without toxicity analysis (for testing)
    """
    text = request.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if not llama_rephraser:
        raise HTTPException(
            status_code=503,
            detail="LLaMA rephrasing service not available"
        )
    
    try:
        result = llama_rephraser.rephrase_comment(text)
        
        if result['success']:
            return {
                "original_text": text,
                "rephrased_text": result['rephrased_text'],
                "success": True
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Rephrasing failed: {result['error']}"
            )
            
    except Exception as e:
        logger.error(f"Error in rephrase endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Rephrasing failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Check environment setup
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        print("   Toxicity detection will not work without it.")
        print("   Set it with: export GOOGLE_API_KEY='your-api-key'")
    
    print("🚀 Starting Enhanced Content Moderation API...")
    print("📖 API Documentation: http://127.0.0.1:8000/docs")
    print("🏥 Health Check: http://127.0.0.1:8000/health")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)