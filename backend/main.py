from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
import time
import asyncio
from enum import Enum

# Import our enhanced components
try:
    from perspective import PerspectiveAPI
    from llama import LlamaRephrase, RephrasingStyle
    from logging_system import ToxicityLogger
    TOXICITY_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Toxicity detection modules not found: {e}")
    TOXICITY_DETECTION_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Content Moderation API", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
perspective_api = None
llama_rephraser = None
toxicity_logger = None

if TOXICITY_DETECTION_AVAILABLE:
    try:
        perspective_api = PerspectiveAPI()
        llama_rephraser = LlamaRephrase()
        toxicity_logger = ToxicityLogger("logs/toxicity_analysis.csv")
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")

# Request/Response Models
class StyleEnum(str, Enum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    FORMAL = "formal"

class AdvancedCommentRequest(BaseModel):
    text: str
    toxicity_threshold: Optional[float] = 0.7
    include_rephrase: Optional[bool] = True
    styles: Optional[List[StyleEnum]] = [StyleEnum.NEUTRAL, StyleEnum.FRIENDLY, StyleEnum.FORMAL]
    max_rephrases: Optional[int] = 3

class RephraseResult(BaseModel):
    style: str
    text: str
    toxicity_score: Optional[float] = None
    improvement: Optional[float] = None
    generation_time: Optional[float] = None

class AdvancedAnalysisResult(BaseModel):
    original_text: str
    toxicity: dict
    rephrases: List[RephraseResult]
    processing_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    fallback_used: bool = False

@app.get("/")
async def root():
    return {
        "message": "Enhanced Content Moderation API v3.0",
        "features": {
            "toxicity_detection": TOXICITY_DETECTION_AVAILABLE and perspective_api is not None,
            "multi_style_rephrasing": TOXICITY_DETECTION_AVAILABLE and llama_rephraser is not None,
            "before_after_analysis": True,
            "comprehensive_logging": True,
            "error_handling": True,
            "fallback_responses": True
        },
        "endpoints": {
            "analyze": "/api/analyze-comment",
            "rephrase_only": "/api/rephrase",
            "stats": "/api/stats",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    status = {
        "api": "healthy",
        "timestamp": time.time(),
        "services": {
            "perspective_api": "not_available",
            "llama_rephraser": "not_available",
            "logging_system": "not_available"
        }
    }
    
    if perspective_api:
        try:
            test_result = perspective_api.analyze_comment("Hello world test")
            status["services"]["perspective_api"] = "healthy" if test_result['success'] else "error"
        except Exception as e:
            status["services"]["perspective_api"] = f"error: {str(e)}"
    
    if llama_rephraser:
        is_connected, message = llama_rephraser.test_connection()
        status["services"]["llama_rephraser"] = "healthy" if is_connected else f"error: {message}"
    
    if toxicity_logger:
        try:
            stats = toxicity_logger.get_log_stats()
            status["services"]["logging_system"] = "healthy" if "error" not in stats else f"error: {stats['error']}"
        except Exception as e:
            status["services"]["logging_system"] = f"error: {str(e)}"
    
    return status

@app.post("/api/analyze-comment", response_model=AdvancedAnalysisResult)
async def analyze_comment_advanced(request: AdvancedCommentRequest):
    """
    Advanced comment analysis with multiple rephrasing styles and comprehensive logging
    """
    start_time = time.time()
    
    if not perspective_api:
        raise HTTPException(
            status_code=503,
            detail="Perspective API not available. Check GOOGLE_API_KEY environment variable."
        )
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 characters)")
    
    # Initialize result structure
    result_data = {
        "original_text": text,
        "toxicity": {},
        "rephrases": [],
        "processing_time_seconds": 0.0,
        "success": True,
        "error_message": None,
        "fallback_used": False,
        "endpoint": "analyze-comment"
    }
    
    try:
        # Step 1: Analyze original toxicity
        logger.info(f"Analyzing comment: {text[:50]}...")
        
        is_toxic, toxicity_score = perspective_api.is_toxic(text, request.toxicity_threshold)
        
        # Get detailed analysis for confidence
        detailed_analysis = perspective_api.analyze_comment(text)
        confidence = 0.0
        if detailed_analysis['success'] and 'toxicity' in detailed_analysis['results']:
            confidence = detailed_analysis['results']['toxicity'].get('confidence', 0.0)
        
        result_data["toxicity"] = {
            "score": toxicity_score,
            "is_toxic": is_toxic,
            "confidence": confidence,
            "threshold_used": request.toxicity_threshold
        }
        
        # Step 2: Generate rephrases if needed
        if is_toxic and request.include_rephrase and llama_rephraser:
            logger.info(f"Generating rephrases for toxic comment (score: {toxicity_score:.3f})")
            
            # Convert StyleEnum to RephrasingStyle
            llama_styles = []
            for style in request.styles[:request.max_rephrases]:
                if style == StyleEnum.NEUTRAL:
                    llama_styles.append(RephrasingStyle.NEUTRAL)
                elif style == StyleEnum.FRIENDLY:
                    llama_styles.append(RephrasingStyle.FRIENDLY)
                elif style == StyleEnum.FORMAL:
                    llama_styles.append(RephrasingStyle.FORMAL)
            
            rephrase_result = llama_rephraser.generate_multiple_rephrases(
                text, toxicity_score, llama_styles
            )
            
            if rephrase_result['success'] and rephrase_result['rephrases']:
                # Analyze toxicity of each rephrase
                for rephrase_data in rephrase_result['rephrases']:
                    rephrase_text = rephrase_data['text']
                    
                    # Get toxicity score for this rephrase
                    _, rephrase_toxicity_score = perspective_api.is_toxic(rephrase_text)
                    
                    # Calculate improvement
                    improvement = toxicity_score - rephrase_toxicity_score
                    
                    result_data["rephrases"].append({
                        "style": rephrase_data['style'],
                        "text": rephrase_text,
                        "toxicity_score": rephrase_toxicity_score,
                        "improvement": improvement,
                        "generation_time": rephrase_data.get('generation_time', 0.0)
                    })
                
                logger.info(f"Generated {len(result_data['rephrases'])} rephrases successfully")
            
            elif rephrase_result.get('fallback_message'):
                # Use fallback message if rephrasing completely failed
                result_data["fallback_used"] = True
                result_data["rephrases"].append({
                    "style": "fallback",
                    "text": rephrase_result['fallback_message'],
                    "toxicity_score": 0.1,  # Assume fallbacks are safe
                    "improvement": toxicity_score - 0.1,
                    "generation_time": 0.0
                })
                result_data["error_message"] = "Rephrasing failed, using fallback response"
            
            else:
                result_data["error_message"] = f"Rephrasing failed: {rephrase_result.get('errors', ['Unknown error'])}"
        
        elif is_toxic and request.include_rephrase and not llama_rephraser:
            result_data["error_message"] = "Rephrasing requested but LLaMA service not available"
        
        # Calculate total processing time
        result_data["processing_time_seconds"] = time.time() - start_time
        result_data["model_used"] = getattr(llama_rephraser, '_working_model', 'unknown') if llama_rephraser else 'none'
        
        # Step 3: Log the analysis (if logging is available)
        if toxicity_logger:
            try:
                toxicity_logger.log_analysis(result_data)
            except Exception as log_error:
                logger.error(f"Logging failed: {log_error}")
        
        # Convert to response model
        response = AdvancedAnalysisResult(
            original_text=result_data["original_text"],
            toxicity=result_data["toxicity"],
            rephrases=result_data["rephrases"],
            processing_time_seconds=result_data["processing_time_seconds"],
            success=result_data["success"],
            error_message=result_data.get("error_message"),
            fallback_used=result_data.get("fallback_used", False)
        )
        
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error in analyze_comment: {str(e)}")
        
        # Log the error analysis
        error_result = {
            "original_text": text,
            "toxicity": {"score": 0.0, "is_toxic": False, "error": str(e)},
            "rephrases": [],
            "processing_time_seconds": time.time() - start_time,
            "success": False,
            "error_message": f"Analysis failed: {str(e)}",
            "endpoint": "analyze-comment"
        }
        
        if toxicity_logger:
            try:
                toxicity_logger.log_analysis(error_result)
            except:
                pass  # Don't fail on logging errors
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/api/rephrase")
async def rephrase_only(request: dict):
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
        # Get requested styles or use default
        requested_styles = request.get("styles", ["neutral", "friendly", "formal"])
        llama_styles = []
        
        for style in requested_styles:
            if style.lower() == "neutral":
                llama_styles.append(RephrasingStyle.NEUTRAL)
            elif style.lower() == "friendly":
                llama_styles.append(RephrasingStyle.FRIENDLY)
            elif style.lower() == "formal":
                llama_styles.append(RephrasingStyle.FORMAL)
        
        result = llama_rephraser.generate_multiple_rephrases(text, styles=llama_styles)
        
        if result['success']:
            return {
                "original_text": text,
                "rephrases": result['rephrases'],
                "success": True,
                "model_used": result.get('model_used', 'unknown')
            }
        else:
            return {
                "original_text": text,
                "rephrases": [],
                "success": False,
                "error": result.get('errors', 'Unknown error'),
                "fallback_message": result.get('fallback_message', 'Sorry, rephrasing failed.')
            }
            
    except Exception as e:
        logger.error(f"Error in rephrase endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Rephrasing failed: {str(e)}"
        )

@app.get("/api/stats")
async def get_stats():
    """
    Get statistics from the logging system
    """
    if not toxicity_logger:
        raise HTTPException(
            status_code=503,
            detail="Logging system not available"
        )
    
    try:
        stats = toxicity_logger.get_log_stats()
        return {
            "success": True,
            "stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
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
        "ocr_text": "",
        "version": "3.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check environment setup
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        print("   Toxicity detection will not work without it.")
        print("   Set it with: export GOOGLE_API_KEY='your-api-key'")
    
    print("🚀 Starting Enhanced Content Moderation API v3.0...")
    print("📖 API Documentation: http://127.0.0.1:8000/docs")
    print("🏥 Health Check: http://127.0.0.1:8000/health")
    print("📊 Statistics: http://127.0.0.1:8000/api/stats")
    print("\n🎯 New Features:")
    print("   • 3 rephrasing styles (neutral, friendly, formal)")
    print("   • Before/after toxicity analysis")
    print("   • Comprehensive CSV logging")
    print("   • Fallback responses")
    print("   • Enhanced error handling")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)