# app/main.py
import os
import logging
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("app")

# Comprehensive prefix data for Nigerian carriers
prefix_to_carrier = {
    # MTN Prefixes
    "0803": "MTN", "0806": "MTN", "0703": "MTN", "0706": "MTN",
    "0813": "MTN", "0816": "MTN", "0810": "MTN", "0814": "MTN",
    "0903": "MTN", "0906": "MTN",
    
    # GLO Prefixes
    "0805": "GLO", "0705": "GLO", "0815": "GLO", "0811": "GLO",
    "0905": "GLO", "0807": "GLO",
    
    # AIRTEL Prefixes
    "0802": "AIRTEL", "0708": "AIRTEL", "0812": "AIRTEL", "0701": "AIRTEL",
    "0808": "AIRTEL", "0902": "AIRTEL", "0907": "AIRTEL", "0901": "AIRTEL",
    
    # 9MOBILE (ETISALAT) Prefixes
    "0809": "9MOBILE", "0817": "9MOBILE", "0818": "9MOBILE",
    "0908": "9MOBILE", "0909": "9MOBILE", "0704": "9MOBILE"
}

# Get environment variables with defaults
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# In production, this should be set to specific origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Input Models
class PhoneNumber(BaseModel):
    phone_number: str = Field(..., min_length=4, max_length=15)
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        # Remove spaces and validate
        v = v.strip()
        if not v.isdigit():
            raise ValueError("Phone number must contain only digits")
        return v

# Output Models
class CarrierResponse(BaseModel):
    carrier: str
    success: bool = True
    message: Optional[str] = None

# Error response model
class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: Optional[str] = None

# Initialize the app
app = FastAPI(
    title="Carrier Lookup API",
    description="API for identifying Nigerian mobile carriers based on phone number prefixes",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="An unexpected error occurred",
            detail=str(exc) if ENVIRONMENT != "production" else None
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
        ).dict()
    )

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Main carrier check endpoint
@app.post("/api/check-carrier", response_model=CarrierResponse)
def check_carrier(data: PhoneNumber):
    try:
        logger.info(f"Checking carrier for number: {data.phone_number[:4]}****")
        phone = data.phone_number
        
        if len(phone) >= 4:
            prefix = phone[:4]
            carrier = prefix_to_carrier.get(prefix, "Unknown")
        else:
            carrier = "Unknown"
            
        return CarrierResponse(carrier=carrier, message="Carrier identified successfully")
    
    except Exception as e:
        logger.error(f"Error checking carrier: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing request")

if __name__ == "__main__":
    # This is used when running locally
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=ENVIRONMENT != "production")
