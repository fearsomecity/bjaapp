from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing import List, Union, Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai
import math
from functools import reduce

load_dotenv()

app = FastAPI(
    title="BFHL REST API",
    description="REST API for mathematical operations and AI responses",
    version="1.0.0"
)

OFFICIAL_EMAIL = "aditya0097.be23@chitkara.edu.in"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class FibonacciRequest(BaseModel):
    fibonacci: int = Field(..., gt=0, description="Positive integer for Fibonacci series")

class PrimeRequest(BaseModel):
    prime: List[int] = Field(..., min_items=1, description="Array of integers")

class LCMRequest(BaseModel):
    lcm: List[int] = Field(..., min_items=2, description="Array of at least 2 integers")

class HCFRequest(BaseModel):
    hcf: List[int] = Field(..., min_items=2, description="Array of at least 2 integers")

class AIRequest(BaseModel):
    AI: str = Field(..., min_length=1, max_length=500, description="Question string")

class BFHLRequest(BaseModel):
    fibonacci: Optional[int] = None
    prime: Optional[List[int]] = None
    lcm: Optional[List[int]] = None
    hcf: Optional[List[int]] = None
    AI: Optional[str] = None

    @model_validator(mode='after')
    def check_at_least_one_field(self):
        if not any([self.fibonacci, self.prime, self.lcm, self.hcf, self.AI]):
            raise ValueError("At least one of fibonacci, prime, lcm, hcf, or AI must be provided")
        return self
class SuccessResponse(BaseModel):
    is_success: bool = True
    official_email: str
    data: Union[List[int], int, str]

class ErrorResponse(BaseModel):
    is_success: bool = False
    official_email: str
    error: str

class HealthResponse(BaseModel):
    is_success: bool = True
    official_email: str

def generate_fibonacci(n: int) -> List[int]:
    """Generate Fibonacci series up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib[:n]

def get_primes(numbers: List[int]) -> List[int]:
    """Filter prime numbers from the list"""
    def is_prime(num):
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(num)) + 1, 2):
            if num % i == 0:
                return False
        return True
    
    return [num for num in numbers if is_prime(num)]

def gcd(a: int, b: int) -> int:
    """Calculate GCD using Euclidean algorithm"""
    while b:
        a, b = b, a % b
    return abs(a)

def lcm(a: int, b: int) -> int:
    """Calculate LCM"""
    return abs(a * b) // gcd(a, b)

def calculate_lcm(numbers: List[int]) -> int:
    """Calculate LCM of a list of numbers"""
    return reduce(lcm, numbers)

def calculate_hcf(numbers: List[int]) -> int:
    """Calculate HCF (GCD) of a list of numbers"""
    return reduce(gcd, numbers)

async def get_ai_response(question: str) -> str:
    """Get AI response from Google Gemini"""
    try:
        if not GEMINI_API_KEY:
            return "AI service not configured"
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            f"Answer this question in exactly one word: {question}"
        )
        
        answer = response.text.strip().split()[0] if response.text else "Unknown"
        return answer
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service error: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return {
        "is_success": True,
        "official_email": OFFICIAL_EMAIL
    }

@app.post("/bfhl", response_model=SuccessResponse, status_code=status.HTTP_200_OK)
async def process_bfhl(request: BFHLRequest):
    """
    Process mathematical operations and AI requests
    
    Accepts one of:
    - fibonacci: int - Generate Fibonacci series
    - prime: List[int] - Filter prime numbers
    - lcm: List[int] - Calculate LCM
    - hcf: List[int] - Calculate HCF/GCD
    - AI: str - Get AI response
    """
    
    try:
        # Fibonacci
        if request.fibonacci is not None:
            if request.fibonacci <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="fibonacci must be a positive integer"
                )
            data = generate_fibonacci(request.fibonacci)
            return {
                "is_success": True,
                "official_email": OFFICIAL_EMAIL,
                "data": data
            }
        
        # Prime
        elif request.prime is not None:
            if not request.prime:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="prime array cannot be empty"
                )
            data = get_primes(request.prime)
            return {
                "is_success": True,
                "official_email": OFFICIAL_EMAIL,
                "data": data
            }
        
        # LCM
        elif request.lcm is not None:
            if len(request.lcm) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="lcm array must contain at least 2 integers"
                )
            if any(x == 0 for x in request.lcm):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="lcm array cannot contain zeros"
                )
            data = calculate_lcm(request.lcm)
            return {
                "is_success": True,
                "official_email": OFFICIAL_EMAIL,
                "data": data
            }
        
        # HCF
        elif request.hcf is not None:
            if len(request.hcf) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="hcf array must contain at least 2 integers"
                )
            data = calculate_hcf(request.hcf)
            return {
                "is_success": True,
                "official_email": OFFICIAL_EMAIL,
                "data": data
            }
        
        # AI
        elif request.AI is not None:
            if not request.AI.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="AI question cannot be empty"
                )
            data = await get_ai_response(request.AI)
            return {
                "is_success": True,
                "official_email": OFFICIAL_EMAIL,
                "data": data
            }
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one of fibonacci, prime, lcm, hcf, or AI must be provided"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "is_success": False,
            "official_email": OFFICIAL_EMAIL,
            "error": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "is_success": False,
            "official_email": OFFICIAL_EMAIL,
            "error": "Internal server error"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
