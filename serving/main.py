from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# --- 1. Model Loading ---
# Load the model and tokenizer only ONCE when the application starts.
# This is a crucial performance optimization to avoid reloading the large model
# on every single API request.
print("Loading fine-tuned model and tokenizer...")
MODEL_PATH = "./rick-llm-final"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    # Use the pipeline for easy text generation
    rick_llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1, # Use GPU if available
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    rick_llm_pipeline = None

# --- 2. API Contract (Pydantic Models) ---
class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The starting text (prompt) to generate a response from.",
        example="Morty, you gotta..."
    )
    max_length: int = Field(50, gt=10, lt=200, description="Maximum length of the generated text.")

class GenerateResponse(BaseModel):
    generated_text: str

# --- 3. FastAPI Application ---
app = FastAPI(
    title="Character LLM Service (Rick)",
    description="An API to interact with a fine-tuned LLM that talks like Rick Sanchez.",
    version="1.0.0"
)

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "ok" if rick_llm_pipeline else "model_not_loaded"}

@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
def generate_text(request: GenerateRequest):
    """Generates text in the style of Rick Sanchez based on a prompt."""
    if not rick_llm_pipeline:
        raise HTTPException(status_code=503, detail="Model is not available.")
    
    try:
        print(f"Received prompt: {request.prompt}")
        outputs = rick_llm_pipeline(
            request.prompt,
            max_length=request.max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id # Important for clean generation
        )
        generated_text = outputs[0]['generated_text']
        print(f"Generated text: {generated_text}")
        return GenerateResponse(generated_text=generated_text)

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))