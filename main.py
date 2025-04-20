from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import json
import pandas as pd
import asyncio
import common
import hashlib
from drop_eval import DropEval
from gpqa_eval import GPQAEval
# from humaneval_eval import HumanEval
from math_eval import MathEval
from mgsm_eval import MGSMEval
from mmlu_eval import MMLUEval
from simpleqa_eval import SimpleQAEval
from sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from datetime import datetime
import os
from sampler.o1_chat_completion_sampler import O1ChatCompletionSampler
from sampler.customgpt_sampler import CustomGPTSampler
from sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Model Evaluator API", 
              description="API for running evaluations on language models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Track active evaluations
active_evaluations = {}
app.mount("/input", StaticFiles(directory="./input"), name="input")
app.mount("/results", StaticFiles(directory="./results"), name="results")
# Model definitions
AVAILABLE_MODELS = {
    "gpt-4o-2024-11-20_assistant": lambda: ChatCompletionSampler(
        model="gpt-4o-2024-11-20",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    ),
    "gpt-4o-2024-11-20_chatgpt": lambda: ChatCompletionSampler(
        model="gpt-4o-2024-11-20",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        max_tokens=2048,
    ),
    "o1-preview": lambda: O1ChatCompletionSampler(
        model="o1-preview",
    ),
    "o1-mini": lambda: O1ChatCompletionSampler(
        model="o1-mini",
    ),
    "gpt-4-turbo-2024-04-09_assistant": lambda: ChatCompletionSampler(
        model="gpt-4-turbo-2024-04-09",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
    ),
    "gpt-4-turbo-2024-04-09_chatgpt": lambda: ChatCompletionSampler(
        model="gpt-4-turbo-2024-04-09",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ),
    "gpt-4o_assistant": lambda: ChatCompletionSampler(
        model="gpt-4o",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    ),
    "gpt-4o_chatgpt": lambda: ChatCompletionSampler(
        model="gpt-4o",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        max_tokens=2048,
    ),
    "gpt-4o-mini-2024-07-18": lambda: ChatCompletionSampler(
        model="gpt-4o-mini-2024-07-18",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    ),
    "claude-3-opus-20240229_empty": lambda: ClaudeCompletionSampler(
        model="claude-3-opus-20240229",
        system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
    ),
    "customgpt": lambda: CustomGPTSampler(model_name="gpt-3.5-turbo"),
}

# Evaluation definitions
AVAILABLE_EVALUATIONS = [
    "simpleqa",
    "mmlu",
    "math",
    "gpqa",
    "mgsm",
    "drop"
]

class EvaluationRequest(BaseModel):
    models: List[str] = Field(..., description="List of model names to evaluate")
    evaluations: List[str] = Field(..., description="List of evaluation names to run")
    debug: bool = Field(False, description="Run in debug mode")
    examples: Optional[int] = Field(None, description="Number of examples to use (overrides default)")
    save_results: bool = Field(True, description="Whether to save results to disk")
    output_dir: Optional[str] = Field(None, description="Custom output directory for results")

def generate_request_id(request_data: Dict) -> str:
    """Generate a unique hash for the request based on its contents and timestamp"""
    request_str = json.dumps(request_data, sort_keys=True)
    unique_input = f"{request_str}-{datetime.now().isoformat()}"
    return hashlib.md5(unique_input.encode()).hexdigest()

def get_evals(eval_name, debug_mode, num_examples=None):
    grading_sampler = ChatCompletionSampler(model="gpt-4o")
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    
    match eval_name:
        case "mmlu":
            return MMLUEval(num_examples=1 if debug_mode else num_examples)
        case "math":
            return MathEval(
                equality_checker=equality_checker,
                num_examples=num_examples,
                n_repeats=1 if debug_mode else 10,
            )
        case "gpqa":
            return GPQAEval(
                n_repeats=1 if debug_mode else 10, num_examples=num_examples
            )
        case "mgsm":
            return MGSMEval(num_examples_per_lang=10 if debug_mode else num_examples)
        case "drop":
            return DropEval(
                num_examples=10 if debug_mode else num_examples,
                train_samples_per_prompt=3,
            )
        case "humaneval":
            return HumanEval(num_examples=10 if debug_mode else num_examples)
        case "simpleqa":
            return SimpleQAEval(
                grader_model=grading_sampler,
                num_examples=10 if debug_mode else num_examples,
            )
        case _:
            raise Exception(f"Unrecognized eval type: {eval_name}")

async def run_evaluations(request: EvaluationRequest, request_id: str) -> AsyncGenerator[str, None]:
    """Run evaluations and yield results directly as they become available"""
    results_dir = request.output_dir or f"results/{request_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    if request.save_results:
        os.makedirs(results_dir, exist_ok=True)
    
    merge_metrics = []
    mergekey2resultpath = {}
    
    try:
        # Mark evaluation as active
        active_evaluations[request_id] = {
            "start_time": datetime.now().isoformat(),
            "models": request.models,
            "evaluations": request.evaluations,
            "status": "running"
        }
        
        # Stream preliminary info
        yield json.dumps({
            "type": "info",
            "request_id": request_id,
            "message": f"Starting evaluations for models: {', '.join(request.models)}",
            "timestamp": datetime.now().isoformat()
        }) + "\n"
        
        for model_name in request.models:
            if model_name not in AVAILABLE_MODELS:
                yield json.dumps({
                    "type": "error",
                    "request_id": request_id,
                    "message": f"Model '{model_name}' not found. Skipping.",
                    "timestamp": datetime.now().isoformat()
                }) + "\n"
                continue
            
            # Initialize the sampler
            sampler = AVAILABLE_MODELS[model_name]()
            
            for eval_name in request.evaluations:
                if eval_name not in AVAILABLE_EVALUATIONS:
                    yield json.dumps({
                        "type": "error",
                        "request_id": request_id,
                        "message": f"Evaluation '{eval_name}' not found. Skipping.",
                        "timestamp": datetime.now().isoformat()
                    }) + "\n"
                    continue
                
                yield json.dumps({
                    "type": "status",
                    "request_id": request_id,
                    "message": f"Running {eval_name} evaluation for {model_name}",
                    "timestamp": datetime.now().isoformat()
                }) + "\n"
                
                try:
                    # Update active evaluation status
                    active_evaluations[request_id]["current_eval"] = f"{eval_name} on {model_name}"
                    
                    # Run the evaluation
                    eval_obj = get_evals(eval_name, request.debug, request.examples)
                    result = eval_obj(sampler)
                    
                    # Process and save results
                    file_stem = f"{eval_name}_{model_name}"
                    debug_suffix = "_DEBUG" if request.debug else ""
                    
                    metrics = result.metrics | {"score": result.score}
                    
                    if request.save_results:
                        report_filename = os.path.join(results_dir, f"{file_stem}{debug_suffix}.html")
                        with open(report_filename, "w") as fh:
                            fh.write(common.make_report(result))
                            
                        result_filename = os.path.join(results_dir, f"{file_stem}{debug_suffix}.json")
                        with open(result_filename, "w") as f:
                            f.write(json.dumps(metrics, indent=2))
                        
                        mergekey2resultpath[file_stem] = result_filename
                    
                    # Stream results
                    result_data = {
                        "type": "result",
                        "request_id": request_id,
                        "eval_name": eval_name,
                        "model_name": model_name,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if request.save_results:
                        result_data["report_path"] = report_filename
                        result_data["result_path"] = result_filename
                    
                    yield json.dumps(result_data) + "\n"
                    
                    # Allow a small pause between operations to manage resources
                    await asyncio.sleep(0.1)
                    
                    # Save for summary
                    metric_value = metrics.get("f1_score", metrics.get("score", None))
                    merge_metrics.append({
                        "eval_name": eval_name,
                        "model_name": model_name,
                        "metric": metric_value
                    })
                    print("Merged metrics:", merge_metrics)
                    
                except Exception as e:
                    yield json.dumps({
                        "type": "error",
                        "request_id": request_id,
                        "message": f"Error in {eval_name} evaluation for {model_name}: {str(e)}",
                        "eval_name": eval_name,
                        "model_name": model_name,
                        "timestamp": datetime.now().isoformat()
                    }) + "\n"

        # Create summary table
        if merge_metrics:
            merge_metrics_df = pd.DataFrame(merge_metrics).pivot_table(
                index="model_name", columns="eval_name", values="metric"
            )

            # Ensure that columns names are reset for JSON serialization
            merge_metrics_df.columns.name = None

            # Fix for JSON serialization - convert to regular nested dict
            summary_dict = merge_metrics_df.reset_index().to_dict(orient='records')
            summary_data = {
                "type": "summary",
                "request_id": request_id,
                "table": summary_dict,
                "markdown": merge_metrics_df.to_markdown(),
                "timestamp": datetime.now().isoformat()
            }
            
            if request.save_results:
                summary_filename = os.path.join(results_dir, "summary.json")
                with open(summary_filename, "w") as f:
                    json.dump(summary_data, f, indent=2)
                summary_data["summary_path"] = summary_filename
            
            yield json.dumps(summary_data) + "\n"
        
        yield json.dumps({
            "type": "complete",
            "request_id": request_id,
            "message": "All evaluations completed",
            "timestamp": datetime.now().isoformat()
        }) + "\n"
    
    except Exception as e:
        yield json.dumps({
            "type": "error",
            "request_id": request_id,
            "message": f"Evaluation process error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }) + "\n"
    
    finally:
        # Update status to completed
        if request_id in active_evaluations:
            active_evaluations[request_id]["status"] = "completed"
            active_evaluations[request_id]["end_time"] = datetime.now().isoformat()
            
            # Schedule cleanup of status after some time
            async def cleanup_status():
                await asyncio.sleep(3600)  # Keep status for 1 hour after completion
                if request_id in active_evaluations:
                    del active_evaluations[request_id]
            
            asyncio.create_task(cleanup_status())

@app.post("/evaluations/run")
async def run_model_evaluations(request: EvaluationRequest = Body(...)):
    """
    Run evaluations on specified models and return streaming results
    """
    # Validate models
    invalid_models = [model for model in request.models if model not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model(s): {', '.join(invalid_models)}. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
        )
    
    # Validate evaluations
    invalid_evals = [eval_name for eval_name in request.evaluations if eval_name not in AVAILABLE_EVALUATIONS]
    if invalid_evals:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid evaluation(s): {', '.join(invalid_evals)}. Available evaluations: {', '.join(AVAILABLE_EVALUATIONS)}"
        )
    
    # Generate a unique request ID
    request_id = generate_request_id(request.dict())
    
    # Return a streaming response - directly connecting to the generator
    return StreamingResponse(
        run_evaluations(request, request_id),
        media_type="application/x-ndjson",
        headers={"X-Request-ID": request_id}
    )

@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Check the status of a specific evaluation"""
    if request_id in active_evaluations:
        return active_evaluations[request_id]
    else:
        return {"status": "not found", "request_id": request_id}

@app.get("/models")
async def list_models():
    """List all available models for evaluation"""
    return {"models": list(AVAILABLE_MODELS.keys())}

@app.get("/evaluations")
async def list_evaluations():
    """List all available evaluation types"""
    return {"evaluations": AVAILABLE_EVALUATIONS}

@app.get("/active-evaluations")
async def list_active_evaluations():
    """List all active evaluations"""
    return {"active_evaluations": active_evaluations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)