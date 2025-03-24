"""
LLM module for handling language model operations.
"""

import torch
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import LLM_MODEL_NAME, DEVICE


class QwenLLM(LLM):
    """
    Wrapper for Qwen models to integrate with LangChain.
    """
    
    # Model parameters
    max_token: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    history_len: int = 3
    model_name: str = LLM_MODEL_NAME
    
    # Runtime attributes
    _model = None
    _tokenizer = None
    
    def __init__(self, **kwargs):
        """Initialize the Qwen language model."""
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Load model and tokenizer if not loaded yet
        if QwenLLM._model is None or QwenLLM._tokenizer is None:
            self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        QwenLLM._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=DEVICE if torch.cuda.is_available() else "auto"
        )
        QwenLLM._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Model loaded on device: {QwenLLM._model.device}")
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "Qwen"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """
        Call the LLM with a prompt and return the generated text.
        
        Args:
            prompt: The input text.
            stop: A list of strings to stop generation when encountered.
            run_manager: Callback manager for LLM runs.
            
        Returns:
            Generated text as a string.
        """
        messages = [
            {"role": "system", "content": "You are a game expert and a helpful assistant for League of Legends players."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = QwenLLM._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Convert input to tensor
        model_inputs = QwenLLM._tokenizer([text], return_tensors="pt").to(QwenLLM._model.device)
        
        # Generate response
        generated_ids = QwenLLM._model.generate(
            **model_inputs,
            max_new_tokens=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            pad_token_id=QwenLLM._tokenizer.pad_token_id
        )
        
        # Extract only new tokens (exclude input tokens)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text
        response = QwenLLM._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_len": self.history_len
        }


# Create a singleton instance for reuse
llm = QwenLLM()