import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
import requests
import io
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen
import inferless
from pydantic import BaseModel, Field
from typing import Optional


@inferless.request
class RequestObjects(BaseModel):
    task_type: str = Field(default="image")
    prompt: str = Field(default="What is shown in this image?")
    content_url: str = Field(default="https://www.ilankelman.org/stopsigns/australia.jpg")
    max_new_tokens: Optional[int] = 128

@inferless.response
class ResponseObjects(BaseModel):
    generated_result: str = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self):
        model_path = "microsoft/Phi-4-multimodal-instruct"
        snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors"])
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,device_map="cuda",torch_dtype="auto",
                                                          trust_remote_code=True,_attn_implementation="flash_attention_2"
                                                         ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        self.user_prompt = "<|user|>"
        self.assistant_prompt = "<|assistant|>"
        self.prompt_suffix = "<|end|>"

    def infer(self, request: RequestObjects) -> ResponseObjects:
        if request.task_type == "image":
            prompt = f"{self.user_prompt}<|image_1|>{request.prompt}{self.prompt_suffix}{self.assistant_prompt}"
            image = Image.open(requests.get(request.content_url, stream=True).raw)
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
        else:
            prompt = f"{self.user_prompt}<|audio_1|>{request.prompt}{self.prompt_suffix}{self.assistant_prompt}"
            audio, samplerate = sf.read(io.BytesIO(urlopen(request.content_url).read()))
            inputs = self.processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')

        generate_ids = self.model.generate(**inputs,max_new_tokens=request.max_new_tokens,
                                           generation_config=self.generation_config,)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]        
        
        generateObject = ResponseObjects(generated_result=response)
        return generateObject

    def finalize(self):
        self.model = None
