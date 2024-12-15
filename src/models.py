# adapted from https://github.com/ninodimontalcino/moralchoice/blob/master/src/models.py
import os
import re
import sys
import time
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict

import torch
import openai
import google.generativeai as genai
from google.api_core import retry
import anthropic

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from langchain_community.llms import HuggingFacePipeline, OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


load_dotenv()

PATH_HF_CACHE = os.getenv('PATH_HF_CACHE')
if(not PATH_HF_CACHE):
    PATH_HF_CACHE = "./cache/"

HF_TOKEN = os.getenv('HF_TOKEN')
GEMINI_KEY = os.getenv('GEMINI_KEY')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_KEY = os.getenv('ANTHROPIC_KEY')

API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

current_dir = os.path.dirname(os.path.abspath(__file__))
model_registry_path = os.path.join(current_dir, 'model_registry.json')

with open(model_registry_path) as file:
    MODELS = json.load(file)

os.makedirs(PATH_HF_CACHE, exist_ok=True)
####################################################################################
# MODEL WRAPPERS
####################################################################################

class LanguageModel:
    """ Generic LanguageModel Class"""

    @staticmethod
    def create(model_name, temperature):
        # create a model instance from the model registry
        try:
            class_name = MODELS[model_name]['model_class']
            cls = getattr(sys.modules[__name__], class_name)
            return cls(model_name, temp=temperature)
        except KeyError:
            raise ValueError(f"Model type '{model_name}' not recognized in the registry.")

    def __init__(self, model_name):
        assert model_name in MODELS, f"Model {model_name} is not supported!"

        # Set some default model variables
        self._model_id = model_name
        self._model_name = MODELS[model_name]["model_name"]
        self._model_endpoint = MODELS[model_name]["endpoint"]
        self._company = MODELS[model_name]["company"]
        self._likelihood_access = MODELS[model_name]["likelihood_access"]
        self._nationality = MODELS[model_name]["nationality"]

    @property
    def company(self):
      return self._company

    @property
    def nationality(self):
      return self._nationality

    @property
    def model_id(self):
        """Return model_id"""
        return self._model_id

    def invoke(self, input_data):
      pass


# ----------------------------------------------------------------------------------------------------------------------
# GEMINI PRO
# ----------------------------------------------------------------------------------------------------------------------

class GeminiModel(LanguageModel):
    """Gemini API Wrapper"""
    def __init__(
          self,
          model_name: str,
          temp: float = 0.5
        ):
        super().__init__(model_name)
        genai.configure(api_key=GEMINI_KEY)
        # Initialize the model and output parser
        config = {
          "temperature" : temp
        }
        safe = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
        ]
        self.model = genai.GenerativeModel(self._model_name, safety_settings = safe, generation_config = config)
        self._temperature = temp


    def invoke(self, input_data):
      prompt = input_data["system"] + "\n" + input_data["instruction"] + "\n" + input_data["dataset"]
      return self.model.generate_content(prompt).text

    @property
    def temperature(self):
       return self._temperature


# ----------------------------------------------------------------------------------------------------------------------
# OpenAI Model
# ----------------------------------------------------------------------------------------------------------------------

class OpenAIModel(LanguageModel):
    """OpenAI API Wrapper"""
    def __init__(
          self,
          model_name: str,
          temp: float = 1.0
        ):
        super().__init__(model_name)

        openai.api_key = OPENAI_KEY
        # Initialize the model and output parser
        self.model = ChatOpenAI(model=self._model_name, temperature = temp)
        self._temperature = temp

        # Define the template
        self.prompt = ChatPromptTemplate.from_template("""{system}\n{instruction}\n{dataset}""")

        # Create the chain
        self.chain = self.prompt | self.model

    def invoke(self, input_data):
        return self.chain.invoke(input_data).content

    @property
    def temperature(self):
       return self._temperature

# ----------------------------------------------------------------------------------------------------------------------
# Anthropic Model class
# ----------------------------------------------------------------------------------------------------------------------

class AnthropicModel(LanguageModel):
    """Anthropic API Wrapper"""
    def __init__(self, model_name: str, temp: float = 1.0, key=ANTHROPIC_KEY):
        super().__init__(model_name)

        # Initialize the model and output parser
        self.model = anthropic.Anthropic(api_key=key)
        self._temperature = temp
        self._model_name = model_name

    def invoke(self, input_data):
      tries = 0
      while tries < len(API_TIMEOUTS):
        try:
            mes = self.model.messages.create(

            model=self._model_name,
            max_tokens=1024,
            system = input_data["system"],
            messages=[
                {"role": "user", "content": input_data["instruction"] + "\n" + input_data["dataset"]}
              ],
            temperature=self._temperature
            )
            return mes.content[0].text
        except Exception as e:
          print(e)
          to_return = "Error"
          tries += 1
          time.sleep(API_TIMEOUTS[tries])

# ----------------------------------------------------------------------------------------------------------------------
# MistralAI Model class
# ----------------------------------------------------------------------------------------------------------------------
class MistralAIModel(LanguageModel):
    """Mistral Models Wrapper using HuggingFace's Model"""
    def __init__(
          self,
          model_name: str,
          prompt_template: str ="""<s>[INST]{system}\n{instruction}\n{dataset}[/INST]""",
          temp: float = 0.5
    ):
        super().__init__(model_name)
        # Setup Device, Model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._temperature = temp

        self._quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, # using 4bit
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            cache_dir=PATH_HF_CACHE,
            device_map="auto",
            quantization_config=self._quantization_config
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=PATH_HF_CACHE
            )

        text_generation_pipeline = transformers.pipeline(
            model=self._model,
            tokenizer=self._tokenizer,
            task="text-generation",
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=True,
            temperature=self._temperature,
            return_full_text=True,
            max_new_tokens=1000,
        )
        self._hfp = HuggingFacePipeline(pipeline=text_generation_pipeline)
        self.make_chain(prompt_template)


    def make_chain(self, prompt_template):
      self._output_parser = StrOutputParser()
      self._prompt = ChatPromptTemplate.from_template(prompt_template)
      self._chain = self._prompt | self._hfp | self._output_parser

    def invoke(self, input_data):
      return self._chain.invoke(input_data)

    @property
    def temperature(self):
      return self._temperature

# ----------------------------------------------------------------------------------------------------------------------
# Gemma Model Class
# ----------------------------------------------------------------------------------------------------------------------

class GemmaModel(LanguageModel):
    """Gemma Models Wrapper using HuggingFace's Model"""
    def __init__(
          self,
          model_name: str,
          prompt_template: str ="""<bos><start_of_turn>user\n{system}\n{instruction} {dataset}<end_of_turn>""",
          temp: float=0.5
    ):
        super().__init__(model_name)
        self._temperature = temp
        # Setup Device, Model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, # using 4bit
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map="auto",
            quantization_config=self._quantization_config,
            torch_dtype=torch.bfloat16,
            cache_dir=PATH_HF_CACHE,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=PATH_HF_CACHE
        )

        text_generation_pipeline = transformers.pipeline(
            model=self._model,
            tokenizer=self._tokenizer,
            task="text-generation",
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=True,
            temperature=self._temperature,
            return_full_text=True,
            max_new_tokens=1000,
        )
        self._hfp = HuggingFacePipeline(pipeline=text_generation_pipeline)
        self.make_chain(prompt_template)


    def make_chain(self, prompt_template):
      # not working as expected
      self._output_parser = StrOutputParser()
      self._prompt = ChatPromptTemplate.from_template(prompt_template)
      self._chain = self._prompt | self._hfp | self._output_parser


    def _to_message(self, input_data):
       return [
          {
             "role": "user",
             "content":  input_data['system'] + "\n" +input_data['instruction'] + " " + input_data['dataset']
          }
        ]

    def invoke(self, input_data):
        input_ids = self._tokenizer.apply_chat_template(
           conversation=self._to_message(input_data),
           tokenize=True,
           add_generation_prompt=True,
           return_tensors='pt'
        )
        output_ids = self._model.generate(
           input_ids.to('cuda'),
           do_sample=True,
           max_new_tokens=1000,
           temperature=self._temperature,
        )
        response = self._tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response

    @property
    def temperature(self):
      return self._temperature


# ----------------------------------------------------------------------------------------------------------------------
# Falcon Model Class
# ----------------------------------------------------------------------------------------------------------------------
class FalconModel(LanguageModel):
    """Falcon Model Wrapper using HuggingFace's Model"""
    def __init__(self,
                 model_name: str,
                 prompt_template: str =
                 """{system}\nUser: {instruction}\n{dataset}\nAssistant:""",
                 temp: float = 0.5):
        super().__init__(model_name)

        # Setup Device, Model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._temperature = temp

        self._quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, # using 8bit
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.float16,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map="auto",
            quantization_config=self._quantization_config,
            cache_dir=PATH_HF_CACHE,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=PATH_HF_CACHE,
        )

        text_generation_pipeline = transformers.pipeline(
            model=self._model,
            tokenizer=self._tokenizer,
            task="text-generation",
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=True,
            temperature=self._temperature,
            return_full_text=True,
            max_new_tokens=1000,
        )
        self._hfp = HuggingFacePipeline(pipeline=text_generation_pipeline)
        self.make_chain(prompt_template)


    def make_chain(self, prompt_template):
      self._output_parser = StrOutputParser()
      self._prompt = ChatPromptTemplate.from_template(prompt_template)
      self._chain = self._prompt | self._hfp | self._output_parser

    # def _to_message(self, input_data):
    #    return [
    #       {
    #          "role": "user",
    #          "content":  input_data['system'] + "\n" +input_data['instruction'] + " " + input_data['dataset']
    #       }
    #     ]

    def invoke(self, input_data):
        # input_ids = self._tokenizer.apply_chat_template(
        #    conversation=self._to_message(input_data),
        #    tokenize=True,
        #    add_generation_prompt=True,
        #    return_tensors='pt'
        # )
        # output_ids = self._model.generate(
        #    input_ids.to('cuda'),
        #    do_sample=True,
        #    max_new_tokens=1000,
        #    temperature=self._temperature,
        # )
        # response = self._tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return self._chain.invoke(input_data)

    @property
    def temperature(self):
        return self._temperature

# ----------------------------------------------------------------------------------------------------------------------
# Yi Model Class
# ----------------------------------------------------------------------------------------------------------------------

class YiModel(LanguageModel):
    """YI Model Wrapper using HuggingFace's Model"""
    def __init__(self,
                 model_name: str,
                 prompt_template: str =
                 """<|im_start|>system\n{system}\n<|im_start|>user\n{instruction} {dataset}\n<|im_end|>""",
                 temp: float=0.5):
        super().__init__(model_name)

        # Setup Device, Model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._temperature = temp

        self._quantization_config = BitsAndBytesConfig(
             load_in_4bit = True,
             bnb_4bit_quanty_type = "nf4",
             bnb_4bit_use_double_quanty = True,
             bnb_4bit_compute_dtype=torch.bfloat16
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map="auto",
            quantization_config=self._quantization_config,
            cache_dir=PATH_HF_CACHE,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=PATH_HF_CACHE
        )

        text_generation_pipeline = transformers.pipeline(
            model=self._model,
            tokenizer=self._tokenizer,
            task="text-generation",
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=True,
            temperature=temp,
            return_full_text=True,
            max_new_tokens=1000,
            repetition_penalty=1.2,
        )
        self._hfp = HuggingFacePipeline(pipeline=text_generation_pipeline)
        self.make_chain(prompt_template)


    def make_chain(self, prompt_template):
      # not working as expected
      self._output_parser = StrOutputParser()
      self._prompt = ChatPromptTemplate.from_template(prompt_template)
      self._chain = self._prompt | self._hfp | self._output_parser

    def invoke(self, input_data):
        messages = [
            {"role": "system", "content": input_data['system']},
            {"role": "user", "content": input_data['instruction'] + " " + input_data['dataset']},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        output_ids = self._model.generate(
           input_ids.to('cuda'),
           do_sample=True,
           temperature=self._temperature,
           max_new_tokens=1000,
           repetition_penalty=1.2
        )
        response = self._tokenizer.decode(
           output_ids[0][input_ids.shape[1]:],
           skip_special_tokens=True
        )
        return response


    @property
    def temperature(self):
      return self._temperature

if __name__ == '__main__':
    pass
