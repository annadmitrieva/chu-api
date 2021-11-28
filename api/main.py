from copy import deepcopy
import json
import re
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig, TokenClassificationPipeline
from fastapi import FastAPI


with open('api/model/characters.json', 'r') as f:
  allowed_chars = json.load(f)


class NotAllowedCharsError(Exception):
  # Raises an exception if unknown symbols are found in input

  def __init__(self, chars, message='These chars are not recognized by the model'):
    self.chars = chars
    self.message = message
    super().__init__(self.message)

  def __str__(self):
    return f'{self.message}: {self.chars}'


def process_string(string):
  # Preprocess input, simple tokenization 
  
  string_chars = set(string)
  
  not_allowed = set()
  for char in string_chars:
    if char not in allowed_chars:
      not_allowed.add(char)

  if not not_allowed:
    return re.findall('\w+', string)
  else:
    print(not_allowed)
    raise NotAllowedCharsError(str(not_allowed))


def postprocess_output(output):
  # Poctprocessing model output: get words, their tags and model scores
  # Scores are averaged for multi-token words

  words_and_tags = []

  for token in output:
    
    if len(token) == 1:
      words_and_tags.append((token[0]['word'], token[0]['entity'], token[0]['score'])) 
    
    elif len(token) > 1:
      tags = set()
      scores = []
      chars = []
      for chunk in token:
        tags.add(chunk['entity'])
        scores.append(chunk['score'])
        chars.append(chunk['word'].replace('#', ''))
      word = ''.join(chars)
      score = np.average(scores)
      words_and_tags.append((word, str(tags), str(score)))
  
  return words_and_tags


def load_pipeline():
    #config = AutoConfig.from_pretrained('api/model/config.json')
    model_chu = AutoModelForTokenClassification.from_pretrained('annadmitrieva/old-church-slavonic-pos')  
    tokenizer_chu = tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") 
    chubert = TokenClassificationPipeline(model=model_chu, tokenizer=tokenizer_chu, task="pos")
    print('Pipeline loaded')
    return chubert

chu_pipeline = load_pipeline()

app = FastAPI()


@app.get("/")
def hello():
    """ Main page of the app. """
    return "Hello World!"


@app.get('/chu_pos_tagging/{text}')
async def get_pos_tags(text: str):
    processed_text = process_string(text)
    output = chu_pipeline(processed_text)
    out_processed = postprocess_output(output)
    return str(out_processed)
