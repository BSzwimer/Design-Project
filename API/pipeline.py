#from __future__ import unicode_literals

import requests
import spacy
import nltk
import random
import os
import sys
import time
import re
import language_tool_python
import numpy as np
import wikipedia
import csv
import torch
import clip
import pickle

from numpy import dot
from numpy.linalg import norm
from array import array
from PIL import Image
from spacy import displacy
from collections import Counter
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from rake_nltk import Rake
#from __future__ import unicode_literals
from bs4 import BeautifulSoup
from collections import namedtuple
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

nlp = spacy.load('en_core_web_lg')
tool = language_tool_python.LanguageTool('en-US')
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

subscription_key = "143964306b7e4cea8f7750613ce3e7de"
endpoint = "https://attack.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
device = "cuda" if torch.cuda.is_available() else "cpu"
openai_model, openai_preprocess = clip.load("ViT-B/32", device=device)




def getPageData(page):
  """
  Scrape a wikipedia page for all adjacent text/alternative text/image caption pairs
  """
  soup = BeautifulSoup(page.html(), 'html.parser')
  element = soup.div.div

  Datum = namedtuple('Datum', 'adj_text, alt_text, img_caption, img_url')

  page_data = []
  while element and element.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
    element = element.next_sibling

  adj_text = ''
  found_image = False

  while element:
    if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
      if found_image:
        page_data.append(Datum(adj_text, alt_text, img_caption, img_url))
      found_image = False
      adj_text = ''
    if element.name == 'div' and not found_image:
      img = element.find('img')
      if img:
        img_caption = element.text
        alt_text = img.get('alt',None)
        img_url = img.get('src', None)
        if alt_text:
          found_image = True
    if element.name == 'p':
      adj_text += element.text
    element = element.next_sibling
  return page_data

"""
Strategies
"""

class Strategy:
  """
  Class to call the methods as strings
  """

  def adj(self, page, datum):
    """
    Get Adjacent text of image
    """
    return datum.adj_text

  def tit(self, page, datum):
    """
    Get page title
    """
    return page.title

  def sum(self, page, datum):
    """
    Get page summary
    """
    return page.summary

  def all(self, page, datum):
    """
    Get all text on a page
    """
    return page.content

  def alt(self, page, datum):
    """
    Get image alternative text
    """
    return datum.alt_text

  def cmp_vsn(self, page, datum):
    description_results = computervision_client.describe_image('http:'+datum.img_url)
    if len(description_results.captions) == 0:
      return "No description detected."
    return description_results.captions[0].text

def get_content(page, datum, input_strategies, output_strategies):
  """
  Get input and out using various strategies
  """
  input = ''
  output = ''

  m = globals()['Strategy']()

  for strategy in input_strategies:
    input += getattr(m, strategy)(page, datum)
  for strategy in output_strategies:
    output += getattr(m, strategy)(page, datum)
  return input, output

"""
Various Helper methods for our pipelines
"""

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())

def flip_entities(entities_dict):
  """
  Flip a dictionary to include value:key pairs from
  key:value pairs
  """
  result = {}
  for key, val in entities_dict.items():
    if len(val) > 0:
      result[key] = val
      for v in val:
        x = [*val, key]
        x.remove(v)
        result[v] = x
  return result

def getRelatedWords(adjacent_text):
  """
  Get words which are mentioned together from the
  Google entity API
  """
  url = "https://language.googleapis.com/v1/documents:analyzeEntities?key=AIzaSyDFgIPD1fYHXk3FFcvN8TMw6SrSfaM7wbY"
  json = {
    "document":{
      "type":"PLAIN_TEXT",
      "content": adjacent_text
    },
    "encodingType":"UTF8"
  }
  response = requests.post(url, json=json)
  entities = response.json()['entities']
  entities_dict = {entity["name"].lower():[x['text']['content'] for x in entity['mentions'] if x['text']['content'] != entity["name"].lower()] for entity in entities if len(entity['mentions']) > 1}
  return flip_entities(entities_dict)

def preprocess(sent):
    """
    Preprocess a given text to extract out the Nouns
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sent.lower())
    sent = [w for w in word_tokens if not w in stop_words]
    sent = nltk.pos_tag(sent)
    return [x[0] for x in sent if 'NN' in x[1]]

def vec(s):
    """
    Convert a word to a given vector
    """
    return nlp.vocab[s].vector

def embed_sentence(s):
    """
    Convert a word to a given vector
    """
    return sbert_model.encode([s])[0]

def cosine(v1, v2):
    """
    Find closest vectors using the cosine rule
    """
    if norm(v1) > 0 and norm(v2) > 0:
        return dot(v1, v2) / (norm(v1) * norm(v2))
    else:
        return 0.0

def spacy_closest(token_list, vec_to_check, n=10):
    """
    Returns the n closest vectors to a given vector
    """
    return sorted(token_list,
                  key=lambda x: cosine(vec_to_check, vec(x)),
                  reverse=True)[:n]

def bert_closest(token_list, vec_to_check):
    """
    Returns the n closest vectors to a given vector
    """
    return sorted(token_list,
                  key=lambda x: cosine(vec_to_check, embed_sentence(x)),
                  reverse=True)[0]


def nounPipeline(input_content, output_content):
  """
  Pipeline for swapping nouns using the noun swapping approach
  """
  image_description_tags = preprocess(output_content)
  adjacent_text_tags = preprocess(input_content)

  entities_dict = getRelatedWords(input_content)
  result_text = output_content
  for img_tag in image_description_tags:
    if img_tag in entities_dict:
      result_text = result_text.replace(img_tag, random.choice(entities_dict[img_tag]))
  return result_text

def phrasePipeline(input_content, output_content):
  """
  Pipeline for swapping key phrases using the key phrases swapping approach
  """
  image_description_tags = preprocess(output_content)
  entities_dict = getRelatedWords(input_content)

  r = Rake() 
  r.extract_keywords_from_text(input_content)
  scored = r.get_ranked_phrases_with_scores() 

  result_text = output_content
  for img_tag in image_description_tags:
    max_for_tag = -1
    best_phrase = None
    if img_tag in entities_dict:
      for val in entities_dict[img_tag]:
        for rank,phrase in scored:
          if val in phrase and rank > max_for_tag:
            max_for_tag = rank
            best_phrase = phrase
    if best_phrase:
      result_text = result_text.replace(img_tag,best_phrase)
  return result_text

def embeddingPipeline(input_content, output_content):
  """
  Pipeline for swapping words based on their word embedding proximity
  """
  doc_adj = nlp(input_content)
  tokens = list([w.text for w in doc_adj if w.is_alpha])

  r = Rake() 
  r.extract_keywords_from_text(output_content)
  scored_img = r.get_ranked_phrases_with_scores() 

  r.extract_keywords_from_text(input_content)
  scored_adj = r.get_ranked_phrases_with_scores()

  result_text = output_content.lower()

  for phrase in scored_img:
    avg_vector = np.mean([vec(w) for w in phrase[1].split(' ')], axis=0)
    closest = spacy_closest(tokens, avg_vector, 25)
    for score in scored_adj:
      for close in closest:
        found = False
        if close in score[1]:
          result_text = result_text.replace(phrase[1], score[1])
          found = True
          break
      if found:
        break

  tool.check(result_text)
  result_text = tool.correct(result_text)
  return result_text

def sentenceEmbeddingPipeline(input_content, output_content):
  """
  Pipeline for finding closest sentence int the input content
  to the output content
  """
  sentences = tokenize.sent_tokenize(input_content)
  query_embedding = sbert_model.encode([output_content])[0]
  return bert_closest(sentences, query_embedding)


def openAIPipeline(input_content, image_url):
    image = openai_preprocess(Image.open(requests.get(image_url, stream=True).raw)).unsqueeze(0).to(device)
    sentences = re.split('[?.,]', input_content)
    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        image_features = openai_model.encode_image(image)
        text_features = openai_model.encode_text(text)

        logits_per_image, logits_per_text = openai_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return sentences[np.argmax(probs)]

def getImageCategories(image_url):
    remote_image_features = ["categories"]
    category_results = computervision_client.analyze_image(image_url , remote_image_features)
    categories = {
        'animal': 0,
        'building': 0,
        'indoor': 0,
        'outdoor': 0,
        'people': 0,
    }
    for category in category_results.categories:
      base_category = category.name.split('_')[0]
      if base_category in categories:
        categories[base_category] = 1
    return categories

def getImageDescription(image_url):
    description_results = computervision_client.describe_image(image_url)
    if len(description_results.captions) == 0:
      return "No description detected.", 0
    return description_results.captions[0].confidence, description_results.captions[0].text

def AggregatePipeline(input_content, output_content, image_url):
  strategy = Strategy()
  image = openai_preprocess(Image.open(requests.get(image_url, stream=True).raw)).unsqueeze(0).to(device)
  input_content += nounPipeline(input_content, output_content) + ". "
  input_content += phrasePipeline(input_content, output_content) + ". "
  input_content += embeddingPipeline(input_content, output_content) + ". "
  input_content += sentenceEmbeddingPipeline(input_content, output_content) + ". "
  sentences = re.split('[?.,]',input_content)
  sentences.append(output_content)
  text = clip.tokenize(sentences).to(device)
  with torch.no_grad():
    image_features = openai_model.encode_image(image)
    text_features = openai_model.encode_text(text)
    logits_per_image, logits_per_text = openai_model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    prob, sentence = sorted(zip(probs[0], sentences))[::-1][0]
    index = sentences.index(output_content)
    output_prob = probs[0][index]
  return prob, sentence, output_prob, output_content

def scoreOutput(result, caption):
  result_embedding = sbert_model.encode([result])[0]
  caption_embedding = sbert_model.encode([caption])[0]
  return cosine(result_embedding, caption_embedding)

"""
Final Model
"""

def getResult(page_name):
    result = {}
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    page = wikipedia.page(page_name)
    data = getPageData(page)
    for datum in data:
      try:
        input_content, output_content = get_content(page, datum, ['adj'], ['cmp_vsn'])

        aggregate_prob, aggregate_description, aggregate_azure_prob, _ = AggregatePipeline(input_content, output_content, 'https:' + datum.img_url)
        azure_prob, azure_description = getImageDescription('https:' + datum.img_url)
        categories = getImageCategories('https:' + datum.img_url)
        prediction = loaded_model.predict(
            [[
              aggregate_prob,
              aggregate_azure_prob,
              azure_prob,
              categories['animal'],
              categories['building'],
              categories['indoor'],
              categories['outdoor'],
              categories['people'],
            ]]
        )
        result['https:' + datum.img_url] = aggregate_description if prediction == 0 else azure_description
      except:
        pass
    return result