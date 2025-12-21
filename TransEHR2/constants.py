import os
import torch

from dotenv import load_dotenv
from transformers import AutoConfig


if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

load_dotenv()
HF_API_TOKEN = os.getenv('HF_READ_TOKEN', None)
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes

PAD = 0
TOKENIZER_PAD_TOKEN = '[PAD]'

LLM_NAME = 'meta-llama/Llama-3.1-8B' # Name of LLM on Hugging Face
# LLM_NAME = 'meta-llama/Llama-3.2-1B' # Small, quantized model for local debugging
MAX_TOKEN_LENGTH = 2048  # Maximum length of text token sequences
TEXT_EMBED_DIM = AutoConfig.from_pretrained(LLM_NAME, token=HF_API_TOKEN).hidden_size
