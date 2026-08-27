import os
import torch

from dotenv import load_dotenv


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

# Text encoder. bge-m3 is a plain XLMRobertaModel with no remote code: 1024-d, 8192 tokens,
# CLS-pooled, 568M parameters, 250k multilingual vocabulary.
#
# Alibaba-NLP/gte-base-en-v1.5 was the preferred candidate -- 768-d, 8192 tokens, 137M parameters,
# 30k English vocabulary -- and it is NOT usable under the transformers>=5.9.0 pin. Its remote code
# in Alibaba-NLP/new-impl registers the `position_ids` buffer with persistent=False and fills it via
# torch.arange at construction. transformers v5 materializes weights into an empty model, so a
# non-persistent buffer absent from the checkpoint is never initialized: the buffer comes back as
# uninitialized memory and every forward pass dies with an IndexError out of the embedding lookup.
# Verified 2026-08-27 against 5.9.0 and 5.16.1, and confirmed as a v5 regression rather than a
# broken model -- the same commit passes under 4.57.6. Revisit only if new-impl is updated upstream.
#
# LLM_NAME = '/software/llm/Alibaba-NLP/gte-base-en-v1.5'  # unusable, see above
LLM_NAME = '/software/llm/BAAI/bge-m3'
# LLM_NAME = '/software/llm/meta-llama/Llama-3.1-8B'
# LLM_NAME = '/software/llm/meta-llama/Llama-3.2-1B' # Small model for local debugging
# LLM_NAME = '/software/llm/meta-llama/Llama-3.1-70B'

# Maximum length of text token sequences. 8192 is bge-m3's full context; discharge summaries
# average ~2,267 tokens, so the 1024 this replaces truncated most of every note.
MAX_TOKEN_LENGTH = 8192

# Pooling used to reduce a token sequence to one embedding. 'cls' takes position 0, which is what
# bge-m3 and gte were contrastively trained to use; 'mean' is the masked mean over non-padding
# tokens, which is what a decoder such as Llama needs because its position 0 is only a BOS token.
TEXT_POOLING = 'cls'
