"""
Configuration settings for BankDataLens
"""

# Euron API Configuration
EURON_API_KEY = "euri-5af13587821689d1c1c8c50ab9fab3e6d04e4800a8489e6bb87b2df2cd408a75"
EURON_EMBEDDINGS_URL = "https://api.euron.one/api/v1/euri/embeddings"
EURON_CHAT_URL = "https://api.euron.one/api/v1/euri/chat/completions"
EURON_EMBEDDING_MODEL = "text-embedding-3-small"
EURON_CHAT_MODEL = "gpt-4.1-nano"

# Pinecone Configuration
PINECONE_API_KEY = "pcsk_Lwcwo_LzURMduzuFeLZn1yAMgmGZieX59f7imscZTXCHcjoXL6LneKShR1UZxARSGgn1P"
PINECONE_INDEX_NAME = "bankdatalens"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Embedding Configuration
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small dimension

# Application Settings
MAX_FILE_SIZE_MB = 100
MAX_ROWS = 100000
BATCH_SIZE = 20
