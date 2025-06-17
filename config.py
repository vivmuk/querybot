from dotenv import dotenv_values
import os

# Load environment variables from .env file
config = dotenv_values(".env")

# Use Neo4j as requested by user
USE_LOCAL_STORAGE = False

# Neo4j credentials (user has initialized Neo4j)
NEO4J_URI = "neo4j+s://e10687d6.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "t5KwbwanUivhP7fe29DCAwPYwqbRbKHpMFtxwAmZ-ZI"

# Venice.ai API key 
VENICE_API_KEY = "SjXF4gSRAvPU_nz99kSeInMOkoZJCNQpSDpWMoJHJ0"

# Comment out the validation for now since we're hardcoding
# required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "VENICE_API_KEY"]
# missing_vars = [var for var in required_vars if not config.get(var)]
# 
# if missing_vars:
#     raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}") 