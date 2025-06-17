import pandas as pd
from neo4j import GraphDatabase
import requests
from typing import List, Dict, Any
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, VENICE_API_KEY

def venice_embed(texts: List[str], model: str = "text-embedding-bge-m3") -> List[List[float]]:
    """Get embeddings from Venice.ai API."""
    # For now, return empty embeddings since API key doesn't have access
    print("Warning: Embeddings not available with current API key - using keyword search only")
    return [[0.0] * 384 for _ in texts]  # Return dummy embeddings

def load_csv(path: str, batch_size: int = 1000) -> None:
    """Load CSV data into Neo4j with AI-powered data understanding."""
    # Read CSV
    df = pd.read_csv(path)
    
    # AI-powered data analysis
    print("🧠 Running AI analysis on dataset...")
    from smart_profiler import analyze_dataset
    analyze_dataset(df, f"Dataset loaded from {path}")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    def process_batch(batch_df: pd.DataFrame):
        with driver.session() as session:
            # Create table node
            session.run(
                "MERGE (t:Table {name: $name})",
                name=path.split("/")[-1].replace('\\', '/')
            )
            
            # Process each row
            for _, row in batch_df.iterrows():
                # Create record node
                record_props = {}
                for col, value in row.items():
                    # Convert pandas types to Python types for Neo4j
                    if pd.isna(value):
                        record_props[col] = None
                    elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                        record_props[col] = float(value) if not pd.isna(value) else None
                    else:
                        record_props[col] = str(value)
                
                record_props["id"] = str(row.name)  # Use index as ID
                
                # Create record node and connect to table
                session.run(
                    """
                    MERGE (r:Record {id: $id})
                    SET r += $props
                    WITH r
                    MATCH (t:Table {name: $table_name})
                    MERGE (t)-[:HAS_ROW]->(r)
                    """,
                    id=record_props["id"],
                    props=record_props,
                    table_name=path.split("/")[-1].replace('\\', '/')
                )
                
                print(f"Processed record {record_props['id']}")
    
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
        process_batch(batch_df)
    
    driver.close()
    print(f"Successfully loaded {len(df)} records from {path}")

if __name__ == "__main__":
    # Example usage
    load_csv("sample.csv") 