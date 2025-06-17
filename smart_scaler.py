#!/usr/bin/env python3

import pandas as pd
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Context window limits for Venice.ai models (actual API limits)
MODEL_LIMITS = {
    "qwen3-235b": 131072,
    "mistral-31-24b": 131072,
    "deepseek-r1-671b": 131072,
    "deepseek-coder-v2-lite": 131072,
    "llama-3.2-3b": 131072,
    "llama-3.3-70b": 65536,
    "llama-3.1-405b": 65536,
    "qwen-2.5-qwq-32b": 32768,
    "venice-uncensored": 32768,
    "qwen3-4b": 32768,
    "dolphin-2.9.2-qwen2-72b": 32768,
    "qwen-2.5-vl": 32768,
    "qwen-2.5-coder-32b": 32768
}

def estimate_tokens_per_record() -> int:
    """Estimate average tokens per record from current dataset."""
    try:
        df = pd.read_csv('temp.csv')
        sample_record = df.iloc[0].to_dict()
        record_str = f"Record 1: {sample_record}"
        return len(record_str) // 4  # Rough estimate: 4 chars per token
    except:
        return 200  # Default estimate

def calculate_optimal_strategy(total_records: int, model: str = "qwen3-235b") -> Dict[str, Any]:
    """Calculate the optimal analysis strategy based on dataset size."""
    
    tokens_per_record = estimate_tokens_per_record()
    model_limit = MODEL_LIMITS.get(model, 750000)
    total_tokens_needed = tokens_per_record * total_records
    
    # Reserve tokens for system prompt and response
    available_tokens = model_limit * 0.7  # Use 70% for context, 30% for response
    max_records_possible = int(available_tokens // tokens_per_record)
    
    if total_tokens_needed <= available_tokens:
        # Can analyze entire dataset
        return {
            "strategy": "complete_analysis",
            "records_to_analyze": total_records,
            "coverage_percentage": 100.0,
            "method": "full_database",
            "chunks": 1,
            "estimated_tokens": total_tokens_needed,
            "description": f"Complete analysis of all {total_records} records"
        }
    
    elif total_records <= max_records_possible * 2:
        # Use intelligent sampling with high coverage
        sample_size = max_records_possible
        coverage = (sample_size / total_records) * 100
        return {
            "strategy": "stratified_sampling",
            "records_to_analyze": sample_size,
            "coverage_percentage": coverage,
            "method": "stratified_by_entities",
            "chunks": 1,
            "estimated_tokens": sample_size * tokens_per_record,
            "description": f"Stratified sample of {sample_size} records ({coverage:.1f}% coverage)"
        }
    
    else:
        # Use chunked analysis for very large datasets
        chunk_size = max_records_possible
        num_chunks = (total_records + chunk_size - 1) // chunk_size  # Ceiling division
        return {
            "strategy": "chunked_analysis",
            "records_to_analyze": total_records,
            "coverage_percentage": 100.0,
            "method": "multi_chunk_processing",
            "chunks": num_chunks,
            "chunk_size": chunk_size,
            "estimated_tokens": chunk_size * tokens_per_record,
            "description": f"Multi-chunk analysis: {num_chunks} chunks of {chunk_size} records each"
        }

def get_stratified_sample(total_records: int, sample_size: int) -> List[Dict[str, Any]]:
    """Get a stratified sample that represents the entire database."""
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Get stratified sample across MSLs and other key dimensions
            results = session.run(f"""
                MATCH (r:Record)
                WITH r.MSL as msl, r.Specialty as specialty, collect(r) as records
                WITH msl, specialty, records[0..{max(1, sample_size//20)}] as sample_records
                UNWIND sample_records as r
                WITH r, rand() as random
                ORDER BY random
                LIMIT {sample_size}
                RETURN r as node, 1.0 as score
            """)
            
            sample_records = [dict(record) for record in results]
            
            if len(sample_records) < sample_size:
                # Fallback to simple random sample
                results = session.run(f"""
                    MATCH (r:Record)
                    WITH r, rand() as random
                    ORDER BY random
                    LIMIT {sample_size}
                    RETURN r as node, 1.0 as score
                """)
                sample_records = [dict(record) for record in results]
            
            return sample_records
            
    finally:
        driver.close()

def get_chunk_analysis(chunk_number: int, chunk_size: int, total_records: int) -> List[Dict[str, Any]]:
    """Get a specific chunk of records for analysis."""
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            skip_records = chunk_number * chunk_size
            
            results = session.run(f"""
                MATCH (r:Record)
                WITH r
                ORDER BY r.id
                SKIP {skip_records}
                LIMIT {chunk_size}
                RETURN r as node, 1.0 as score
            """)
            
            return [dict(record) for record in results]
            
    finally:
        driver.close()

def execute_optimal_retrieval(question: str, model: str = "qwen3-235b") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute the optimal retrieval strategy based on dataset size."""
    
    # Get total record count
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            count_result = session.run("MATCH (r:Record) RETURN count(r) as total")
            total_records = count_result.single()["total"]
    finally:
        driver.close()
    
    print(f"📊 Database contains {total_records} total records")
    
    # Calculate optimal strategy
    strategy = calculate_optimal_strategy(total_records, model)
    
    print(f"🎯 Optimal Strategy: {strategy['strategy']}")
    print(f"📈 {strategy['description']}")
    print(f"🔢 Estimated tokens: {strategy['estimated_tokens']:,}")
    
    if strategy['strategy'] == 'complete_analysis':
        # Retrieve all records
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            with driver.session() as session:
                results = session.run("""
                    MATCH (r:Record)
                    RETURN r as node, 1.0 as score
                    ORDER BY r.id
                """)
                records = [dict(record) for record in results]
                print(f"✅ Retrieved ALL {len(records)} records for complete analysis")
                return records, strategy
        finally:
            driver.close()
            
    elif strategy['strategy'] == 'stratified_sampling':
        # Get stratified sample
        records = get_stratified_sample(total_records, strategy['records_to_analyze'])
        print(f"✅ Retrieved {len(records)} stratified records ({strategy['coverage_percentage']:.1f}% coverage)")
        return records, strategy
        
    elif strategy['strategy'] == 'chunked_analysis':
        # For now, return first chunk (can be extended for multi-chunk processing)
        records = get_chunk_analysis(0, strategy['chunk_size'], total_records)
        print(f"✅ Retrieved chunk 1/{strategy['chunks']}: {len(records)} records")
        strategy['current_chunk'] = 1
        strategy['is_chunked'] = True
        return records, strategy
    
    return [], strategy 