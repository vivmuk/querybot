#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Any, Tuple
from config import VENICE_API_KEY

def analyze_data_with_ai(sample_data: pd.DataFrame, data_description: str = "") -> Dict[str, Any]:
    """Use AI to understand the dataset structure and generate query strategies."""
    
    # Get sample records for AI analysis
    sample_records = sample_data.head(5).to_dict('records')
    column_info = []
    
    for col in sample_data.columns:
        col_sample = sample_data[col].dropna().head(3).tolist()
        # Convert numpy types to Python native types for JSON serialization
        col_sample = [str(x) if pd.isna(x) else (int(x) if isinstance(x, (np.int64, np.int32)) else (float(x) if isinstance(x, (np.float64, np.float32)) else str(x))) for x in col_sample]
        
        col_info = {
            "name": col,
            "type": str(sample_data[col].dtype),
            "samples": col_sample,
            "unique_count": int(sample_data[col].nunique()),
            "null_count": int(sample_data[col].isnull().sum())
        }
        column_info.append(col_info)
    
    # Create AI prompt for data understanding
    prompt = f"""Analyze this dataset and provide a comprehensive understanding:

DATASET DESCRIPTION: {data_description}

COLUMNS:
{json.dumps(column_info, indent=2)}

SAMPLE RECORDS:
{json.dumps(sample_records[:3], indent=2)}

Please provide a JSON response with:
1. "data_type": What kind of dataset this is (e.g., "sales", "medical", "automotive", "pharma", "financial")
2. "primary_entities": Main entities/subjects (e.g., ["customers", "products"], ["patients", "treatments"], ["MSL", "HCP"])
3. "key_columns": Most important columns for analysis
4. "categorical_columns": Columns good for grouping/counting (brands, types, names, etc.)
5. "numerical_columns": Columns good for comparisons (prices, dates, scores, etc.)
6. "date_columns": Time-related columns
7. "identifier_columns": ID or unique identifier columns
8. "query_patterns": Common query types this data would support
9. "business_context": What business questions this data typically answers

Format as valid JSON only."""

    try:
        headers = {
            "Authorization": f"Bearer {VENICE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen3-235b",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1500,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://api.venice.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120  # Increased timeout for reasoning models
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_analysis = result['choices'][0]['message']['content']
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response
                start_idx = ai_analysis.find('{')
                end_idx = ai_analysis.rfind('}') + 1
                json_str = ai_analysis[start_idx:end_idx]
                
                analysis = json.loads(json_str)
                print("🧠 AI Data Analysis Complete!")
                return analysis
                
            except json.JSONDecodeError:
                print("⚠️ AI response wasn't valid JSON, using fallback analysis")
                return _fallback_analysis(sample_data)
                
    except Exception as e:
        print(f"⚠️ AI analysis failed: {e}, using fallback")
        fallback_result = _fallback_analysis(sample_data)
        
        # Enhanced fallback for pharma data
        if 'MSL' in sample_data.columns:
            fallback_result.update({
                "data_type": "pharma",
                "primary_entities": ["MSL", "HCP"],
                "business_context": "Pharmaceutical MSL interactions and engagements",
                "query_patterns": ["count", "engagement", "interaction", "dfo", "msl", "hcp", "specialty"]
            })
        
        return fallback_result

def _fallback_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Fallback analysis using pattern matching."""
    
    categorical_cols = []
    numerical_cols = []
    date_cols = []
    identifier_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Date columns
        if any(pattern in col_lower for pattern in ['date', 'time', 'year', 'created', 'updated', 'registration']):
            date_cols.append(col)
        
        # Identifier columns
        elif any(pattern in col_lower for pattern in ['id', 'uuid', 'key', 'index']) or df[col].nunique() == len(df):
            identifier_cols.append(col)
        
        # Numerical columns
        elif df[col].dtype in ['int64', 'float64'] or any(pattern in col_lower for pattern in ['price', 'cost', 'amount', 'value', 'count', 'score', 'rating', 'mileage', 'engine']):
            numerical_cols.append(col)
        
        # Categorical columns
        elif df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
            categorical_cols.append(col)
    
    return {
        "data_type": "unknown",
        "primary_entities": [],
        "key_columns": list(df.columns[:5]),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "date_columns": date_cols,
        "identifier_columns": identifier_cols,
        "query_patterns": ["count", "filter", "compare", "aggregate"],
        "business_context": "General data analysis"
    }

def create_dynamic_query_classifier(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create query classification rules based on AI analysis."""
    
    cat_cols = analysis.get('categorical_columns', [])
    num_cols = analysis.get('numerical_columns', [])
    date_cols = analysis.get('date_columns', [])
    entities = analysis.get('primary_entities', [])
    data_type = analysis.get('data_type', 'unknown')
    
    # Create dynamic patterns
    count_patterns = ['count', 'how many', 'total', 'number of', 'entries per', 'unique']
    comparison_patterns = ['highest', 'lowest', 'most', 'least', 'best', 'worst', 'top', 'bottom']
    temporal_patterns = ['oldest', 'newest', 'latest', 'earliest', 'recent', 'first', 'last']
    
    # Add domain-specific patterns
    if data_type == 'pharma' or any('msl' in str(e).lower() for e in entities):
        count_patterns.extend(['dfo', 'engagement', 'interaction', 'hcp'])
        comparison_patterns.extend(['productive', 'effective', 'successful'])
    elif data_type == 'automotive':
        count_patterns.extend(['manufacturer', 'brand', 'model'])
        comparison_patterns.extend(['expensive', 'cheap', 'reliable'])
    elif data_type == 'sales':
        count_patterns.extend(['customer', 'product', 'transaction'])
        comparison_patterns.extend(['revenue', 'profit', 'sales'])
    
    return {
        'categorical_columns': cat_cols,
        'numerical_columns': num_cols,
        'date_columns': date_cols,
        'count_patterns': count_patterns,
        'comparison_patterns': comparison_patterns,
        'temporal_patterns': temporal_patterns,
        'data_context': analysis
    }

def smart_query_analysis(question: str, classifier_rules: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligently analyze queries based on learned data patterns."""
    
    question_lower = question.lower()
    
    # Determine query type
    query_type = "general"
    target_columns = []
    strategy = "semantic_search"
    
    # Count/Frequency queries
    if any(pattern in question_lower for pattern in classifier_rules['count_patterns']):
        query_type = "count_frequency"
        target_columns = classifier_rules['categorical_columns']
        strategy = "aggregation"
    
    # Comparison queries
    elif any(pattern in question_lower for pattern in classifier_rules['comparison_patterns']):
        query_type = "comparison"
        target_columns = classifier_rules['numerical_columns']
        strategy = "sorting"
    
    # Temporal queries
    elif any(pattern in question_lower for pattern in classifier_rules['temporal_patterns']):
        query_type = "temporal"
        target_columns = classifier_rules['date_columns']
        strategy = "temporal_sorting"
    
    # Find most relevant columns for the query
    relevant_columns = []
    for word in question_lower.split():
        for col in classifier_rules.get('categorical_columns', []) + classifier_rules.get('numerical_columns', []):
            if word in col.lower() or col.lower() in word:
                relevant_columns.append(col)
    
    return {
        'query_type': query_type,
        'strategy': strategy,
        'target_columns': target_columns,
        'relevant_columns': relevant_columns,
        'confidence': 0.8 if relevant_columns else 0.5
    }

# Global variable to store analysis
current_data_analysis = None
current_classifier_rules = None

def analyze_dataset(df: pd.DataFrame, description: str = "") -> None:
    """Analyze dataset and store results globally."""
    global current_data_analysis, current_classifier_rules
    
    print("🔍 Analyzing dataset with AI...")
    current_data_analysis = analyze_data_with_ai(df, description)
    current_classifier_rules = create_dynamic_query_classifier(current_data_analysis)
    
    print(f"📊 Dataset Type: {current_data_analysis.get('data_type', 'unknown')}")
    print(f"🎯 Primary Entities: {current_data_analysis.get('primary_entities', [])}")
    print(f"📈 Key Columns: {current_data_analysis.get('key_columns', [])}")
    print("✅ Smart query classification ready!")

def get_smart_query_strategy(question: str) -> Dict[str, Any]:
    """Get intelligent query strategy for any question."""
    if current_classifier_rules is None:
        return {"query_type": "general", "strategy": "semantic_search", "confidence": 0.3}
    
    return smart_query_analysis(question, current_classifier_rules) 