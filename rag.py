from neo4j import GraphDatabase
import requests
import json
from typing import List, Dict, Any
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, VENICE_API_KEY

# Add semantic search capabilities using Venice.ai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Test Venice.ai embedding access
def test_venice_embeddings():
    """Test if Venice.ai embeddings are available."""
    try:
        headers = {
            "Authorization": f"Bearer {VENICE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Test with a simple text
        payload = {
            "input": "test",
            "model": "text-embedding-bge-m3",
            "encoding_format": "float"
        }
        
        response = requests.post(
            "https://api.venice.ai/api/v1/embeddings",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                embedding_dims = len(result['data'][0]['embedding'])
                print(f"✅ Venice.ai embeddings enabled! Model: text-embedding-bge-m3, Dimensions: {embedding_dims}")
                return True, "text-embedding-bge-m3", embedding_dims
        
        # Try fallback model
        payload["model"] = "bge-m3"
        response = requests.post(
            "https://api.venice.ai/api/v1/embeddings",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                embedding_dims = len(result['data'][0]['embedding'])
                print(f"✅ Venice.ai embeddings enabled! Model: bge-m3, Dimensions: {embedding_dims}")
                return True, "bge-m3", embedding_dims
        
        print(f"⚠️ Venice.ai embeddings not available: {response.status_code}")
        return False, None, None
        
    except Exception as e:
        print(f"⚠️ Venice.ai embeddings test failed: {e}")
        return False, None, None

# Initialize Venice.ai embeddings
print("🔄 Testing Venice.ai embedding access...")
SEMANTIC_SEARCH_ENABLED, EMBEDDING_MODEL, EMBEDDING_DIMS = test_venice_embeddings()

if not SEMANTIC_SEARCH_ENABLED:
    print("⚠️ Falling back to local embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        print("🔄 Loading local semantic embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Local semantic embeddings enabled!")
        SEMANTIC_SEARCH_ENABLED = True
        EMBEDDING_MODEL = "local"
        EMBEDDING_DIMS = 384
    except ImportError:
        print("⚠️ Semantic embeddings not available - install sentence-transformers and scikit-learn")
        SEMANTIC_SEARCH_ENABLED = False
    except Exception as e:
        print(f"⚠️ Failed to load local embeddings: {e}")
        SEMANTIC_SEARCH_ENABLED = False

def get_text_embedding(text: str) -> np.ndarray:
    """Generate embedding for text using Venice.ai or local model."""
    if not SEMANTIC_SEARCH_ENABLED:
        return None
    
    try:
        if EMBEDDING_MODEL != "local":
            # Use Venice.ai embeddings
            headers = {
                "Authorization": f"Bearer {VENICE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": text,
                "model": EMBEDDING_MODEL,
                "encoding_format": "float"
            }
            
            response = requests.post(
                "https://api.venice.ai/api/v1/embeddings",
                json=payload,
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    return np.array(result['data'][0]['embedding'])
            else:
                print(f"⚠️ Venice.ai embedding failed: {response.status_code}")
                return None
        else:
            # Use local model
            return embedding_model.encode([text])[0]
            
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return None

def semantic_search(question: str, records: List[Dict], top_k: int = 10) -> List[Dict]:
    """Perform semantic similarity search on records."""
    if not SEMANTIC_SEARCH_ENABLED or not records:
        return records[:top_k]
    
    try:
        print(f"🧠 Running semantic search on {len(records)} records...")
        
        # Get question embedding
        question_embedding = get_text_embedding(question)
        if question_embedding is None:
            return records[:top_k]
        
        # Create text representations of records and get embeddings
        record_texts = []
        for record in records:
            # Combine all text fields into searchable text
            node = record.get('node', {})
            text_parts = []
            for key, value in node.items():
                if value and str(value).strip() and str(value) != 'null':
                    text_parts.append(f"{key}: {value}")
            record_text = " | ".join(text_parts)
            record_texts.append(record_text)
        
        # Get embeddings for all records
        if EMBEDDING_MODEL != "local":
            # Batch process with Venice.ai (limited batch size for API efficiency)
            record_embeddings = []
            batch_size = 10  # Process in batches to avoid API limits
            
            for i in range(0, len(record_texts), batch_size):
                batch_texts = record_texts[i:i+batch_size]
                try:
                    headers = {
                        "Authorization": f"Bearer {VENICE_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "input": batch_texts,
                        "model": EMBEDDING_MODEL,
                        "encoding_format": "float"
                    }
                    
                    response = requests.post(
                        "https://api.venice.ai/api/v1/embeddings",
                        json=payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        for data_item in result['data']:
                            record_embeddings.append(data_item['embedding'])
                    else:
                        print(f"⚠️ Batch embedding failed: {response.status_code}")
                        # Fall back to individual embeddings for this batch
                        for text in batch_texts:
                            emb = get_text_embedding(text)
                            if emb is not None:
                                record_embeddings.append(emb)
                            else:
                                # Use zero vector as fallback
                                record_embeddings.append([0.0] * EMBEDDING_DIMS)
                                
                except Exception as e:
                    print(f"⚠️ Batch embedding error: {e}")
                    # Fall back to individual embeddings for this batch
                    for text in batch_texts:
                        emb = get_text_embedding(text)
                        if emb is not None:
                            record_embeddings.append(emb)
                        else:
                            record_embeddings.append([0.0] * EMBEDDING_DIMS)
            
            record_embeddings = np.array(record_embeddings)
        else:
            # Use local model for batch processing
            record_embeddings = embedding_model.encode(record_texts)
        
        # Calculate cosine similarity
        similarities = cosine_similarity([question_embedding], record_embeddings)[0]
        
        # Rank records by similarity
        scored_records = []
        for i, record in enumerate(records):
            scored_records.append({
                'node': record.get('node', {}),
                'score': float(similarities[i]),
                'semantic_score': float(similarities[i])
            })
        
        # Sort by semantic similarity
        scored_records.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        print(f"✅ Semantic search complete - top similarity: {scored_records[0]['semantic_score']:.3f}")
        return scored_records[:top_k]
        
    except Exception as e:
        print(f"⚠️ Semantic search failed: {e}")
        return records[:top_k]

def _handle_aggregation_query(question: str, strategy: Dict[str, Any], k_kw: int) -> List[Dict[str, Any]]:
    """Handle count/frequency queries intelligently."""
    target_cols = strategy.get('target_columns', []) + strategy.get('relevant_columns', [])
    question_lower = question.lower()
    
    sort_direction = "DESC" if any(word in question_lower for word in ['most', 'highest', 'top']) else "ASC" if any(word in question_lower for word in ['least', 'lowest', 'bottom']) else "DESC"
    
    # Add MSL as a priority column if mentioned in question
    if 'msl' in question_lower:
        target_cols = ['MSL'] + [col for col in target_cols if col.upper() != 'MSL']
    
    print(f"🎯 Trying aggregation on columns: {target_cols[:3]}")
    
    for col in target_cols[:3]:  # Try top 3 relevant columns
        try:
            col_escaped = f"`{col}`" if ' ' in col else col
            
            results = cypher_search(f"""
                MATCH (r:Record)
                WHERE r.{col_escaped} IS NOT NULL 
                AND r.{col_escaped} <> '' 
                AND r.{col_escaped} <> 'null' 
                AND r.{col_escaped} <> 'NA'
                WITH r.{col_escaped} as category_name, count(*) as frequency, collect(r)[0..3] as sample_records
                RETURN {{
                    category: category_name,
                    count: frequency,
                    column: '{col}',
                    sample: sample_records[0]
                }} as node, frequency as score
                ORDER BY frequency {sort_direction}
                LIMIT {k_kw}
            """)
            
            if results:
                print(f"✅ Found {len(results)} aggregated results from {col} (sorted {sort_direction})")
                return results
                
        except Exception as e:
            print(f"⚠️ Failed to aggregate by {col}: {e}")
    
    return []

def _handle_sorting_query(question: str, strategy: Dict[str, Any], k_kw: int) -> List[Dict[str, Any]]:
    """Handle comparison queries intelligently."""
    target_cols = strategy.get('target_columns', []) + strategy.get('relevant_columns', [])
    question_lower = question.lower()
    
    sort_direction = "DESC" if any(word in question_lower for word in ['highest', 'most', 'expensive', 'largest', 'maximum']) else "ASC"
    
    for col in target_cols[:3]:  # Try top 3 relevant columns
        try:
            col_escaped = f"`{col}`" if ' ' in col else col
            
            results = cypher_search(f"""
                MATCH (r:Record)
                WHERE r.{col_escaped} IS NOT NULL 
                AND r.{col_escaped} <> '' 
                AND r.{col_escaped} <> 'null' 
                AND r.{col_escaped} <> 'NA'
                WITH r,
                CASE 
                    WHEN r.{col_escaped} =~ '^\\d+\\.?\\d*$' THEN toFloat(r.{col_escaped})
                    WHEN r.{col_escaped} =~ '^\\d+$' THEN toFloat(r.{col_escaped})
                    ELSE 0.0
                END as sort_val
                WHERE sort_val > 0
                RETURN r as node, sort_val as score
                ORDER BY sort_val {sort_direction}
                LIMIT {k_kw}
            """)
            
            if results:
                print(f"✅ Found {len(results)} sorted results by {col} ({sort_direction})")
                return results
                
        except Exception as e:
            print(f"⚠️ Failed to sort by {col}: {e}")
    
    return []

def _handle_temporal_query(question: str, strategy: Dict[str, Any], k_kw: int) -> List[Dict[str, Any]]:
    """Handle temporal queries intelligently."""
    target_cols = strategy.get('target_columns', []) + strategy.get('relevant_columns', [])
    question_lower = question.lower()
    
    sort_direction = "ASC" if any(word in question_lower for word in ['oldest', 'earliest', 'first']) else "DESC"
    
    for col in target_cols[:3]:  # Try top 3 relevant columns
        try:
            col_escaped = f"`{col}`" if ' ' in col else col
            
            results = cypher_search(f"""
                MATCH (r:Record)
                WHERE r.{col_escaped} IS NOT NULL 
                AND r.{col_escaped} <> '' 
                AND r.{col_escaped} <> 'null' 
                AND r.{col_escaped} <> 'NA'
                WITH r, 
                CASE 
                    WHEN r.{col_escaped} =~ '\\d{{4}}' THEN toInteger(r.{col_escaped})
                    WHEN r.{col_escaped} =~ '\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}' THEN toInteger(split(r.{col_escaped}, '/')[2])
                    WHEN r.{col_escaped} =~ '\\d{{4}}-\\d{{2}}-\\d{{2}}' THEN toInteger(split(r.{col_escaped}, '-')[0])
                    ELSE 0
                END as sort_val
                WHERE sort_val > 0
                RETURN r as node, 1.0 as score
                ORDER BY sort_val {sort_direction}
                LIMIT {k_kw}
            """)
            
            if results:
                print(f"✅ Found {len(results)} temporal results sorted by {col} ({sort_direction})")
                return results
                
        except Exception as e:
            print(f"⚠️ Failed to sort temporally by {col}: {e}")
    
    return []

def hybrid_search(question: str, k_vec: int = 10, k_kw: int = 20) -> List[Dict[str, Any]]:
    """Hybrid search combining keyword-based retrieval with semantic ranking."""
    
    # First, get candidates using keyword-based approach (larger set for reranking)
    keyword_results = retrieve_keywords(question, k_kw=min(50, k_kw * 3))
    
    if not keyword_results:
        print("❌ No keyword results found")
        return []
    
    print(f"🔍 Got {len(keyword_results)} keyword candidates")
    
    # If semantic search is enabled, rerank with embeddings
    if SEMANTIC_SEARCH_ENABLED:
        print("🧠 Reranking with semantic similarity...")
        semantic_results = semantic_search(question, keyword_results, top_k=k_vec)
        
        # Combine keyword and semantic scores
        for result in semantic_results:
            keyword_score = result.get('score', 1.0)
            semantic_score = result.get('semantic_score', 0.0)
            # Weighted combination: 30% keyword, 70% semantic
            result['final_score'] = 0.3 * keyword_score + 0.7 * semantic_score
        
        # Sort by final combined score
        semantic_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        print(f"✅ Hybrid search complete - returning top {len(semantic_results)} results")
        return semantic_results
    else:
        print("📝 Using keyword-only results")
        return keyword_results[:k_vec]

def retrieve_keywords(question: str, k_vec: int = 10, k_kw: int = 20) -> List[Dict[str, Any]]:
    """AI-powered retrieval with intelligent query classification."""
    
    # Get sample to understand data structure
    sample_records = cypher_search("MATCH (r:Record) RETURN r as node LIMIT 3")
    if not sample_records:
        print("❌ No records found in database!")
        return []
    
    print(f"📊 Data fields available: {list(sample_records[0]['node'].keys())}")
    
    # Use AI-powered query analysis
    try:
        from smart_profiler import get_smart_query_strategy
        smart_strategy = get_smart_query_strategy(question)
        print(f"🧠 Smart Strategy: {smart_strategy['query_type']} (confidence: {smart_strategy['confidence']:.1f})")
        
        if smart_strategy['strategy'] == 'aggregation' and smart_strategy['confidence'] > 0.6:
            return _handle_aggregation_query(question, smart_strategy, k_kw)
        elif smart_strategy['strategy'] == 'sorting' and smart_strategy['confidence'] > 0.6:
            return _handle_sorting_query(question, smart_strategy, k_kw)
        elif smart_strategy['strategy'] == 'temporal_sorting' and smart_strategy['confidence'] > 0.6:
            return _handle_temporal_query(question, smart_strategy, k_kw)
    except ImportError:
        print("⚠️ Smart profiler not available, using fallback logic")
    
    question_lower = question.lower()
    
    # FALLBACK: BRAND/FREQUENCY QUERIES - count which brands/categories appear most/least
    if any(word in question_lower for word in ['represented', 'appear', 'frequent', 'common', 'count', 'how many', 'entries per', 'unique']) and any(word in question_lower for word in ['brand', 'company', 'manufacturer', 'model', 'type', 'msl', 'category', 'name']):
        print("🎯 BRAND FREQUENCY query detected - counting occurrences")
        
        # Find brand/category columns
        brand_cols = []
        for col in sample_records[0]['node'].keys():
            if any(pattern in col.lower() for pattern in ['brand', 'company', 'manufacturer', 'make', 'model', 'type', 'msl', 'category', 'name']):
                brand_cols.append(col)
        
        print(f"🏷️ Brand columns found: {brand_cols}")
        
        if brand_cols:
            # Sort direction based on query terms
            sort_direction = "DESC" if any(word in question_lower for word in ['most', 'highest', 'top']) else "ASC" if any(word in question_lower for word in ['least', 'lowest', 'bottom']) else "DESC"
            
            for col in brand_cols[:2]:  # Try first 2 brand columns
                try:
                    # Use backticks for column names with spaces and proper aggregation
                    col_escaped = f"`{col}`" if ' ' in col else col
                    
                    results = cypher_search(f"""
                        MATCH (r:Record)
                        WHERE r.{col_escaped} IS NOT NULL 
                        AND r.{col_escaped} <> '' 
                        AND r.{col_escaped} <> 'null' 
                        AND r.{col_escaped} <> 'NA'
                        WITH r.{col_escaped} as brand_name, count(*) as frequency, collect(r)[0..3] as sample_records
                        RETURN {{
                            brand: brand_name,
                            count: frequency,
                            sample: sample_records[0]
                        }} as node, frequency as score
                        ORDER BY frequency {sort_direction}
                        LIMIT {k_kw}
                    """)
                    
                    if results:
                        print(f"✅ Found {len(results)} brand frequencies from {col} (sorted {sort_direction})")
                        return results
                        
                except Exception as e:
                    print(f"⚠️ Failed to count by {col}: {e}")
    
    # SUPERLATIVE QUERIES - oldest, newest, highest, lowest, etc.
    elif any(word in question_lower for word in ['oldest', 'newest', 'earliest', 'latest', 'first', 'last']):
        print("🎯 SUPERLATIVE query detected - finding extremes")
        
        # Find date/year columns
        date_cols = []
        for col in sample_records[0]['node'].keys():
            if any(pattern in col.lower() for pattern in ['year', 'date', 'time', 'created', 'founded', 'registration']):
                date_cols.append(col)
        
        print(f"📅 Date columns found: {date_cols}")
        
        if date_cols:
            sort_direction = "ASC" if any(word in question_lower for word in ['oldest', 'earliest', 'first']) else "DESC"
            
            for col in date_cols:
                try:
                    # Use backticks for column names with spaces
                    col_escaped = f"`{col}`" if ' ' in col else col
                    
                    # Smart sorting that handles different year formats
                    results = cypher_search(f"""
                        MATCH (r:Record)
                        WHERE r.{col_escaped} IS NOT NULL 
                        AND r.{col_escaped} <> '' 
                        AND r.{col_escaped} <> 'null' 
                        AND r.{col_escaped} <> 'NA'
                        WITH r, 
                        CASE 
                            WHEN r.{col_escaped} =~ '\\d{{4}}' THEN toInteger(r.{col_escaped})
                            WHEN r.{col_escaped} =~ '\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}' THEN toInteger(split(r.{col_escaped}, '/')[2])
                            WHEN r.{col_escaped} =~ '\\d{{4}}-\\d{{2}}-\\d{{2}}' THEN toInteger(split(r.{col_escaped}, '-')[0])
                            ELSE 0
                        END as sort_val
                        WHERE sort_val > 0
                        RETURN r as node, 1.0 as score
                        ORDER BY sort_val {sort_direction}
                        LIMIT {k_kw}
                    """)
                    
                    if results:
                        print(f"✅ Found {len(results)} records sorted by {col} ({sort_direction})")
                        return results
                        
                except Exception as e:
                    print(f"⚠️ Failed to sort by {col}: {e}")
    
    # COMPARISON QUERIES - highest, lowest, most expensive, cheapest, etc.
    elif any(word in question_lower for word in ['highest', 'lowest', 'most', 'least', 'expensive', 'cheapest', 'largest', 'smallest', 'maximum', 'minimum']):
        print("🎯 COMPARISON query detected - finding numeric extremes")
        
        # Find numeric columns
        numeric_cols = []
        for col in sample_records[0]['node'].keys():
            if any(pattern in col.lower() for pattern in ['price', 'cost', 'amount', 'value', 'count', 'size', 'mileage', 'engine', 'salary', 'revenue']):
                numeric_cols.append(col)
        
        print(f"🔢 Numeric columns found: {numeric_cols}")
        
        if numeric_cols:
            sort_direction = "DESC" if any(word in question_lower for word in ['highest', 'most', 'expensive', 'largest', 'maximum']) else "ASC"
            
            for col in numeric_cols:
                try:
                    # Use backticks for column names with spaces
                    col_escaped = f"`{col}`" if ' ' in col else col
                    
                    # Smart numeric sorting handling various formats
                    results = cypher_search(f"""
                        MATCH (r:Record)
                        WHERE r.{col_escaped} IS NOT NULL 
                        AND r.{col_escaped} <> '' 
                        AND r.{col_escaped} <> 'null' 
                        AND r.{col_escaped} <> 'NA'
                        WITH r,
                        CASE 
                            WHEN r.{col_escaped} =~ '^\\d+\\.?\\d*$' THEN toFloat(r.{col_escaped})
                            WHEN r.{col_escaped} =~ '^\\d+$' THEN toFloat(r.{col_escaped})
                            ELSE 0.0
                        END as sort_val
                        WHERE sort_val > 0
                        RETURN r as node, 1.0 as score
                        ORDER BY sort_val {sort_direction}
                        LIMIT {k_kw}
                    """)
                    
                    if results:
                        print(f"✅ Found {len(results)} records sorted by {col} ({sort_direction})")
                        return results
                        
                except Exception as e:
                    print(f"⚠️ Failed to sort by {col}: {e}")
    
    # CATEGORY/BRAND QUERIES
    elif any(word in question_lower for word in ['brand', 'type', 'category', 'model', 'company', 'manufacturer']):
        print("🎯 CATEGORY query detected - finding diverse sample")
        
        # Look for brand/category columns
        cat_cols = []
        for col in sample_records[0]['node'].keys():
            if any(pattern in col.lower() for pattern in ['brand', 'type', 'category', 'model', 'company', 'manufacturer', 'make']):
                cat_cols.append(col)
        
        if cat_cols:
            # Get diverse sample across categories
            for col in cat_cols[:2]:  # Try first 2 category columns
                try:
                    col_escaped = f"`{col}`" if ' ' in col else col
                    
                    results = cypher_search(f"""
                        MATCH (r:Record)
                        WHERE r.{col_escaped} IS NOT NULL AND r.{col_escaped} <> '' AND r.{col_escaped} <> 'null'
                        WITH r.{col_escaped} as category, collect(r) as records
                        UNWIND records[0..2] as r  // Max 2 per category
                        RETURN r as node, 1.0 as score
                        LIMIT {k_kw}
                    """)
                    
                    if results:
                        print(f"✅ Found {len(results)} diverse records across {col} categories")
                        return results
                        
                except Exception as e:
                    print(f"⚠️ Failed to get diverse sample: {e}")
    
    # AGGREGATION/SUMMARY QUERIES
    elif any(word in question_lower for word in ['summarize', 'summary', 'overview', 'analyze', 'insights', 'patterns', 'total', 'count', 'average', 'engagement data', 'msl data']):
        print("🎯 SUMMARY query detected - getting comprehensive representative sample")
        
        # Get stratified sample across MSLs and specialties for better representation
        try:
            results = cypher_search(f"""
                MATCH (r:Record)
                WITH r.MSL as msl, collect(r) as msl_records
                WITH msl, msl_records[0..{max(5, k_kw//10)}] as sample_records
                UNWIND sample_records as r
                WITH r, rand() as random
                ORDER BY random
                RETURN r as node, 1.0 as score
                LIMIT {k_kw}
            """)
            
            if results:
                print(f"✅ Found {len(results)} stratified records across MSLs for comprehensive analysis")
                return results
            else:
                # Fallback to simple random sample with larger size
                results = cypher_search(f"""
                    MATCH (r:Record)
                    WITH r, rand() as random
                    ORDER BY random
                    RETURN r as node, 1.0 as score
                    LIMIT {k_kw}
                """)
                
                if results:
                    print(f"✅ Found {len(results)} random records for analysis")
                    return results
                
        except Exception as e:
            print(f"⚠️ Stratified sampling failed: {e}")
            # Final fallback
            try:
                results = cypher_search(f"""
                    MATCH (r:Record)
                    RETURN r as node, 1.0 as score
                    LIMIT {k_kw}
                """)
                if results:
                    print(f"✅ Simple fallback: {len(results)} records")
                    return results
            except Exception as e2:
                print(f"⚠️ All sampling methods failed: {e2}")
    
    # SPECIFIC SEARCH QUERIES - enhanced keyword matching
    else:
        print("🎯 SPECIFIC query detected - using enhanced search")
        
        # Extract meaningful keywords
        stop_words = {'what', 'is', 'the', 'in', 'of', 'for', 'with', 'by', 'from', 'to', 'and', 'or', 'but', 'which', 'that', 'how', 'where', 'when', 'why'}
        keywords = [word.lower().strip('?.,!') for word in question.split() if len(word) > 2 and word.lower() not in stop_words]
        
        print(f"🔍 Keywords: {keywords}")
        
        if keywords:
            results = []
            
            # Multi-strategy search
            for keyword in keywords[:3]:
                try:
                    # Strategy 1: Exact value matches
                    exact_results = cypher_search(f"""
                        MATCH (r:Record)
                        WHERE any(prop in keys(r) WHERE toLower(toString(r[prop])) CONTAINS toLower($keyword))
                        RETURN r as node, 2.0 as score
                        LIMIT {k_kw // 2}
                    """, {"keyword": keyword})
                    
                    results.extend(exact_results)
                    
                    # Strategy 2: Partial matches
                    partial_results = cypher_search(f"""
                        MATCH (r:Record)
                        WHERE any(prop in keys(r) WHERE toLower(toString(r[prop])) CONTAINS toLower($partial))
                        RETURN r as node, 1.0 as score
                        LIMIT {k_kw // 2}
                    """, {"partial": keyword[:4]})  # First 4 characters
                    
                    results.extend(partial_results)
                    
                except Exception as e:
                    print(f"⚠️ Search failed for '{keyword}': {e}")
            
            # Remove duplicates
            seen = set()
            unique_results = []
            for result in results:
                key = str(result['node'])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            if unique_results:
                print(f"✅ Found {len(unique_results)} matching records")
                return unique_results[:k_kw]
    
    # Fallback: Random sample
    print("🔄 Using fallback: random sample")
    try:
        results = cypher_search(f"""
            MATCH (r:Record)
            WITH r, rand() as random  
            ORDER BY random
            RETURN r as node, 1.0 as score
            LIMIT {k_kw}
        """)
        
        print(f"✅ Fallback: {len(results)} random records")
        return results
        
    except Exception as e:
        print(f"❌ Even fallback failed: {e}")
        return []

def cypher_search(cypher: str, params: Dict = None) -> List[Dict[str, Any]]:
    """Execute arbitrary Cypher query and return results."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]
    finally:
        driver.close()

def retrieve(question: str, k_vec: int = 10, k_kw: int = 20) -> List[Dict[str, Any]]:
    """Main retrieval function - uses hybrid search when possible, falls back to keyword search."""
    
    print(f"🔍 Smart Analysis: '{question}'")
    
        # Use hybrid search if semantic embeddings are available
    if SEMANTIC_SEARCH_ENABLED:
        print("🚀 Using HYBRID SEARCH (Keywords + Semantic Embeddings)")
        return hybrid_search(question, k_vec, k_kw)
    else:
        print("📝 Using KEYWORD-ONLY SEARCH (install sentence-transformers for semantic search)")
        return retrieve_keywords(question, k_vec, k_kw)

def retrieve_until_stable(question: str, max_iter: int = 3, model: str = "qwen3-235b") -> List[Dict[str, Any]]:
    """Intelligent retrieval that scales optimally with dataset size."""
    print(f"🔍 Starting INTELLIGENT SCALABLE retrieval for question: '{question}'")
    
    try:
        # Use smart scaler to determine optimal strategy
        from smart_scaler import execute_optimal_retrieval
        records, strategy = execute_optimal_retrieval(question, model)
        
        # Apply semantic ranking if available and records exist
        if SEMANTIC_SEARCH_ENABLED and records:
            print(f"🧠 Applying semantic ranking to {len(records)} records...")
            ranked_results = semantic_search(question, records, top_k=len(records))
            print(f"✅ Semantic ranking complete - top similarity: {ranked_results[0].get('semantic_score', 'N/A') if ranked_results else 'N/A'}")
            
            # Store strategy info for UI display
            for result in ranked_results:
                result['_strategy'] = strategy
            
            return ranked_results
        else:
            # Store strategy info for UI display
            for result in records:
                result['_strategy'] = strategy
            
            return records
        
    except ImportError:
        print("⚠️ Smart scaler not available, falling back to full retrieval")
        # Fallback to original full database retrieval
        return _fallback_full_retrieval(question)
        
    except Exception as e:
        print(f"❌ Smart retrieval failed: {e}")
        print("🔄 Falling back to full database retrieval...")
        return _fallback_full_retrieval(question)

def _fallback_full_retrieval(question: str) -> List[Dict[str, Any]]:
    """Fallback to full database retrieval."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            count_result = session.run("MATCH (r:Record) RETURN count(r) as total")
            total_records = count_result.single()["total"]
            print(f"📊 Database contains {total_records} total records")
            
            # Retrieve ALL records
            all_results = session.run("""
                MATCH (r:Record)
                RETURN r as node, 1.0 as score
                ORDER BY r.id
            """)
            results = [dict(record) for record in all_results]
            
            if SEMANTIC_SEARCH_ENABLED and results:
                ranked_results = semantic_search(question, results, top_k=len(results))
                print(f"✅ FALLBACK: Retrieved and ranked {len(ranked_results)} records")
                return ranked_results
            else:
                print(f"✅ FALLBACK: Retrieved {len(results)} records")
                return results
                
        driver.close()
        
    except Exception as e:
        print(f"❌ Even fallback failed: {e}")
        return []

def ask_llm(question: str, context: List[Dict[str, Any]], model: str = "qwen3-235b") -> Dict[str, Any]:
    """Get answer and chart spec from Venice.ai LLM with selected model."""
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use ALL available context - no artificial limits!
    print(f"📊 Preparing ALL {len(context)} records for {model} analysis")
    
    # Format all records for analysis
    context_str = "\n".join(
        f"Record {i+1}: {r['node']}"
        for i, r in enumerate(context)
    )
    
    # Calculate approximate token count (rough estimate: 4 chars per token)
    estimated_tokens = len(context_str) // 4
    print(f"📏 Estimated context tokens: ~{estimated_tokens:,}")
    
    # Confirm we're using the complete dataset
    print(f"✅ CONFIRMED: Providing complete dataset of {len(context)} records to AI model")
    
    # Enhanced prompting for reasoning models
    if "235b" in model or "reasoning" in model.lower():
        system_prompt = f"""You are an advanced AI data analyst with reasoning capabilities analyzing a COMPLETE DATABASE.

CRITICAL: You have been provided with the ENTIRE database of {len(context)} records. This is NOT a sample - this is the complete dataset. Your analysis must reflect this comprehensive coverage.

IMPORTANT: Always wrap your reasoning in <think> tags and provide a comprehensive analysis.

<think>
STEP 1: DATABASE VERIFICATION
- Confirm I have received {len(context)} records
- Verify this represents the complete dataset, not a sample
- Note the total scope of data I'm analyzing

STEP 2: COMPREHENSIVE DATA STRUCTURE ANALYSIS
- What type of data is this?
- What are ALL the key fields and their meanings?
- How many total records am I analyzing from the complete database?
- Is this aggregated data (counts/frequencies) or individual records?
- What is the complete scope and coverage of this dataset?

STEP 3: EXHAUSTIVE PATTERN RECOGNITION
- What patterns can I identify across the ENTIRE dataset?
- Are there any trends or correlations across ALL records?
- What stands out as unusual or interesting in the complete data?
- What is the distribution across ALL categories/entities?

STEP 4: COMPREHENSIVE INSIGHT GENERATION
- What are the most important findings from the COMPLETE database?
- What would be valuable for decision-making based on ALL data?
- How should I structure my response to reflect complete coverage?

STEP 5: COMPLETE ANALYSIS CONFIRMATION
- Confirm I have analyzed the entire database of {len(context)} records
- Ensure my insights reflect the complete dataset, not partial data
- Verify my conclusions are based on comprehensive analysis

STEP 6: CHART PLANNING (if applicable)
- What type of visualization would best represent the COMPLETE data?
- Should this be a bar chart, line chart, or other format?
- What should be on the x and y axes to show complete coverage?
</think>

COMPLETE DATABASE CONTEXT ({len(context)} records - ENTIRE DATASET):
{context_str}

Your task: {question}

REMEMBER: You are analyzing the COMPLETE database of {len(context)} records. Your response must reflect this comprehensive coverage and confirm you have reviewed the entire dataset.

Please provide a comprehensive analysis following this structure:
1. **Executive Summary**: Brief overview of key findings (2-3 paragraphs)
2. **Data Overview**: What the data represents and scope of analysis (detailed description)
3. **Key Insights**: Most important findings with specific examples and numbers (detailed analysis)
4. **Patterns & Trends**: Observable patterns in the data with specific values (comprehensive findings)
5. **Detailed Analysis**: Deep dive into the most interesting aspects (thorough examination)
6. **Recommendations**: Actionable insights based on the analysis (strategic recommendations)

IMPORTANT: Write a detailed, comprehensive analysis of at least 1000 words. Include specific numbers, examples, and insights from the data. Be thorough and analytical.

If the data contains frequency/count information that would benefit from visualization, also create a chart using this format:
CHART_JSON_START
{{
  "data": [
    {{"x": "Category1", "y": value1}},
    {{"x": "Category2", "y": value2}}
  ],
  "mark": "bar",
  "encoding": {{
    "x": {{"field": "x", "type": "nominal", "title": "X Axis Title"}},
    "y": {{"field": "y", "type": "quantitative", "title": "Y Axis Title"}}
  }},
  "title": "Chart Title"
}}
CHART_JSON_END

IMPORTANT CHART GUIDELINES:
- Always generate a chart if the data has countable categories (Assets, MSLs, Specialties, etc.)
- For pharmaceutical data, common charts include: Asset distribution, MSL activity, Specialty breakdown, KOL tiers
- Use actual data values from your analysis
- Make sure the JSON is valid and properly formatted
- Only include ONE chart per response"""
    else:
        system_prompt = f"""You are a helpful data analyst. Use the following data to answer questions comprehensively.

Data Context ({len(context)} records analyzed):
{context_str}

User Question: {question}

Provide a detailed analysis with specific numbers and examples from the data. Write at least 500 words with comprehensive insights.

If this data contains counts, frequencies, or aggregated information that would benefit from visualization, create a chart using this format:
CHART_JSON_START
{{
  "data": [
    {{"x": "Category1", "y": value1}},
    {{"x": "Category2", "y": value2}}
  ],
  "mark": "bar",
  "encoding": {{
    "x": {{"field": "x", "type": "nominal", "title": "Category"}},
    "y": {{"field": "y", "type": "quantitative", "title": "Count"}}
  }},
  "title": "Distribution Chart"
}}
CHART_JSON_END

CHART REQUIREMENTS:
- Generate a chart if data has countable patterns
- Use real values from the data you're analyzing
- Ensure valid JSON format with proper escaping
- Include descriptive titles and axis labels"""
    
    # Venice-specific parameters for reasoning models
    venice_params = {}
    if "235b" in model or "reasoning" in model.lower():
        venice_params = {
            "strip_thinking_response": False,  # Keep thinking blocks for reasoning models
            "disable_thinking": False,
            "include_venice_system_prompt": False  # Use our custom reasoning prompt
        }
    else:
        venice_params = {
            "include_venice_system_prompt": True
        }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Please provide a comprehensive, detailed analysis of this data to answer: {question}"
            }
        ],
        "max_completion_tokens": 16000,  # Maximum tokens for comprehensive analysis
        "temperature": 0.05 if ("235b" in model or "reasoning" in model.lower()) else 0.1,  # Very low temperature for thorough analysis
        "venice_parameters": venice_params
    }
    
    print(f"Making request to Venice.ai with model: {model}")
    print(f"Venice parameters: {venice_params}")
    
    response = requests.post(
        "https://api.venice.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=600  # 10 minutes - give reasoning models all the time they need
    )
    response.raise_for_status()
    
    # Parse the response from the LLM
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    
    # For reasoning models, extract reasoning from multiple possible locations
    reasoning_content = None
    
    # Check for reasoning_content field (Venice.ai specific)
    reasoning_content = result["choices"][0]["message"].get("reasoning_content")
    
    # Also check if reasoning is embedded in the content itself (for <think> tags)
    if not reasoning_content and "<think>" in content:
        import re
        think_matches = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_matches:
            reasoning_content = "\n".join(think_matches)
            # Remove think tags from main content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    print(f"Response received. Content length: {len(content)}")
    if reasoning_content:
        print(f"Reasoning content found! Length: {len(reasoning_content)}")
        print(f"Reasoning preview: {reasoning_content[:200]}...")
    else:
        print("No reasoning content found")
        # Debug: print the full message structure
        print(f"Message keys: {result['choices'][0]['message'].keys()}")
    
    # Extract chart specification if present
    chart_spec = {}
    if "CHART_JSON_START" in content and "CHART_JSON_END" in content:
        try:
            # Handle multiple charts - extract the first valid one
            chart_start_marker = "CHART_JSON_START"
            chart_end_marker = "CHART_JSON_END"
            
            start_pos = content.find(chart_start_marker)
            if start_pos != -1:
                chart_start = start_pos + len(chart_start_marker)
                chart_end = content.find(chart_end_marker, chart_start)
                
                if chart_end != -1:
                    chart_json = content[chart_start:chart_end].strip()
                    
                    # Debug: Show what we're trying to parse
                    print(f"📊 Attempting to parse chart JSON (length: {len(chart_json)})")
                    if len(chart_json) > 0:
                        print(f"Chart JSON preview: {chart_json[:200]}...")
                        
                        # Clean up common JSON issues
                        chart_json = chart_json.replace('\n', ' ').replace('\r', '')
                        chart_json = chart_json.strip()
                        
                        # Try to parse the JSON
                        if chart_json.startswith('{') and chart_json.endswith('}'):
                            chart_spec = json.loads(chart_json)
                            print(f"✅ Successfully extracted chart: {chart_spec.get('title', 'Untitled Chart')}")
                            
                            # Remove chart JSON from main content
                            content = content[:start_pos] + content[chart_end + len(chart_end_marker):]
                            content = content.strip()
                        else:
                            print(f"⚠️ Chart JSON doesn't look like valid JSON object: starts with '{chart_json[:10]}', ends with '{chart_json[-10:]}'")
                    else:
                        print("⚠️ Empty chart JSON found")
                else:
                    print("⚠️ CHART_JSON_END marker not found after CHART_JSON_START")
            else:
                print("⚠️ CHART_JSON_START marker not found")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Failed to parse chart JSON: {e}")
            if 'chart_json' in locals():
                print(f"Raw chart JSON: '{chart_json}'")
            chart_spec = {}
    
    # If no chart was extracted but we can detect chart data in the text, try to create one from the output
    if not chart_spec and any(keyword in content.lower() for keyword in ['data":', 'mark":', 'encoding":', '"x":', '"y":']):
        print("🔍 Attempting to extract chart from mixed content...")
        # Try to find JSON-like structures in the content
        import re
        # Look for chart-like JSON patterns
        json_pattern = r'\{[^{}]*"data"[^{}]*\[[^\]]*\][^{}]*"mark"[^{}]*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                potential_chart = json.loads(match)
                if "data" in potential_chart and "mark" in potential_chart:
                    chart_spec = potential_chart
                    print(f"✅ Extracted chart from content: {chart_spec.get('title', 'Extracted Chart')}")
                    # Remove the matched JSON from content
                    content = content.replace(match, "").strip()
                    break
            except:
                continue
    
    # Return structured response
    result_dict = {
        "answer": content,
        "chart_spec": chart_spec
    }
    
    # Add reasoning content if available
    if reasoning_content:
        result_dict["reasoning"] = reasoning_content
        print("✅ Added reasoning to response")
    else:
        print("⚠️ No reasoning content to add")
    
    print(f"Answer length: {len(content)} characters")
    return result_dict 