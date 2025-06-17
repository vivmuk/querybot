import streamlit as st
import pandas as pd
import altair as alt
import time
from ingest import load_csv
from rag import retrieve_until_stable, ask_llm
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

st.set_page_config(
    page_title="Graph-RAG Dashboard",
    page_icon="📊",
    layout="wide"
)

# Available models for selection
AVAILABLE_MODELS = {
    "qwen3-235b": "Qwen 3 235B (Reasoning) 🧠",
    "qwen3-4b": "Qwen 3 4B (Fast) ⚡",
    "mistral-31-24b": "Mistral 3.1 24B 🎯",
    "llama-3.2-3b": "Llama 3.2 3B (Fast) ⚡",
    "llama-3.3-70b": "Llama 3.3 70B 💪",
    "venice-uncensored": "Venice Uncensored (Default) 🔓"
}

def check_data_exists():
    """Check if data already exists in Neo4j."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (r:Record) RETURN count(r) as count")
            count = result.single()["count"]
            driver.close()
            return count > 0
    except Exception as e:
        st.error(f"Error checking database: {e}")
        return False

def get_record_count():
    """Get the total number of records in the database."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (r:Record) RETURN count(r) as count")
            count = result.single()["count"]
            driver.close()
            return count
    except Exception as e:
        st.error(f"Error getting record count: {e}")
        return 0

def clear_database():
    """Clear all data from Neo4j."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            driver.close()
        st.success("Database cleared successfully!")
        return True
    except Exception as e:
        st.error(f"Error clearing database: {e}")
        return False

st.title("📊 Graph-RAG Dashboard")
st.markdown("Upload your CSV file and ask questions about your data using AI!")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    selected_model = st.selectbox(
        "🤖 Choose AI Model:",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        index=0  # Default to qwen3-235b
    )
    
    # Show model info
    if "235b" in selected_model:
        st.success("🧠 **Reasoning Model Selected!**")
        st.info("This model provides step-by-step analysis and detailed insights.")
    elif "reasoning" in AVAILABLE_MODELS[selected_model].lower():
        st.info("🧠 This model supports advanced reasoning!")
    elif "fast" in AVAILABLE_MODELS[selected_model].lower():
        st.info("⚡ Fast model - optimized for speed!")
    
    # Database management
    st.header("🗄️ Database")
    data_exists = check_data_exists()
    
    if data_exists:
        record_count = st.empty()
        # Get actual count
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                result = session.run("MATCH (r:Record) RETURN count(r) as count")
                count = result.single()["count"]
                driver.close()
                record_count.success(f"✅ {count} records loaded")
        except:
            record_count.info("📊 Data exists in database")
        
        if st.button("🗑️ Clear Database"):
            clear_database()
            st.rerun()
    else:
        st.info("💾 No data in database")

# File upload section
st.header("📁 Data Upload")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Show file info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size} bytes"
    }
    st.json(file_details)
    
    # Check if we should load data
    should_load = not data_exists or st.button("🔄 Reload Data")
    
    if should_load:
        # Save uploaded file temporarily
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Show preview
        df_preview = pd.read_csv("temp.csv")
        st.subheader("📋 Data Preview")
        st.dataframe(df_preview.head(), use_container_width=True)
        
        # Load data with AI analysis
        with st.spinner("🧠 AI is analyzing your dataset..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Phase 1: AI Analysis
            status_text.text("🧠 Understanding dataset structure...")
            progress_bar.progress(0.2)
            time.sleep(0.5)
            
            # Phase 2: Loading
            status_text.text("📊 Loading and indexing data...")
            progress_bar.progress(0.6)
            
            # Actually load the data (includes AI analysis)
            load_csv("temp.csv")
            
            progress_bar.progress(1.0)
            status_text.text("✅ AI analysis and loading complete!")
        
        # Show AI insights
        try:
            from smart_profiler import current_data_analysis
            if current_data_analysis:
                st.success("🧠 **AI Dataset Analysis Complete!**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Data Type", current_data_analysis.get('data_type', 'Unknown').title())
                with col2:
                    entities = current_data_analysis.get('primary_entities', [])
                    st.metric("🎯 Key Entities", f"{len(entities)} identified")
                with col3:
                    key_cols = current_data_analysis.get('key_columns', [])
                    st.metric("📈 Key Columns", f"{len(key_cols)} found")
                
                # Show insights
                if entities:
                    st.info(f"**Primary Entities Detected**: {', '.join(entities)}")
                
                business_context = current_data_analysis.get('business_context', '')
                if business_context and business_context != 'General data analysis':
                    st.info(f"**Business Context**: {business_context}")
                
                st.success("✅ **Smart Query System Activated!** The AI now understands your data structure and will provide more accurate responses.")
        except ImportError:
            st.warning("⚠️ Smart profiler not available - using basic analysis")
        
        st.success("✅ Data loaded successfully!")
        st.rerun()

# Query interface
if data_exists:
    st.header("❓ Ask Questions About Your Data")
    
    # Generate smart example questions based on AI analysis
    example_questions = []
    try:
        from smart_profiler import current_data_analysis
        if current_data_analysis:
            data_type = current_data_analysis.get('data_type', '').lower()
            entities = current_data_analysis.get('primary_entities', [])
            categorical_cols = current_data_analysis.get('categorical_columns', [])
            
            # Generic starter
            example_questions.append("Summarize the key insights from this data")
            
            # Data type specific questions
            if data_type == 'pharma' or any('msl' in str(e).lower() for e in entities):
                example_questions.extend([
                    "Which MSLs have the most productive engagements?",
                    "What are the main therapeutic insights?",
                    "Show me engagement trends by specialty"
                ])
            elif data_type == 'automotive':
                example_questions.extend([
                    "Which car brands are most represented?",
                    "What's the most expensive car in the database?",
                    "Show me vehicle distribution by year"
                ])
            elif data_type == 'sales':
                example_questions.extend([
                    "Which products generate the most revenue?",
                    "Show me sales trends over time",
                    "Who are the top performing customers?"
                ])
            else:
                # Generic but smart questions based on columns
                if categorical_cols:
                    example_questions.append(f"Count entries by {categorical_cols[0]}")
                    if len(categorical_cols) > 1:
                        example_questions.append(f"Compare {categorical_cols[0]} vs {categorical_cols[1]}")
                
                example_questions.append("What are the most interesting patterns in this data?")
                
    except ImportError:
        # Fallback questions
        example_questions = [
            "Summarize the data",
            "What are the main insights?",
            "Show me key patterns",
            "Count entries by category",
            "What stands out in this dataset?"
        ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input("💬 Your question:", placeholder="Ask anything about your data...")
    
    with col2:
        st.markdown("**Examples:**")
        for i, example in enumerate(example_questions[:3]):
            if st.button(f"💡 {example}", key=f"example_{i}"):
                question = example
    
    if question:
        # Create columns for the response
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show animated loading for retrieval
            with st.spinner("🔍 Searching through your data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Retrieval phase
                status_text.text("🔍 Finding relevant information...")
                progress_bar.progress(0.3)
                
                context = retrieve_until_stable(question, model=selected_model)
                
                progress_bar.progress(0.6)
                status_text.text(f"📊 Found {len(context)} relevant records")
                time.sleep(0.5)
                
                if not context:
                    st.info("🤔 No matching data found. Try a different question.")
                else:
                    # AI Analysis phase
                    status_text.text(f"🤖 Analyzing with {AVAILABLE_MODELS[selected_model]}...")
                    progress_bar.progress(0.8)
                    
                    # Add reasoning animation for reasoning models
                    reasoning_container = None
                    if "235b" in selected_model:
                        reasoning_container = st.container()
                        with reasoning_container:
                            st.markdown("### 🧠 AI Reasoning Process")
                            reasoning_placeholder = st.empty()
                            reasoning_steps = [
                                "🔍 Examining data structure and content...",
                                "📊 Identifying key patterns and relationships...",
                                "💭 Applying analytical reasoning...",
                                "🎯 Formulating insights and conclusions...",
                                "📈 Generating comprehensive analysis..."
                            ]
                            for i, step in enumerate(reasoning_steps):
                                reasoning_placeholder.text(step)
                                time.sleep(0.4)
                    
                    # Get LLM response with selected model
                    response = ask_llm(question, context, model=selected_model)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display reasoning content if available
                    if reasoning_container and response.get("reasoning"):
                        with reasoning_container:
                            with st.expander("🧠 AI Reasoning Details", expanded=True):
                                st.markdown("**Model's Step-by-Step Thinking:**")
                                reasoning_text = response["reasoning"]
                                
                                # Format reasoning nicely
                                if "STEP" in reasoning_text:
                                    # Split into steps for better display
                                    steps = reasoning_text.split("STEP")
                                    for i, step in enumerate(steps):
                                        if step.strip():
                                            if i == 0:
                                                st.text_area("Initial Analysis", step.strip(), height=100)
                                            else:
                                                st.text_area(f"Step {i}", step.strip(), height=150)
                                else:
                                    st.text_area("AI Reasoning Process", reasoning_text, height=300)
                    
                    # Display results
                    st.subheader("🎯 Analysis Results")
                    
                    # Show intelligent scaling analysis info
                    total_records = get_record_count()
                    
                    # Get strategy info from context if available
                    strategy_info = None
                    if context and '_strategy' in context[0]:
                        strategy_info = context[0]['_strategy']
                    
                    if strategy_info:
                        if strategy_info['strategy'] == 'complete_analysis':
                            st.success(
                                f"🎯 **COMPLETE DATABASE ANALYSIS**: Analyzed ALL **{len(context)}** records in the database.\n"
                                f"✅ **100% Coverage**: This analysis reflects the entire dataset with no sampling.\n"
                                f"🔢 **Tokens Used**: ~{strategy_info['estimated_tokens']:,} tokens"
                            )
                        elif strategy_info['strategy'] == 'stratified_sampling':
                            st.info(
                                f"🎯 **INTELLIGENT STRATIFIED SAMPLING**: Analyzed **{len(context)}** carefully selected records.\n"
                                f"📊 **Coverage**: {strategy_info['coverage_percentage']:.1f}% of database with representative sampling across all entities.\n"
                                f"🔢 **Tokens Used**: ~{strategy_info['estimated_tokens']:,} tokens\n"
                                f"💡 **Why**: Dataset too large for complete analysis - using smart sampling for comprehensive insights."
                            )
                        elif strategy_info['strategy'] == 'chunked_analysis':
                            st.warning(
                                f"🎯 **CHUNKED ANALYSIS**: Analyzing **{len(context)}** records from chunk {strategy_info.get('current_chunk', 1)}/{strategy_info['chunks']}.\n"
                                f"📊 **Total Coverage**: Will analyze all {total_records} records across {strategy_info['chunks']} chunks.\n"
                                f"🔢 **Tokens Used**: ~{strategy_info['estimated_tokens']:,} tokens per chunk\n"
                                f"💡 **Why**: Very large dataset requires chunked processing for thorough analysis."
                            )
                    else:
                        # Fallback display
                        if len(context) == total_records:
                            st.success(
                                f"🎯 **COMPLETE DATABASE ANALYSIS**: Analyzed ALL **{len(context)}** records in the database.\n"
                                f"✅ **100% Coverage**: This analysis reflects the entire dataset with no sampling."
                            )
                        else:
                            st.info(
                                f"📈 Analyzed **{len(context)}** records out of **{total_records}** total records in database.\n"
                                f"💡 **Note**: Intelligent scaling system optimizes analysis based on dataset size."
                            )
                    
                    # Display the main answer first (qualitative analysis)
                    st.subheader("📋 Final Analysis")
                    
                    # For reasoning models, show that reasoning was used
                    if response.get("reasoning") and ("235b" in selected_model or "reasoning" in AVAILABLE_MODELS[selected_model].lower()):
                        st.info("🧠 **Enhanced Analysis**: This response was generated using advanced reasoning capabilities for deeper insights.")
                    
                    # Enhanced answer display with better formatting
                    answer_text = response["answer"]
                    
                    # Clean up any remaining think tags that might have leaked through
                    if "<think>" in answer_text:
                        import re
                        answer_text = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL).strip()
                    
                    # Debug: Show answer length
                    print(f"Answer length: {len(answer_text)} characters")
                    
                    # Display full answer with proper formatting
                    if len(answer_text) > 100:  # Only show if we have substantial content
                        # Check if it's structured analysis
                        if any(marker in answer_text for marker in ["**Executive Summary**", "**Data Overview**", "1.", "##"]):
                            # Structured analysis - use markdown for better formatting
                            st.markdown(answer_text)
                        elif len(answer_text) > 2000:
                            # Very long answer - use expandable sections
                            # Split into paragraphs
                            paragraphs = [p.strip() for p in answer_text.split('\n\n') if p.strip()]
                            if len(paragraphs) > 4:
                                # Show first 2 paragraphs, then expandable sections
                                for para in paragraphs[:2]:
                                    st.write(para)
                                with st.expander("📖 Continue Reading Full Analysis...", expanded=True):
                                    for para in paragraphs[2:]:
                                        st.write(para)
                            else:
                                st.markdown(answer_text)
                        else:
                            # Regular answer
                            st.markdown(answer_text)
                    else:
                        # Short answer - show with warning
                        st.warning("⚠️ **Short Response Detected**: The AI provided a brief response. This may indicate an issue with the query or data processing.")
                        if answer_text.strip():
                            st.markdown(answer_text)
                        else:
                            st.error("❌ **Empty Response**: The AI response was empty. Please try rephrasing your question.")
                            # Show debugging info
                            st.info("🔍 **Debug Info**: Please check if your environment variables are properly configured.")
                    
                    # Display chart if generated (after the qualitative analysis)
                    if response.get("chart_spec") and response["chart_spec"]:
                        st.subheader("📊 Generated Chart")
                        try:
                            chart_spec = response["chart_spec"]
                            
                            # Convert to format Streamlit can handle
                            if "data" in chart_spec and chart_spec["data"]:
                                import pandas as pd
                                import altair as alt
                                
                                # Create DataFrame from chart data
                                df_chart = pd.DataFrame(chart_spec["data"])
                                
                                # Create Altair chart based on mark type
                                if chart_spec.get("mark") == "bar":
                                    chart = alt.Chart(df_chart).mark_bar().encode(
                                        x=alt.X('x:N', title=chart_spec.get("encoding", {}).get("x", {}).get("title", "Category")),
                                        y=alt.Y('y:Q', title=chart_spec.get("encoding", {}).get("y", {}).get("title", "Value"))
                                    ).properties(
                                        title=chart_spec.get("title", "Generated Chart"),
                                        width=600,
                                        height=400
                                    )
                                elif chart_spec.get("mark") == "line":
                                    chart = alt.Chart(df_chart).mark_line(point=True).encode(
                                        x=alt.X('x:O', title=chart_spec.get("encoding", {}).get("x", {}).get("title", "Category")),
                                        y=alt.Y('y:Q', title=chart_spec.get("encoding", {}).get("y", {}).get("title", "Value"))
                                    ).properties(
                                        title=chart_spec.get("title", "Generated Chart"),
                                        width=600,
                                        height=400
                                    )
                                else:
                                    # Default to bar chart
                                    chart = alt.Chart(df_chart).mark_bar().encode(
                                        x='x:N',
                                        y='y:Q'
                                    ).properties(
                                        title=chart_spec.get("title", "Generated Chart")
                                    )
                                
                                st.altair_chart(chart, use_container_width=True)
                                
                                # Show chart data
                                with st.expander("📊 Chart Data", expanded=False):
                                    st.dataframe(df_chart)
                            else:
                                st.info("📊 Chart specification was generated but contains no data to display.")
                                
                        except Exception as e:
                            st.error(f"❌ Error displaying chart: {str(e)}")
                            # Show raw chart spec for debugging
                            with st.expander("🔍 Debug: Raw Chart Specification"):
                                st.json(chart_spec)
        
        with col2:
            if context:
                st.markdown("### 📋 Data Sources")
                st.info(f"📊 **{len(context)}** records analyzed")
                st.info(f"🤖 Model: **{AVAILABLE_MODELS[selected_model]}**")
                
                # Show explainability details
                with st.expander("🔍 Search Explainability", expanded=False):
                    st.markdown("**How we found these results:**")
                    st.markdown("- 🔍 **Search Type**: Hybrid (Keywords + Semantic)")
                    st.markdown("- ⚖️ **Scoring**: 30% keyword relevance + 70% semantic similarity")
                    st.markdown("- 🎯 **Query Understanding**: Auto-detected query type from your question")
                    st.markdown("- 📊 **Data Strategy**: Smart column detection and targeted search")
                    
                    # Show top results with scores if available
                    if context and any('score' in item for item in context):
                        st.markdown("**📈 Top Results by Relevance Score:**")
                        for i, item in enumerate(context[:5]):
                            if i < 5:  # Show top 5
                                node = item.get('node', {})
                                
                                # Get various score types
                                semantic_score = item.get('semantic_score', item.get('score', 0))
                                keyword_score = item.get('keyword_score', item.get('score', 0))
                                
                                # Show key identifying info from the record
                                identifier = ""
                                for key in ['Brand', 'Model', 'Year', 'Name', 'Title', 'ID', 'MSL', 'HCP', 'Asset']:
                                    if key in node and str(node[key]).strip() and str(node[key]) != 'null':
                                        identifier = f"{str(node[key])[:30]}"
                                        break
                                
                                if not identifier:
                                    # Fallback to first non-empty field
                                    for key, value in node.items():
                                        if str(value).strip() and str(value) != 'null':
                                            identifier = f"{key}: {str(value)[:20]}"
                                            break
                                
                                st.text(f"#{i+1}: {identifier} | Semantic: {semantic_score:.3f} | Keyword: {keyword_score:.3f}")
                    else:
                        st.markdown("**📊 Analysis Coverage:**")
                        st.text(f"• Complete database analysis of {len(context)} records")
                        st.text(f"• Intelligent scaling strategy applied")
                        st.text(f"• Comprehensive insights generated")
                
                # Show sample of data being analyzed
                with st.expander("🔍 View Sample Data", expanded=False):
                    if context:
                        # Show a few sample records with better formatting
                        for i, item in enumerate(context[:3]):
                            with st.container():
                                st.write(f"**Sample Record {i+1}:**")
                                # Format the record data nicely
                                record_data = item["node"]
                                formatted_data = {}
                                for key, value in record_data.items():
                                    if str(value).strip() and str(value) != 'null':
                                        formatted_data[key] = value
                                st.json(formatted_data)
                    else:
                        st.info("No sample data available")
else:
    st.info("👆 Please upload a CSV file to get started!") 