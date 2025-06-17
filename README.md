# 🧠 Graph-RAG Dashboard

An intelligent data analysis platform that combines **Graph databases**, **RAG (Retrieval-Augmented Generation)**, and **AI reasoning** to provide comprehensive insights from your CSV data.

## ✨ Features

- **🔍 Intelligent Data Analysis**: Upload CSV files and get AI-powered insights
- **🧠 Advanced Reasoning**: Uses Venice.ai's Qwen3 235B model with step-by-step thinking
- **📊 Interactive Charts**: Automatic visualization generation based on data patterns
- **🎯 Smart Scaling**: Handles datasets from 100 to 10,000+ records intelligently
- **🔍 Semantic Search**: BGE-M3 embeddings for accurate data retrieval
- **📈 Graph Database**: Neo4j for efficient relationship mapping
- **🎨 Modern UI**: Clean Streamlit interface with explainable AI features

## 🚀 Live Demo

**[Try it now on Streamlit Cloud →](https://your-app-url.streamlit.app)**

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Database**: Neo4j (cloud)
- **AI Models**: Venice.ai (Qwen3 235B + BGE-M3 embeddings)
- **Visualization**: Altair charts
- **Backend**: Python with intelligent scaling algorithms

## 📋 What It Does

1. **📤 Upload CSV Data**: Drop any CSV file (pharmaceutical, sales, automotive, etc.)
2. **🧠 AI Analysis**: Automatically profiles data and applies smart reasoning
3. **🔍 Ask Questions**: Natural language queries about your data
4. **📊 Get Insights**: Comprehensive analysis with charts and explanations
5. **🎯 Explainable AI**: See exactly how the AI found and analyzed your data

## 🎯 Perfect For

- **📊 Data Scientists**: Quick exploratory data analysis
- **💼 Business Analysts**: Generate insights from business data
- **🏥 Pharmaceutical**: MSL engagement analysis, HCP insights
- **🚗 Automotive**: Vehicle data analysis and trends
- **📈 Sales Teams**: Customer and product analysis

## 🌟 Key Innovations

### 🧠 **Smart Scaling Technology**
- **≤2000 records**: Complete analysis of entire dataset
- **2000-5000 records**: Intelligent stratified sampling  
- **>5000 records**: Chunked analysis with comprehensive coverage

### 🎯 **Advanced RAG System**
- **Hybrid Search**: Combines keyword and semantic similarity
- **Context Optimization**: Automatically adjusts analysis depth
- **Smart Query Classification**: Detects aggregation, sorting, and filtering needs

### 🔍 **Explainable AI**
- **Reasoning Display**: See the AI's step-by-step thinking process
- **Search Transparency**: Understand how results were found
- **Strategy Explanation**: Know why specific analysis methods were chosen

## 🚀 Quick Start

### Option 1: Use Streamlit Cloud (Recommended)
1. Visit the live demo above
2. Upload your CSV file
3. Start asking questions!

### Option 2: Run Locally
```bash
git clone https://github.com/yourusername/graph-rag-dashboard.git
cd graph-rag-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## 🔐 Environment Variables

For deployment, set these in your Streamlit Cloud secrets:

```toml
NEO4J_URI = "your-neo4j-uri"
NEO4J_USER = "your-username" 
NEO4J_PASSWORD = "your-password"
VENICE_API_KEY = "your-venice-api-key"
```

## 📝 Sample Questions

- "Give me a qualitative analysis of the entire dataset"
- "Which MSLs have the most productive engagements?"
- "Show me the distribution of assets in the database"
- "What are the main therapeutic insights?"
- "Analyze engagement patterns by specialty"

## 🏗️ Architecture

```
📤 CSV Upload → 🔍 Smart Profiling → 🗄️ Neo4j Storage → 🧠 AI Analysis → 📊 Results + Charts
                                                              ↑
                                                    Venice.ai (Qwen3 235B + BGE-M3)
```

## 🎯 Intelligent Features

- **Auto-detects data types** (pharmaceutical, automotive, sales, etc.)
- **Smart column classification** (categorical, numerical, temporal)
- **Adaptive analysis strategies** based on dataset size
- **Context-aware questioning** with example suggestions
- **Real-time explainability** for all AI decisions

## 🤝 Contributing

Contributions welcome! This project showcases advanced RAG techniques and AI reasoning.

## 📄 License

MIT License - Feel free to use for your own projects!

---

**Built with ❤️ using Venice.ai, Neo4j, and Streamlit** 