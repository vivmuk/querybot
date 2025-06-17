# 🚀 Query Bot Deployment Guide

## 📝 Overview
Your Query Bot is a sophisticated Graph-RAG system with Venice.ai integration, Neo4j database, and intelligent scaling. Here are the best deployment options:

## 🏆 Recommended: Streamlit Cloud (FREE & EASY)

### Why Streamlit Cloud?
- ✅ **FREE** for public repos
- ✅ **Zero configuration** needed
- ✅ **Automatic scaling**
- ✅ **Perfect for Streamlit apps**
- ✅ **Built-in secrets management**

### Steps to Deploy:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Graph-RAG Query Bot"
   git branch -M main
   git remote add origin https://github.com/yourusername/query-bot.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo
   - Main file: `app.py`
   - Click "Deploy"

3. **Add Secrets** (in Streamlit Cloud dashboard):
   ```toml
   # .streamlit/secrets.toml format
   NEO4J_URI = "neo4j+s://e10687d6.databases.neo4j.io"
   NEO4J_USER = "neo4j"
   NEO4J_PASSWORD = "t5KwbwanUivhP7fe29DCAwPYwqbRbKHpMFtxwAmZ-ZI"
   VENICE_API_KEY = "SjXF4gSRAvPU_nz99kSeInMOkoZJCNQpSDpWMoJHJ0"
   ```

## 🌐 Alternative: Heroku

### Steps:
1. **Install Heroku CLI**
2. **Create Heroku App**:
   ```bash
   heroku create your-query-bot
   ```
3. **Set Environment Variables**:
   ```bash
   heroku config:set NEO4J_URI="neo4j+s://e10687d6.databases.neo4j.io"
   heroku config:set NEO4J_USER="neo4j"
   heroku config:set NEO4J_PASSWORD="t5KwbwanUivhP7fe29DCAwPYwqbRbKHpMFtxwAmZ-ZI"
   heroku config:set VENICE_API_KEY="SjXF4gSRAvPU_nz99kSeInMOkoZJCNQpSDpWMoJHJ0"
   ```
4. **Deploy**:
   ```bash
   git push heroku main
   ```

## 🔧 Alternative: Railway

### Steps:
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repo
3. Add environment variables
4. Deploy automatically

## ⚡ Alternative: Render

### Steps:
1. Go to [render.com](https://render.com)
2. Create new web service
3. Connect GitHub repo
4. Add environment variables
5. Deploy

## 🏠 Self-Hosting Options

### Docker Deployment:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Local Production:
```bash
# Install dependencies
pip install -r requirements.txt

# Run with production settings
streamlit run app.py --server.port=80 --server.address=0.0.0.0
```

## 🔐 Environment Variables Required

All deployment platforms need these variables:

```bash
NEO4J_URI=neo4j+s://e10687d6.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=t5KwbwanUivhP7fe29DCAwPYwqbRbKHpMFtxwAmZ-ZI
VENICE_API_KEY=SjXF4gSRAvPU_nz99kSeInMOkoZJCNQpSDpWMoJHJ0
```

## 🎯 Best Choice: Streamlit Cloud

**For your use case, I strongly recommend Streamlit Cloud because:**
- Free and reliable
- Zero configuration
- Perfect for Streamlit apps
- Built-in secrets management
- Automatic updates from GitHub
- Great performance for Graph-RAG apps

## 🧪 Testing Your Deployment

Once deployed, test these features:
1. **Database Connection**: Check if Neo4j connects
2. **Embeddings**: Test Venice.ai semantic search
3. **Smart Scaling**: Try with different query types
4. **MSL Counting**: Test "how many entries per unique MSL"
5. **Complex Queries**: Test reasoning capabilities

## 📱 Production URLs

After deployment, your app will be available at:
- **Streamlit Cloud**: `https://yourapp.streamlit.app`
- **Heroku**: `https://your-query-bot.herokuapp.com`
- **Railway**: `https://your-query-bot.up.railway.app`
- **Render**: `https://your-query-bot.onrender.com`

## 🔧 Troubleshooting

### Common Issues:
1. **Memory Limits**: Use smart scaling (already implemented)
2. **Timeout Issues**: Increase timeouts in config
3. **Dependencies**: Ensure all packages in requirements.txt
4. **Secrets**: Double-check environment variables

### Performance Optimization:
- App uses intelligent scaling for large datasets
- Semantic search caches embeddings
- Neo4j connection pooling
- Smart query classification

Your Graph-RAG system is production-ready! 🚀 