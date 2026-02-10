# 🎫 SmartSupport AI

> **AI-powered customer support ticket routing and resolution using RAG (Retrieval-Augmented Generation)**

SmartSupport AI automatically assigns support tickets to the right team and suggests resolutions by learning from historical ticket data. Built with FastAPI, Endee vector database, and Llama 3 LLM.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Development](#-development)
- [License](#-license)

---

## ✨ Features

### Core Features
- **🎯 Intelligent Team Assignment** - Automatically routes tickets to the right support team using vector similarity search
- **💡 Auto-Resolution Suggestions** - Provides resolution suggestions based on similar historical tickets
- **🤖 RAG-Powered Responses** - Uses Llama 3 LLM with retrieval augmentation for contextual answers
- **⚡ Fast Vector Search** - Sub-millisecond similarity search using Endee (HNSW algorithm)
- **📊 Confidence Scoring** - Provides confidence metrics for team assignments
- **🌐 Modern Web UI** - Clean, responsive interface built with HTML/CSS/JS

### Technical Features
- **Direct HTTP Client** - Native Endee integration without SDK dependencies
- **Batch Processing** - Efficient bulk data ingestion with progress tracking
- **Connection Pooling** - Optimized HTTP sessions with automatic retries
- **Error Handling** - Comprehensive validation and error recovery
- **Health Monitoring** - Built-in diagnostics for all system components

---

## 🏗️ Architecture

```
┌─────────────┐
│   Web UI    │ ← HTML/CSS/JS Interface
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  FastAPI    │ ← REST API Server (Python)
└──────┬──────┘
       │
       ├──→ SentenceTransformers ← all-MiniLM-L6-v2 (384d embeddings)
       │
       ├──→ Endee Vector DB      ← HNSW index (cosine similarity)
       │
       └──→ Ollama (Llama 3)     ← LLM for RAG responses
```

### Data Flow

1. **Ingestion**: `ingest_tickets.py` → CSV → Embeddings → Endee
2. **Query**: User ticket → Embedding → Vector search → Top-K results
3. **Assignment**: Majority vote from similar tickets → Team prediction
4. **RAG Resolution**: Similar tickets → Context → LLM → Generated response

---

## 📦 Prerequisites

### Required Components

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Core runtime |
| **Endee** | Latest | Vector database |
| **pip** | Latest | Package manager |

### Optional Components

| Component | Version | Purpose |
|-----------|---------|---------|
| **Ollama** | Latest | LLM for RAG endpoints |
| **CUDA** | 11.0+ | GPU acceleration (optional) |

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB free space
- **OS**: Linux, macOS, or Windows (WSL2)

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/smartsupport-ai.git
cd smartsupport-ai
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 3: Install Endee Vector Database

```bash
# Clone and build Endee
cd ~
git clone https://github.com/billionai/endee.git
cd endee
./build.sh avx2  # Or: ./build.sh neon (for ARM), ./build.sh basic (fallback)

# Create data directory
mkdir -p data
```

### Step 4: Install Ollama (Optional - for RAG)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows (via WSL2)
curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3 model
ollama pull llama3
```

---

## 🏃 Quick Start

### 1. Start Endee

```bash
# Terminal 1
cd ~/endee
export NDD_DATA_DIR=$(pwd)/data
./build/ndd-avx2  # Or: ./build/ndd-neon for ARM
```

Expected output:
```
Endee server started on http://localhost:8080
```

### 2. Prepare Data

Place your `cleaned_tickets.csv` file in the `data/` directory:

```bash
mkdir -p data
# Copy your CSV file
cp /path/to/cleaned_tickets.csv data/
```

**CSV Format Required:**
```csv
ticket_id,description,team,resolution
1001,"Payment failed with error code 500","Billing and Payments","Reset payment gateway cache"
1002,"Cannot log into account","Technical Support","Password reset email sent"
```

### 3. Run System Diagnostics

```bash
# Check all components
python test_setup.py

# Verbose mode
python test_setup.py --verbose
```

Expected output:
```
✅ Python 3.11.0 (compatible)
✅ All 10 required packages installed
✅ CSV file found (45.2 MB)
✅ Endee is running
✅ Embedding model working
...
🎉 All systems ready!
```

### 4. Ingest Data

```bash
# Load tickets into vector database
python ingest_tickets.py
```

Expected output:
```
[1/4]  Checking Endee server…
  ✅  Endee running at localhost:8080
[2/4]  Loading CSV…
  ✅  16,000 tickets ready
[3/4]  Creating index…
  ✅  Index 'tickets' created
[4/4]  Ingesting...
Ingesting: 100%|██████████| 16000/16000
✅  Inserted : 16,000 tickets
🎉  Ingestion complete!
```

### 5. Start API Server

```bash
# Terminal 2
uvicorn backend.main:app --reload

# Or for production
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. Open Web UI

Navigate to: **http://127.0.0.1:8000/ui**

Or use the interactive API docs: **http://127.0.0.1:8000/docs**

---

## 💻 Usage

### Web Interface

1. **Fill in ticket details**:
   - Summary (required)
   - Description (required)
   - Issue type, system, priority (optional)

2. **Choose action**:
   - **⚡ Assign Team** - Get team recommendation
   - **💡 Suggest Resolution** - Get resolution suggestion

3. **View results**:
   - Assigned team with confidence score
   - Suggested resolution
   - Similar past tickets

### API Examples

#### Assign Ticket to Team

```bash
curl -X POST "http://127.0.0.1:8000/assign" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My credit card payment keeps failing with error code 500",
    "top_k": 5
  }'
```

Response:
```json
{
  "predicted_team": "Billing and Payments",
  "confidence": 0.85,
  "similar_tickets": 5,
  "status": "success"
}
```

#### Get Resolution Suggestion

```bash
curl -X POST "http://127.0.0.1:8000/resolve" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Cannot access my account after password reset",
    "top_k": 3
  }'
```

Response:
```json
{
  "suggested_resolution": "Please check your spam folder for the reset email. If not received, contact support@company.com for manual verification.",
  "status": "success"
}
```

#### RAG-Based Assignment (Requires Ollama)

```bash
curl -X POST "http://127.0.0.1:8000/assign-rag" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Server keeps crashing every morning at 9 AM",
    "top_k": 5
  }'
```

Response:
```json
{
  "team": "Technical Support",
  "reason": "Similar scheduled outage issues were previously handled by Technical Support, involving server maintenance and cron jobs.",
  "status": "success"
}
```

---

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Returns status of all system components.

### Assignment Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/assign` | POST | Vector-based team assignment |
| `/assign-rag` | POST | LLM-enhanced team assignment |

### Resolution Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/resolve` | POST | Retrieve resolution from similar ticket |
| `/resolve-rag` | POST | LLM-generated contextual resolution |

### Request Parameters

```json
{
  "text": "Ticket description (10-5000 chars)",
  "top_k": 5  // Number of similar tickets to retrieve (1-50)
}
```

---

## 📁 Project Structure

```
smartsupport-ai/
├── backend/
│   ├── __init__.py
│   ├── embedder.py         # SentenceTransformers wrapper
│   ├── endee_client.py     # Direct HTTP client for Endee
│   └── main.py             # FastAPI application
│
├── static/
│   └── index.html          # Web UI
│
├── data/
│   └── cleaned_tickets.csv # Training data (16K tickets)
│
├── ingest_tickets.py       # Data ingestion script
├── test_setup.py           # System diagnostics
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## ⚙️ Configuration

### Embedding Model

Edit `backend/embedder.py`:
```python
MODEL_NAME = "all-MiniLM-L6-v2"  # Change to another model
EMBEDDING_DIM = 384              # Update dimension accordingly
```

### Vector Database

Edit `backend/endee_client.py`:
```python
INDEX_NAME = "tickets"           # Index name
DIMENSION = 384                  # Must match embedding model
BASE_URL = "http://localhost:8080/api/v1"  # Endee URL
```

### API Server

Edit `backend/main.py`:
```python
# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 60  # seconds
```

---

## 🔧 Troubleshooting

### Common Issues

#### ❌ "Cannot connect to Endee"

**Solution:**
```bash
# Check if Endee is running
curl http://localhost:8080/health

# Start Endee if not running
cd ~/endee
export NDD_DATA_DIR=$(pwd)/data
./build/ndd-avx2
```

#### ❌ "Index 'tickets' does not exist"

**Solution:**
```bash
# Run data ingestion
python ingest_tickets.py
```

#### ❌ "sentence-transformers not installed"

**Solution:**
```bash
pip install sentence-transformers torch
```

#### ❌ "Ollama not reachable"

**Solution:**
```bash
# Check Ollama status
ollama list

# Start Ollama (it usually runs as a service)
ollama serve

# Pull llama3 if missing
ollama pull llama3
```

#### ⚠️ "Low confidence" assignments

**Causes:**
- Insufficient training data
- Ticket description too vague
- No similar historical tickets

**Solutions:**
- Add more training examples to CSV
- Provide more detailed ticket descriptions
- Adjust `top_k` parameter

### Performance Issues

#### Slow embedding generation

```bash
# Use GPU acceleration (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or use smaller model
# Edit backend/embedder.py: MODEL_NAME = "all-MiniLM-L12-v2"
```

#### Slow vector search

```bash
# Increase Endee HNSW parameters
# Edit backend/endee_client.py:
DEFAULT_M = 32       # More connections (default: 16)
DEFAULT_EF_CON = 256 # Higher construction EF (default: 128)
DEFAULT_EF = 256     # Higher search EF (default: 128)
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
uvicorn backend.main:app --reload --log-level debug

# Test with diagnostics
python test_setup.py --verbose
```

---

## 🛠️ Development

### Running Tests

```bash
# System diagnostics
python test_setup.py

# Test individual components
python -c "from backend.embedder import embed_text; print(len(embed_text('test')))"
python -c "from backend.endee_client import check_connection; print(check_connection())"
```

### Adding New Features

#### Custom Team Logic

Edit `backend/main.py` → `assign_ticket()`:
```python
# Add custom rules
if "urgent" in text.lower():
    return AssignResponse(
        predicted_team="Priority Team",
        confidence=1.0,
        similar_tickets=0,
        status="custom_rule"
    )
```

#### New Endpoints

```python
@app.post("/custom-endpoint")
def custom_handler(req: TicketRequest):
    # Your logic here
    return {"result": "success"}
```

### Environment Variables

```bash
# Disable model downloads (use cached)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Custom ports
export ENDEE_PORT=8080
export API_PORT=8000
export OLLAMA_PORT=11434
```

---

## 📊 Data Format

### Input CSV Requirements

```csv
ticket_id,description,team,resolution
<str>,<str:10-5000>,<str>,<str>
```

**Columns:**
- `ticket_id` (string): Unique identifier
- `description` (string): Ticket text (10-5000 chars)
- `team` (string): Team that handled the ticket
- `resolution` (string): How the issue was resolved

**Example:**
```csv
ticket_id,description,team,resolution
T-1001,"Payment failed with error 500","Billing and Payments","Reset payment gateway cache and retry"
T-1002,"Cannot login to account","Technical Support","Sent password reset email to user@example.com"
T-1003,"Request to add new user","HR and Accounts","Created new account and sent credentials"
```

### Best Practices

1. **Clean descriptions**: Remove PII, standardize formatting
2. **Consistent teams**: Use same team names across tickets
3. **Detailed resolutions**: More context = better suggestions
4. **Sufficient volume**: Minimum 100 tickets per team recommended

---

## 🚀 Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY static/ ./static/
COPY ingest_tickets.py .

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using systemd

```ini
[Unit]
Description=SmartSupport AI API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/smartsupport
ExecStart=/opt/smartsupport/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name support.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📈 Performance Benchmarks

### Embedding Generation
- **Single text**: ~50ms (CPU), ~5ms (GPU)
- **Batch (100)**: ~500ms (CPU), ~50ms (GPU)

### Vector Search
- **Top-1**: <1ms (for 100K vectors)
- **Top-10**: <2ms (for 100K vectors)

### End-to-End
- **/assign**: 50-100ms
- **/resolve**: 50-100ms
- **/assign-rag**: 5-10s (with Ollama)
- **/resolve-rag**: 3-5s (with Ollama)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🙏 Acknowledgments

- **Endee** - High-performance vector database
- **SentenceTransformers** - Neural text embeddings
- **FastAPI** - Modern Python web framework
- **Ollama** - Local LLM runtime
- **all-MiniLM-L6-v2** - Efficient embedding model by Microsoft

---

## 📧 Support

- **Issues**: GitHub Issues
- **Email**: support@example.com
- **Docs**: http://127.0.0.1:8000/docs (when server is running)

---

**Built with ❤️ for better customer support**