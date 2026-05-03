# Pluggable Knowledge Bridge RAG MCP - Quick Start (EN + 中文)

---

## English Version

### What This Project Does

This project is a modular RAG system with:

- Document ingestion (PDF/TXT/MD)
- Vector retrieval (Chroma)
- Hybrid retrieval (Dense + Sparse)
- Streamlit dashboard
- MCP server tools for Copilot/Claude clients

### Prerequisites

- macOS / Linux
- Python 3.10+ (3.11 recommended)
- Git
- Ollama (if using local embeddings)

### 1) Clone and Enter

```bash
git clone <your-repo-url>
cd pluggable-knowledge-bridge-rag-mcp
```

### 2) Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install Dependencies

```bash
pip install -e .
```

### 4) Configure Secrets

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_API_KEY=your_real_key_here
```

Security rule:
- Keep real secrets in `.env` only
- Never commit `.env`
- Keep placeholders only in `.env.example`

### 5) Configure `config/settings.yaml`

Example:

```yaml
llm:
  provider: "openai"
  model: "astron-code-latest"
  api_key: "${LLM_API_KEY}"
  base_url: "https://maas-coding-api.cn-huabei-1.xf-yun.com/v2"
  temperature: 0.0
  max_tokens: 4096

embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"
  dimensions: 768

vector_store:
  provider: "chroma"
  persist_directory: "./data/db/chroma"
  collection_name: "knowledge_hub_768"
```

Note: Chroma collection dimension is fixed after first write.  
If you switch from 128-dim to 768-dim, use a new `collection_name`.

### 6) Start Required Services

Start Ollama:

```bash
ollama serve
```

Verify model:

```bash
ollama list
```

If needed:

```bash
ollama pull nomic-embed-text
```

### 7) Start Dashboard

```bash
cd /path/to/pluggable-knowledge-bridge-rag-mcp
source .venv/bin/activate
python scripts/start_dashboard.py
```

Open:

`http://localhost:8501`

### 8) Ingest Your First Document

Dashboard path:
- Ingestion Manager -> upload file -> collection `default` -> Start Ingestion

CLI path:

```bash
python scripts/ingest.py --path "/absolute/path/to/your.pdf" --collection default --force --verbose
```

Success criteria:
- `Success`
- `Chunks: N` where `N > 0`
- Data Browser chunk count increases

### 9) Run Queries

```bash
python scripts/query.py --query "What is the core goal of this memo?" --top-k 5 --verbose
python scripts/query.py --query "What operational actions were proposed?" --top-k 5 --verbose
python scripts/query.py --query "What dates/budgets/metrics are mentioned?" --top-k 5 --verbose
```

Success criteria:
- non-empty results
- source points to the ingested PDF

### 10) Use as MCP Server (VS Code Copilot)

Create/confirm `.vscode/mcp.json`:

```json
{
  "servers": {
    "knowledge-hub": {
      "command": "/absolute/path/to/project/.venv/bin/python",
      "args": ["main.py"],
      "cwd": "/absolute/path/to/project"
    }
  }
}
```

Then:
1. `MCP: List Servers` -> `knowledge-hub` should be `Running`
2. Open **GitHub Copilot Chat** (not Claude Code panel)
3. Ask:
   - `Please call query_knowledge_hub with collection=default and query="What is the core goal of this memo?"`

### Verification Checklist

- [ ] Ollama running
- [ ] Dashboard reachable on port 8501
- [ ] Ingestion success with chunks > 0
- [ ] CLI query returns relevant results
- [ ] MCP server running in VS Code
- [ ] Copilot can call MCP tools

### Common Issues

1. `Collection expecting embedding ... 128 vs 768`  
Use a new `vector_store.collection_name`.

2. `Failed to connect to Ollama`  
Run `ollama serve`.

3. Query returns 0 results  
Check ingestion success, collection alignment, and query traces.

4. MCP starts then stops  
Ensure MCP command uses project `.venv/bin/python` and `args: ["main.py"]`.

---

## 中文版

### 项目作用

这是一个可插拔 RAG 系统，支持：

- 文档摄入（PDF/TXT/MD）
- 向量检索（Chroma）
- 混合检索（Dense + Sparse）
- Streamlit Dashboard
- MCP Server（供 Copilot/Claude 调用）

### 前置要求

- macOS / Linux
- Python 3.10+（推荐 3.11）
- Git
- Ollama（如果使用本地向量模型）

### 1）克隆并进入项目

```bash
git clone <your-repo-url>
cd pluggable-knowledge-bridge-rag-mcp
```

### 2）创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3）安装依赖

```bash
pip install -e .
```

### 4）配置密钥

```bash
cp .env.example .env
```

编辑 `.env`：

```env
LLM_API_KEY=你的真实密钥
```

安全规则：
- 真密钥只放 `.env`
- `.env` 不要提交 GitHub
- `.env.example` 只放占位符

### 5）配置 `config/settings.yaml`

示例：

```yaml
llm:
  provider: "openai"
  model: "astron-code-latest"
  api_key: "${LLM_API_KEY}"
  base_url: "https://maas-coding-api.cn-huabei-1.xf-yun.com/v2"
  temperature: 0.0
  max_tokens: 4096

embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"
  dimensions: 768

vector_store:
  provider: "chroma"
  persist_directory: "./data/db/chroma"
  collection_name: "knowledge_hub_768"
```

注意：Chroma collection 首次写入后会锁定维度。  
若从 128 维切到 768 维，需要改新的 `collection_name`。

### 6）启动必要服务

启动 Ollama：

```bash
ollama serve
```

检查模型：

```bash
ollama list
```

如需下载：

```bash
ollama pull nomic-embed-text
```

### 7）启动 Dashboard

```bash
cd /path/to/pluggable-knowledge-bridge-rag-mcp
source .venv/bin/activate
python scripts/start_dashboard.py
```

访问：

`http://localhost:8501`

### 8）摄入第一份文档

Dashboard 方式：
- Ingestion Manager -> 上传文件 -> collection 填 `default` -> Start Ingestion

命令行方式：

```bash
python scripts/ingest.py --path "/absolute/path/to/your.pdf" --collection default --force --verbose
```

成功标准：
- 输出 `Success`
- `Chunks: N` 且 `N > 0`
- Data Browser 顶部 chunk 数增加

### 9）执行查询

```bash
python scripts/query.py --query "这份备忘录的核心目标是什么？" --top-k 5 --verbose
python scripts/query.py --query "提到了哪些运营动作？" --top-k 5 --verbose
python scripts/query.py --query "有哪些时间、预算或指标数字？" --top-k 5 --verbose
```

成功标准：
- 有结果返回
- source 指向你刚摄入的 PDF

### 10）作为 MCP Server 给 VS Code Copilot 使用

创建/确认 `.vscode/mcp.json`：

```json
{
  "servers": {
    "knowledge-hub": {
      "command": "/absolute/path/to/project/.venv/bin/python",
      "args": ["main.py"],
      "cwd": "/absolute/path/to/project"
    }
  }
}
```

然后：
1. `MCP: List Servers`，确认 `knowledge-hub` 是 `Running`
2. 打开 **GitHub Copilot Chat**（不是 Claude Code 面板）
3. 提问时明确调用：
   - `请调用 query_knowledge_hub，collection=default，query=这份备忘录的核心目标是什么？`

### 验证清单

- [ ] Ollama 正常运行
- [ ] Dashboard 可访问（8501）
- [ ] Ingestion 成功且 chunks > 0
- [ ] CLI 查询有命中
- [ ] MCP server 在 VS Code 里是 Running
- [ ] Copilot 能调用 MCP tools

### 常见问题

1. `Collection expecting embedding ... 128 vs 768`  
改 `vector_store.collection_name` 为新值。

2. `Failed to connect to Ollama`  
先执行 `ollama serve`。

3. Query 返回 0  
先查 ingestion、collection 是否一致，再看 query traces。

4. MCP 启动后立刻停止  
确认 MCP command 指向项目 `.venv/bin/python`，`args` 为 `["main.py"]`。
