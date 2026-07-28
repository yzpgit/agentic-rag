# 腾讯 CloudBase 云托管部署指南

> 部署后获得公网 URL，手机浏览器直接访问，支持流式问答。

## 前置准备

1. **腾讯云账号**：访问 [cloud.tencent.com](https://cloud.tencent.com) 注册
2. **开通 CloudBase**：进入 [CloudBase 控制台](https://tcb.cloud.tencent.com)，首次进入会引导开通「云托管」服务（地域选**上海**）
3. **创建环境**：在控制台创建一个环境（免费额度足够演示），记下 **环境 ID**（形如 `cloud1-xxxxx`）
4. **LongCat API Key**：你已有的 `ak_26n8JJ1491HJ0Hd7o73s33V74DW1p`

---

## 方式一：控制台上传代码包（最简单，推荐首次部署）

### 步骤

1. **下载代码包**：`agentic-rag-cloudbase.zip`（已为你打包好）

2. 进入 [CloudBase 控制台 → 云托管 → 服务列表](https://tcb.cloud.tencent.com/dev#/platform-run)，点击「新建服务」

3. 填写服务信息：
   - **服务名称**：`agentic-rag`
   - **部署方式**：上传代码包
   - **代码包**：上传 `agentic-rag-cloudbase.zip`
   - **端口**：`80`（重要！）
   - **规格**：0.5 核 1G（默认即可）
   - **最小副本数**：`0`（低成本模式，无请求时不计费）
   - **最大副本数**：`5`

4. **配置环境变量**（关键步骤）：在「高级设置」或服务创建后的「环境变量」中添加：
   ```
   OPENAI_BASE_URL = https://api.longcat.chat/openai/v1
   OPENAI_API_KEY  = ak_26n8JJ1491HJ0Hd7o73s33V74DW1p
   ```

5. 点击「创建」并等待构建（约 3-5 分钟，主要时间在 pip install）

6. 构建完成后，服务列表会显示**访问地址**（形如 `https://agentic-rag-xxx-xxx.sh.run`），手机浏览器打开即可使用

---

## 方式二：GitHub 自动部署（推荐长期维护）

适合后续迭代：代码 push 到 GitHub 自动触发部署。

### 步骤

1. **把代码推到 GitHub 仓库**（公开或私有均可）

2. 进入 CloudBase 控制台 → 云托管 → 新建服务 → 部署方式选「Git 代码库」

3. 授权并关联你的 GitHub 仓库，分支填 `main`

4. 端口填 `80`，环境变量同方式一

5. 后续 `git push` 即自动部署

---

## 方式三：CLI 部署（适合电脑端操作）

```bash
# 1. 安装 CLI
npm i -g @cloudbase/cli

# 2. 登录（会打开浏览器授权）
tcb login

# 3. 进入项目目录
cd agentic-rag

# 4. 编辑 cloudbaserc.json，把 {{ENV_ID}} 替换为你的环境 ID，
#    {{YOUR_LONGCAT_API_KEY}} 替换为你的 LongCat API Key

# 5. 部署
tcb cloudrun deploy --port 80
```

---

## 部署后验证

1. 访问 `https://你的服务地址/health`，应返回 `{"status":"ok"}`

2. 访问 `https://你的服务地址/`，应看到 RAG 对话界面

3. 直接提问测试（已预置 RAG 技术介绍文档）：
   - "RAG 解决了什么问题？"
   - "什么是混合检索？"
   - "BM25 和向量检索有什么区别？"

---

## 重要说明

### 1. 冷启动延迟
低成本模式（minNum=0）无请求时缩容到 0，首次访问会有 **5-10 秒冷启动**。如需常驻，把最小副本数改为 1（会持续计费）。

### 2. 数据持久性
云托管实例本地存储是临时的，实例重启后**用户上传的文档会丢失**，但**镜像内预置的示例文档（rag-intro.md）会在每次启动时自动入库**。如需持久化用户上传文档，需接入 COS 对象存储（阶段2+ 优化项）。

### 3. 费用
- 免费额度足够演示（每月 100 万次请求 + 计算资源）
- 无流量时缩容到 0，不计费
- 实际使用按秒计费，演示场景几乎零成本

### 4. 域名
CloudBase 提供的是 `*.sh.run` 公网域名，自动 HTTPS，手机直接访问，无需额外配置。

---

## 常见问题

**Q：构建失败怎么办？**
A：查看构建日志。最常见是依赖安装超时，可在 `requirements.txt` 里临时注释掉 `sentence-transformers` 和 `faiss-cpu`（当前默认用 BM25，不需要它们）。

**Q：服务起来了但提问没反应？**
A：检查环境变量是否配置正确，特别是 `OPENAI_API_KEY`。可在 CloudBase 日志面板查看实时日志。

**Q：想用向量检索/混合检索？**
A：编辑 `config/config.yaml`，把 `retriever.mode` 改为 `hybrid`。但会增大镜像体积（需下载 embedding 模型）且首次启动变慢。
