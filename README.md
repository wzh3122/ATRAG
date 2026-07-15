# ATRAG

ATRAG 是一个支持混合检索、Agent 工具调用和可配置 LLM 路由的知识库应用。

## 核心链路

1. 客户端通过现有聊天接口提交消息。
2. 混合前置路由优先调用用户选择的 LLM，判断是否需要工具、知识库、联网搜索或直接回答。
3. 路由 LLM 不可用、超时或返回无效结果时，确定性规则路由接管。
4. 原有 Agent 结合路由结果进行第二次判断并执行工具。
5. 结果继续通过项目原有的流式响应链路返回。

路由模型通过 `routing_completion` 指定。服务端按当前用户的模型供应商配置解析 API Key；未指定时复用主回答模型。

## 目录

- `atrag/`：后端、Agent、检索、MCP、任务和数据库迁移。
- `web/`：前端。
- `config/`：Celery 等进程配置。
- `deploy/atrag/`：Helm 部署配置。
- `docker-compose.yml`：本地完整运行环境。

## 本地开发

```bash
make install
make compose-infra
make migrate
make run-backend
```

数据库名、对象存储桶和数据卷统一使用 `atrag`，代码使用 `atrag` 包名和 `ATRAG_*` 环境变量。
