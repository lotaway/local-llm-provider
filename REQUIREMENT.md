# 目标

改成多Agent形式，问答Agent->规划Agent->路由Agent->任务Agent（LLM/RAG/MCP）->验证Agent->是否高风险需要human接入->回到规划Agent确认是否完成，否则继续下一步或者重新规划，直到完成目标。

# 需要完成的强化特性

1. Context Window 信噪比控制

目标不是把最多内容塞进 prompt，而是让每个 token 的价值最高。

2. Hybrid Search：向量 vs 关键词加权

Weighted Linear Fusion或者Learned Hybrid Retrieval

3. RAG添加重排序

使用BM25->Embedding->Cross-Encoder或者更好的方式，文档数量目前是几十个，但需要适应那种上千法律文档的或者可配置切换的。

4. Graph RAG：跨文档推理与结构化知识（可选）

目前暂时无太多跨文档知识，但是需要找到可行方案，包括docker配置。

5. 添加更多MCP工具方便调用

例如联网查询、文件搜索、图片文字解析、图片识别、音频识别等。

6. 添加权限控制

不同工具需要提供安全级别，让Agent Runtime执行请求工具操作时可以判断是否需要human介入以及提供正确反馈或者操作。