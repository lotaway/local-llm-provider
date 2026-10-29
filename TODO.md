# TODO — 未完成需求汇总

## 1. Agent 加载 Skill + MCP（来源：CLAUDE_SPEC.md）

- [x] 实现 MCP 协议 `initialize` / `tools/list` 动态发现，补齐 `DEGRADED` 退化状态与健康检查（当前仅按配置静态注册 tools，`DEGRADED` 枚举未被使用）
- [x] 在 `utils/mcp_loader.py` 读取并应用 `LLP_MCP_DEFAULT_SAFETY`，将安全级别传递给 `register_tool` 与 Capability
- [x] 在 `.env.example` 补充 `LLP_MCP_CONNECTIONS_DIR`、`LLP_MCP_CONNECTIONS_FILE`、`LLP_MCP_DEFAULT_SAFETY`
- [x] 统一安全级别取值：Capability 默认 `SAFE` 与规格的 `LOW|MEDIUM|HIGH|CRITICAL` 对齐，并核对 PermissionManager 双阶段审批提示
- [x] 新增 `tests/test_mcp_env_loading.py`、`tests/test_skills_env_loading.py`（规格测试计划列出但文件不存在）

## 2. MOLT 类脑记忆系统（来源：MOLT_BRAIN_PLAN.md + MOLT_BRAIN_SPEC.md）

- [x] 修复 `services/evolution_dispatcher.py`：`EvolutionDispatcher` 类缺失，`__init__` 与全部方法被错误嵌套进 `EvolutionConfig`；且 `config: EvolutionConfig | None` 在类体内前向引用会在导入时抛 `NameError`，导致 `clients/agents/runtime_factory.py` 无法导入
- [x] 实现 `VersionManager` 真实逻辑：冲突检测（语义相似度）、同源合并、异源保留版本、版本链查询、回滚；接入 `Neo4jRepository.save_ltm / get_ltm_versions_chain / create_ltm_version_relation`
- [x] 实现 `AbstractionEngine.abstract_to_ltm`：LLM 抽象总结、生成标准化结论（conclusion/conditions/confidence/sources）、写入 Neo4j LTM、结论向量写入 Milvus、关键词写入 ES
- [x] 为 `MoltController` 增加动态策略调整 `evaluate_and_adjust`（recall threshold / decay rate / abstraction 触发阈值），当前仅有 `collect_metrics`
- [x] 实现 `IMoltScheduler`（当前仅有接口），将 decay / abstraction / evaluation 周期任务注册到现有调度器
- [x] 修复/更新 `tests/test_molt_integration.py`：引用了已不存在的 `FeedbackConfig / DecayConfig / AbstractionConfig / MoltConfig / VersionConfig`，且构造签名不匹配，测试无法收集
- [x] 将 `graph_retriever` 接入 `FeedbackService.on_recall` 回调（目前仅 `hybrid_retriever` 有）
- [x] 实现 M1 感觉缓存（MongoDB TTL collection）与 M2 工作记忆（进程内 7±2、rehearsal count）
- [x] 实现 LTM 结论向量（Milvus）与 ES 关键词 Tag 写入，补全可解释性字段（来源、过期/强化时间）
- [x] 核对 `LEARNING` 开关覆盖所有进化入口（调度器、回滚等路径）

## 3. RAG 架构（来源：RAG_SPEC.md）

- [ ] 查询协同：Neo4j 命中 Concept 后反查关联 `chunk_id`，并在 Milvus 检索时按 `chunk_id` 过滤（当前仅把图上下文拼入 prompt）
- [ ] 对齐 ES 索引字段为 `chunk_id/doc_id/text/filename` 并支持高亮（当前仅写入 `content + metadata`）
- [ ] 补齐 `infrastructure/rag/milvus_store.py` 与 `infrastructure/rag/local_rag_service.py::MilvusVectorStore` 的占位实现（`add_documents` 仅返回 id、`similarity_search` 返回空）
- [ ] 校验增量/可重跑流程：`documents.status`（chunked/embedded/graphed）断点续跑与幂等
- [ ] 增加 checksum 变更触发重建与失败状态重试

## 4. TTS 情感语音系统（来源：TTS_SPEC.md）

- [ ] 初始化 `third_party/StyleTTS2` 子模块并安装依赖（当前目录为空）
- [ ] 实现 `StyleTTS2Adapter.generate_speech` 真实推理（当前为占位 `pass`）
- [ ] 实现 `clone_voice` 完整流程：参考音频→Speaker Embedding→Redis 缓存→MongoDB `tts_speakers` 持久化（当前为占位）
- [ ] 新增 HTTP 接口 `POST /tts`、`POST /tts/clone`
- [ ] 实现 Emotion Parser（连续 VAD：valence/arousal/dominance/speed/pitch/energy）
- [ ] 实现流式输出（首包延迟 <500ms）
- [ ] 接入用户偏好记忆 `tts_user_profiles`
- [ ] 将质量网关（WER/DNSMOS/Similarity 自动拦截）接入 TTS 生成链路
- [ ] 实现后处理呼吸感与随机 jitter

## 5. TTS 质量评估与线上监控（来源：TTS_QUALITY_ESTIMATE.md）

- [ ] 生产监控抽样（5%~10% 请求）接入评估流水线
- [ ] 每日统计、滚动均值、P90/P95、异常比例
- [ ] 报警阈值与通知（WER / DNSMOS / Speaker Similarity / 失败率）
- [ ] 生成失败率、音频时长异常、重复片段检测
- [ ] 用户反馈分布（差评 / 投诉 / 重新生成次数）
- [ ] 固定测试句集 + 每日回归测试 + 版本对比曲线
- [ ] 补齐 `tts_estimate` 的 DNSMOS 真实推理细节（当前 `_run_inference` 为简化实现，需按官方前处理/分帧）
- [ ] 为 `tts_estimate` 增加批量评估与 CLI 参数入口（当前 `main.py` 使用 mock 路径）
