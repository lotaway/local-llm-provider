"""
Molt Controller：元认知调度中心

职责：
1. 监控记忆系统指标
2. 动态调整策略（衰减率、召回阈值）
3. 调度抽象任务
4. 处理冲突
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics

from repositories.mongodb_repository import MongoDBRepository
from repositories.neo4j_repository import Neo4jRepository
from services.feedback_service import FeedbackService
from services.decay_scheduler import DecayScheduler
from services.abstraction_engine import AbstractionEngine

logger = logging.getLogger(__name__)


@dataclass
class MoltMetrics:
    """记忆系统指标"""

    recall_hit_rate: float = 0.0
    avg_importance: float = 0.0
    decay_rate_effective: float = 0.0
    abstraction_rate: float = 0.0
    total_chunks: int = 0
    episodic_chunks: int = 0
    ltm_nodes: int = 0
    discarded_chunks: int = 0
    stale_chunks: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MoltController:
    """Molt Controller"""

    RECALL_HIT_RATE_THRESHOLD = 0.3
    MIN_ABSTRACTION_RATE = 0.01
    STALE_THRESHOLD_DAYS = 30

    def __init__(
        self,
        mongo_repo: MongoDBRepository = None,
        neo4j_repo: Neo4jRepository = None,
        feedback_service: FeedbackService = None,
        decay_scheduler: DecayScheduler = None,
        abstraction_engine: AbstractionEngine = None,
    ):
        self.mongo_repo = mongo_repo or MongoDBRepository()
        self.neo4j_repo = neo4j_repo or Neo4jRepository()
        self.feedback_service = feedback_service or FeedbackService(self.mongo_repo)
        self.decay_scheduler = decay_scheduler or DecayScheduler(
            self.mongo_repo, self.feedback_service
        )
        self.abstraction_engine = abstraction_engine or AbstractionEngine(
            self.mongo_repo, self.neo4j_repo, self.feedback_service
        )

        self.strategies = {
            "decay_rate": 0.01,
            "recall_threshold": 0.5,
            "abstraction_min_episodes": 5,
            "abstraction_min_importance": 0.6,
        }

    def collect_metrics(self) -> MoltMetrics:
        """收集记忆系统指标"""
        stats = self.decay_scheduler.get_decay_stats()

        total_chunks = stats.get("total_chunks", 0)
        episodic_chunks = stats.get("episodic_chunks", 0)
        discarded_chunks = stats.get("discarded_chunks", 0)
        avg_importance = stats.get("average_importance", 0.0)

        ltm_nodes = len(self.neo4j_repo.get_ltm_by_topic("")) or 0
        if isinstance(ltm_nodes, list):
            ltm_nodes = len(ltm_nodes) if ltm_nodes else 0

        stale_chunks = len(
            self.feedback_service.get_stale_memories(days=self.STALE_THRESHOLD_DAYS)
        )

        return MoltMetrics(
            recall_hit_rate=self._calculate_recall_hit_rate(),
            avg_importance=avg_importance,
            decay_rate_effective=stats.get("decay_rate", 0.01),
            abstraction_rate=self._calculate_abstraction_rate(),
            total_chunks=total_chunks,
            episodic_chunks=episodic_chunks,
            ltm_nodes=ltm_nodes,
            discarded_chunks=discarded_chunks,
            stale_chunks=stale_chunks,
        )

    def _calculate_recall_hit_rate(self) -> float:
        """计算召回命中率"""
        top_memories = self.feedback_service.get_top_memories(limit=100)
        if not top_memories:
            return 0.5

        reviewed = sum(1 for m in top_memories if m.get("last_reviewed"))
        return reviewed / len(top_memories)

    def _calculate_abstraction_rate(self) -> float:
        """计算抽象频率"""
        ltms = self.neo4j_repo.get_ltm_by_topic("")
        if not ltms:
            return 0.0

        recent_ltms = [
            l
            for l in ltms
            if isinstance(l.get("created_at"), datetime)
            and (datetime.utcnow() - l["created_at"]).days < 7
        ]
        return len(recent_ltms) / max(1, len(ltms))

    def evaluate_and_adjust(self) -> Dict[str, Any]:
        """
        评估指标并调整策略

        Returns:
            调整报告
        """
        metrics = self.collect_metrics()
        adjustments = []

        if metrics.recall_hit_rate < self.RECALL_HIT_RATE_THRESHOLD:
            old_threshold = self.strategies.get("recall_threshold", 0.5)
            new_threshold = max(0.1, old_threshold * 0.9)
            self.strategies["recall_threshold"] = new_threshold
            adjustments.append(
                {
                    "type": "recall_threshold",
                    "old": old_threshold,
                    "new": new_threshold,
                    "reason": f"Low recall hit rate: {metrics.recall_hit_rate:.2%}",
                }
            )

        if metrics.stale_chunks > metrics.episodic_chunks * 0.5:
            old_decay = self.strategies.get("decay_rate", 0.01)
            new_decay = min(0.1, old_decay * 1.2)
            self.strategies["decay_rate"] = new_decay
            adjustments.append(
                {
                    "type": "decay_rate",
                    "old": old_decay,
                    "new": new_decay,
                    "reason": f"Too many stale chunks: {metrics.stale_chunks}",
                }
            )

        if metrics.abstraction_rate < self.MIN_ABSTRACTION_RATE:
            old_min = self.strategies.get("abstraction_min_episodes", 5)
            new_min = max(3, old_min - 1)
            self.strategies["abstraction_min_episodes"] = new_min
            adjustments.append(
                {
                    "type": "abstraction_min_episodes",
                    "old": old_min,
                    "new": new_min,
                    "reason": f"Low abstraction rate: {metrics.abstraction_rate:.2%}",
                }
            )

        return {
            "metrics": {
                "recall_hit_rate": f"{metrics.recall_hit_rate:.2%}",
                "avg_importance": f"{metrics.avg_importance:.4f}",
                "abstraction_rate": f"{metrics.abstraction_rate:.2%}",
                "total_chunks": metrics.total_chunks,
                "ltm_nodes": metrics.ltm_nodes,
                "stale_chunks": metrics.stale_chunks,
            },
            "adjustments": adjustments,
            "current_strategies": self.strategies,
        }

    def run_decay_job(self) -> Dict[str, Any]:
        """执行衰减任务"""
        stats = self.decay_scheduler.apply_decay_all()
        self.decay_scheduler.cleanup_low_importance()
        return stats

    def run_abstraction_job(self) -> List[Dict[str, Any]]:
        """执行抽象任务"""
        candidates = self.abstraction_engine.get_abstraction_candidates()
        results = []

        for candidate in candidates[:10]:
            topic = candidate["topic"]
            result = self.abstraction_engine.abstract_to_ltm(topic)
            if result:
                results.append(
                    {
                        "topic": topic,
                        "ltm_topic_version": result.ltm_topic_version,
                        "confidence": result.confidence,
                    }
                )

        return results

    def run_full_cycle(self) -> Dict[str, Any]:
        """
        运行完整的Molt周期

        1. 收集指标
        2. 调整策略
        3. 执行衰减
        4. 执行抽象
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": None,
            "adjustments": None,
            "decay_stats": None,
            "abstraction_results": [],
        }

        try:
            report["metrics"] = self.evaluate_and_adjust()
            report["adjustments"] = report["metrics"]["adjustments"]

            report["decay_stats"] = self.run_decay_job()

            abstraction_results = self.run_abstraction_job()
            report["abstraction_results"] = abstraction_results
            report["abstraction_count"] = len(abstraction_results)

        except Exception as e:
            logger.error(f"Molt cycle failed: {e}")
            report["error"] = str(e)

        return report

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        metrics = self.collect_metrics()
        return {
            "healthy": True,
            "metrics": {
                "recall_hit_rate": f"{metrics.recall_hit_rate:.2%}",
                "avg_importance": f"{metrics.avg_importance:.4f}",
                "total_chunks": metrics.total_chunks,
                "episodic_chunks": metrics.episodic_chunks,
                "ltm_nodes": metrics.ltm_nodes,
                "discarded_chunks": metrics.discarded_chunks,
                "stale_chunks": metrics.stale_chunks,
            },
            "strategies": self.strategies,
            "last_updated": datetime.utcnow().isoformat(),
        }

    def close(self):
        """关闭所有连接"""
        self.mongo_repo.close()
        self.neo4j_repo.close()
        self.feedback_service.close()


class MoltConfig:
    """Molt Controller 配置"""

    RECALL_HIT_RATE_THRESHOLD = 0.3
    MIN_ABSTRACTION_RATE = 0.01
    STALE_THRESHOLD_DAYS = 30
    DECAY_INTERVAL_HOURS = 6
    ABSTRACTION_INTERVAL_HOURS = 12
    EVALUATION_INTERVAL_HOURS = 24

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "recall_hit_rate_threshold": cls.RECALL_HIT_RATE_THRESHOLD,
            "min_abstraction_rate": cls.MIN_ABSTRACTION_RATE,
            "stale_threshold_days": cls.STALE_THRESHOLD_DAYS,
            "decay_interval_hours": cls.DECAY_INTERVAL_HOURS,
            "abstraction_interval_hours": cls.ABSTRACTION_INTERVAL_HOURS,
            "evaluation_interval_hours": cls.EVALUATION_INTERVAL_HOURS,
        }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    controller = MoltController()
    status = controller.get_status()
    print("Molt Controller Status:")
    for k, v in status["metrics"].items():
        print(f"  {k}: {v}")
    print(f"Strategies: {status['strategies']}")
    controller.close()
