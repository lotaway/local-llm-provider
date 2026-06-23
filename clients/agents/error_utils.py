"""Utilities for exception handling and evidence extraction"""

import json
import logging
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EvidenceExtractor:
    """Non-LLM logic for gathering and structured evidence extraction"""

    @staticmethod
    def extract_from_exception(exception: Exception, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evidence from an exception and current context"""
        tb = traceback.format_exc()
        
        # 1. Deterministic clipping (Request/Span ID, Time Window)
        # Assuming context might have request_id or span_id
        request_id = context.get("request_id") or context.get("trace_id")
        
        # 2. Statistical & Rule-based compression
        stack_lines = tb.strip().split("\n")
        root_stack = stack_lines[-1] if stack_lines else str(exception)
        
        # 3. Structural Evidence
        evidence = {
            "error_signatures": [
                {
                    "id": type(exception).__name__,
                    "message": str(exception),
                    "count": 1,
                    "first_seen": datetime.now().isoformat(),
                    "root_stack": root_stack
                }
            ],
            "agent_context": {
                "name": agent_name,
                "iteration": context.get("iteration_count", 0),
                "request_id": request_id
            },
            "time_span": {
                "start": (datetime.now() - timedelta(minutes=1)).isoformat(),
                "end": datetime.now().isoformat()
            }
        }
        
        return evidence

class SystemSnapshot:
    """Collector for runtime metrics and environment snapshots"""

    @staticmethod
    def capture(context: Dict[str, Any]) -> Dict[str, Any]:
        """Capture current system state snapshot"""
        # In a real system, this would fetch hardware metrics, service status, etc.
        # Here we capture important context variables and pseudo-metrics
        
        snapshot = {
            "runtime_metrics": {
                "iteration": context.get("iteration_count", 0),
                "history_length": len(context.get("history", [])),
                "context_size": len(json.dumps(context, default=str))
            },
            "environment_diff": {
                "recent_changes": context.get("recent_changes", []),
                "active_agent": context.get("current_agent")
            },
            "signals": [
                {
                    "name": "heartbeat",
                    "status": "ok",
                    "sample_time": datetime.now().isoformat(),
                    "confidence": 1.0
                }
            ]
        }
        
        return snapshot
