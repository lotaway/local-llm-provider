import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from constants import MONGO_URI, MONGO_DB_NAME, ROOT_TASK, ROOT_TASK_INTERVAL, ENABLE_SCHEDULER, MONGO_USER, MONGO_PASSWORD

logger = logging.getLogger("service.scheduler")

class TaskScheduler:
    def __init__(self, agent_runtime=None):
        self.agent_runtime = agent_runtime
        self._running = False
        self._tasks_loop = None
        self._db = None
        self._collection = None

    async def initialize(self):
        if not ENABLE_SCHEDULER:
            logger.info("Scheduler is disabled.")
            return

        try:
            from pymongo import MongoClient
            
            mongo_uri = MONGO_URI
            if MONGO_USER and MONGO_PASSWORD:
                if "@" not in mongo_uri:
                    uri_parts = mongo_uri.replace("mongodb://", "").split("/")
                    host_port = uri_parts[0]
                    rest = "/" + "/".join(uri_parts[1:]) if len(uri_parts) > 1 else ""
                    mongo_uri = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{host_port}{rest}"
                    
            client = MongoClient(mongo_uri)
            self.client = client
            self._db = client[MONGO_DB_NAME]
            self._collection = self._db["scheduled_tasks"]
            logger.info("Scheduler initialized with MongoDB storage.")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB for scheduler: {e}")

        # Check for root task
        if ROOT_TASK:
            await self.register_root_task()

    async def register_root_task(self):
        task_id = "root_background_task"
        existing = self._collection.find_one({"task_id": task_id})
        if not existing:
            task = {
                "task_id": task_id,
                "description": ROOT_TASK,
                "interval": ROOT_TASK_INTERVAL,
                "last_run": 0,
                "status": "active",
                "created_at": datetime.utcnow()
            }
            self._collection.update_one({"task_id": task_id}, {"$set": task}, upsert=True)
            logger.info(f"Registered root task: {ROOT_TASK}")

    def start(self):
        if not ENABLE_SCHEDULER:
            return
        self._running = True
        self._tasks_loop = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started.")

    async def stop(self):
        self._running = False
        if self._tasks_loop:
            self._tasks_loop.cancel()
            try:
                await self._tasks_loop
            except asyncio.CancelledError:
                pass
        if hasattr(self, 'client') and self.client:
            self.client.close()
        logger.info("Scheduler stopped.")

    async def _run_loop(self):
        while self._running:
            try:
                tasks = self._collection.find({"status": "active"})
                for task in tasks:
                    now = time.time()
                    if now - task.get("last_run", 0) >= task.get("interval", 3600):
                        await self._execute_task(task)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
            
            await asyncio.sleep(60) # Check every minute

    async def _execute_task(self, task_data: Dict[str, Any]):
        import globals as backend_globals
        if not backend_globals.agent_runtime:
            try:
                from agents.runtime_factory import RuntimeFactory
                from model_providers import LocalLLModel
                from rag import LocalRAG
                from agents.context_storage import create_context_storage
                
                logger.info("Initializing AgentRuntime for scheduled task...")
                local_model = LocalLLModel.init_local_model()
                l_rag = backend_globals.local_rag or LocalRAG(local_model)
                c_storage = backend_globals.context_storage or create_context_storage()
                
                backend_globals.agent_runtime = RuntimeFactory.create_with_all_agents(
                    local_model,
                    rag_instance=l_rag,
                    permission_manager=backend_globals.permission_manager,
                    context_storage=c_storage,
                    session_id="background_scheduler"
                )
            except Exception as e:
                logger.error(f"Failed to initialize AgentRuntime in scheduler: {e}")
                return

        runtime = backend_globals.agent_runtime
        task_id = task_data["task_id"]
        description = task_data["description"]
        
        logger.info(f"Executing scheduled task: {task_id} - {description}")
        
        # Update last run immediately to prevent double execution
        self._collection.update_one({"task_id": task_id}, {"$set": {"last_run": time.time()}})

        try:
            # Create a dedicated session for this run
            session_id = f"task_{task_id}_{int(time.time())}"
            runtime.session_id = session_id
            runtime.reset()
            
            # Execute agent
            state = await runtime.execute(description)
            
            # Record execution history
            self._collection.update_one(
                {"task_id": task_id}, 
                {"$push": {"history": {
                    "timestamp": datetime.utcnow(),
                    "status": state.status.value,
                    "summary": str(state.final_result)[:500] if state.final_result else ""
                }}}
            )
            
            logger.info(f"Task {task_id} completed with status: {state.status.value}")
            
        except Exception as e:
            logger.error(f"Failed to execute task {task_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._collection.update_one(
                {"task_id": task_id}, 
                {"$push": {"history": {
                    "timestamp": datetime.utcnow(),
                    "status": "failed",
                    "error": str(e)
                }}}
            )
