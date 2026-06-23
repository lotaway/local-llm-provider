import asyncio
from acp import run_agent
from model_providers import LocalLLModel
from adapters.protocol_bridge import ProtocolBridge


def build_bridge() -> ProtocolBridge:
    model = LocalLLModel.init_local_model()
    return ProtocolBridge(model)


async def run_stdio_server():
    bridge = build_bridge()
    await run_agent(bridge)


def main():
    asyncio.run(run_stdio_server())
