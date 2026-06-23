import asyncio
import sys
import os
import uuid
from typing import Any
from clients.remote_llm import RemoteLLM, BACKEND_URL_DEFAULT
from clients.agents.runtime_factory import RuntimeFactory


BACKEND_ENV_VAR = "LLP_BACKEND_URL"


class ContextManager:
    def __init__(self):
        self._files: dict[str, str] = {}

    def add_file(self, path: str) -> str | None:
        if not os.path.isfile(path):
            return f"FileNotFound: {path}"
        with open(path) as f:
            content = f.read()
        name = os.path.basename(path)
        self._files[name] = content
        return None

    def add_directory(self, path: str) -> list[str]:
        errors = []
        if not os.path.isdir(path):
            return [f"DirectoryNotFound: {path}"]
        for root, _, files in os.walk(path):
            for fname in files:
                fpath = os.path.join(root, fname)
                error = self.add_file(fpath)
                if error:
                    errors.append(error)
        return errors

    def clear(self):
        self._files.clear()

    def get_context(self) -> dict:
        return dict(self._files)

    def summary(self) -> str:
        if not self._files:
            return "No files in context"
        parts = [f"Context ({len(self._files)} files):"]
        for name, content in self._files.items():
            lines = len(content.splitlines())
            parts.append(f"  {name}: {lines} lines")
        return "\n".join(parts)


class CliRunner:
    def __init__(self, backend_url: str):
        self._llm = RemoteLLM(backend_url)
        self._context = ContextManager()

    async def run(self):
        session_id = str(uuid.uuid4())
        runtime = RuntimeFactory.create_with_all_agents(
            self._llm,
            session_id=session_id,
        )

        print(f"Session: {session_id}")
        print("Type /help for commands")

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                handled = await self._handle_command(user_input)
                if handled:
                    continue
                else:
                    break

            runtime.state.context["attached_files"] = self._context.get_context()

            async def display_callback(chunk: str):
                print(chunk, end="", flush=True)

            state = await runtime.execute(
                user_input,
                start_agent="qa",
                stream_callback=display_callback,
            )

            final = state.final_result or state.error_message or ""
            print(final)

    async def _handle_command(self, cmd: str) -> bool:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("/exit", "/quit"):
            print("Goodbye")
            return False

        if command == "/help":
            self._show_help()
            return True

        if command == "/add":
            if not arg:
                print("Usage: /add <path>")
                return True
            error = self._context.add_file(arg)
            if error:
                print(f"Error: {error}")
            else:
                print(f"Added: {arg}")
            return True

        if command == "/add-dir":
            if not arg:
                print("Usage: /add-dir <path>")
                return True
            errors = self._context.add_directory(arg)
            if errors:
                for e in errors:
                    print(f"Error: {e}")
            else:
                print(f"Added directory: {arg}")
            return True

        if command == "/context":
            print(self._context.summary())
            return True

        if command == "/clear":
            self._context.clear()
            print("Context cleared")
            return True

        print(f"Unknown command: {command}")
        return True

    def _show_help(self):
        print("Commands:")
        print("  /add <path>      Add a file to context")
        print("  /add-dir <path>  Add a directory to context")
        print("  /context         Show current context contents")
        print("  /clear           Clear context")
        print("  /exit            Exit")
        print("  /help            Show this help")
        print("  Any other text   Send as a query to the agent")


def main():
    backend_url = os.environ.get(BACKEND_ENV_VAR, BACKEND_URL_DEFAULT)
    runner = CliRunner(backend_url)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
