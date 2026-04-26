from . import Plugin, tool
import subprocess
from typing import Annotated, override
from pathlib import Path
import base64
import datetime
from ..tools import ToolResult
from ..models.base import File


class Bash(Plugin):
    @override
    def get_instructions(self) -> str | None:
        s = f"CURRENT TIME: {datetime.datetime.now().isoformat()}\n"
        s += f"CURRENT WORKING DIRECTORY: `{str(Path.cwd())}`.\n"
        return s

    @tool(name="Bash")
    async def run_bash_command(
        self,
        command: Annotated[str, "The bash command to run"],
        cwd: Annotated[
            str | None, "The optional working directory for the command"
        ] = None,
    ) -> str:
        """Run a bash command and return the output."""
        await self.agent.user_consent_guard("Run command: " + command)
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.stdout.strip()

    @tool(name="LoadFile")
    async def load_file(
        self, path: Annotated[str, "The path to the file to load."]
    ) -> ToolResult:
        """
        Load NON-TEXT files into the conversation so that you can inspect their content.
        Only images (.png, .jpg, .jpeg) and PDFs (.pdf) are supported. No directory is allowed, only file paths.
        For text files, directly read them using the `cat` command in Bash.
        """
        if not path:
            return ToolResult(output="No file path provided.")

        p = Path(path)

        if not p.exists():
            return ToolResult(output=f"File '{path}' does not exist.")

        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            media_type = f"image/png" if p.suffix.lower() == ".png" else f"image/jpeg"
        elif p.suffix.lower() == ".pdf":
            media_type = "application/pdf"
        else:
            return ToolResult(output=f"Unsupported file type: '{p.suffix}'.")

        bytes = p.read_bytes()
        base64_url = f"data:{media_type};base64,{base64.b64encode(bytes).decode()}"
        file = File(media_type=media_type, data=base64_url)
        return ToolResult(files=[file], output=f"Loaded file '{path}' successfully.")
