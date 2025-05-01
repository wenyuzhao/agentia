from contextlib import redirect_stderr
from agentia.message import UserConsentEvent
from . import Plugin
from ..tools import tool
from typing import Annotated
import traceback


class CodePlugin(Plugin):
    def user_consent(self, code: str):
        """Acquire user consent before executing code."""
        if self.agent.user_consent_enabled:
            result = yield UserConsentEvent(
                "Execute this code?", metadata={"code": code}
            )
            if isinstance(result, bool):
                return result
            else:
                return False
        return True

    @tool
    def execute(self, python_code: Annotated[str, "The python code to run."]):
        """Execute python code and return the result. The expression must be an valid python expression that can be execuated by `eval()`."""
        from contextlib import redirect_stdout
        import io

        result = yield from self.user_consent(python_code)
        if not result:
            return {
                "stdout": "",
                "stderr": "",
                "success": False,
                "error": "User consent not granted.",
            }

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(python_code, globals())
                    o = out.getvalue()
                    e = err.getvalue()
                    result = {
                        "stdout": o,
                        "stderr": e,
                        "success": True,
                    }
                except Exception as ex:
                    o = out.getvalue()
                    e = err.getvalue()
                    result = {
                        "stdout": o,
                        "stderr": e,
                        "success": False,
                        "error": str(ex),
                        "traceback": repr(traceback.format_exc()),
                    }

        return result
