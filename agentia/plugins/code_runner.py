from contextlib import redirect_stderr
from . import Plugin, tool
from typing import Annotated
import traceback


class CodeRunner(Plugin):
    @tool
    def execute(self, python_code: Annotated[str, "The python code to run."]):
        """Execute python code and return the result. The expression must be an valid python expression that can be execuated by `eval()`."""
        from contextlib import redirect_stdout
        import io

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
