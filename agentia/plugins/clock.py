import datetime
from . import Plugin
from .. import tool


class ClockPlugin(Plugin):
    @tool
    def get_current_utc_time(self):
        """Get the current UTC time in ISO format"""
        return datetime.datetime.now(datetime.timezone.utc).isoformat()
