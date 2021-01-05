import sys

class Logger(object):

    _instance = None

    def __init__(self, console_message, statusbar_message):
        Logger._instance = self

        # functions
        self.statusbar_message = statusbar_message
        self.console_message = console_message

    @classmethod
    def console_message(cls, text, add_new_line=True):
        if cls._instance is None:
            return

        cls._instance.console_message(str(text), add_new_line)

    @classmethod
    def status_message(cls, text, delay=3000):
        if cls._instance is None:
            return

        cls._instance.statusbar_message(str(text), delay)

    @classmethod
    def message(cls, text, delay=3000):
        if cls._instance is None:
            return

        cls._instance.console_message(str(text))
        cls._instance.statusbar_message(str(text), delay)


class Transcript:
    """Class used to redirect std output to pyqtconsole"""

    def __init__(self):
        self.terminal = sys.stdout

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        Logger.console_message(message, False)

    def flush(self):
        pass

