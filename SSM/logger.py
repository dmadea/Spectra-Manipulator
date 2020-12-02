

class Logger(object):

    _instance = None

    def __init__(self, console_message, statusbar_message):
        Logger._instance = self

        # functions
        self.statusbar_message = statusbar_message
        self.console_message = console_message

    @classmethod
    def console_message(cls, text):
        if cls._instance is None:
            return

        cls._instance.console_message(str(text))

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













