from PyQt5 import QtCore

from PyQt5.QtWidgets import QDockWidget, QWidget

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
import spectramanipulator


# great help from here https://stackoverflow.com/questions/11513132/embedding-ipython-qt-console-in-a-pyqt-application/12375397#12375397
# just copied, works like charm :)

# qtconsole package is not yet compatible with pyqt6
# https://github.com/jupyter/qtconsole/issues/504
# in future, it will be available

class ConsoleWidget(RichJupyterWidget):

    def __init__(self, app, customBanner=None,  *args, **kwargs):
        super(ConsoleWidget, self).__init__(*args, **kwargs)
        self.app = app

        if customBanner is not None:
            self.banner = customBanner

        self.font_size = 6
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel(show_banner=False)
        kernel_manager.kernel.gui = 'qt'
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            self.app.exit()

        self.exit_requested.connect(stop)

    def set_focus(self):
        # TODO--- does not work...
        self._control.setFocus()

    def push_vars(self, variableDict):
        """
        Given a dictionary containing name / value pairs, push those variables
        to the Jupyter console widget
        """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clear(self):
        """
        Clears the terminal
        """
        self._control.clear()

        # self.kernel_manager

    def print_text(self, text):
        """
        Prints some plain text to the console
        """
        # self._append_plain_text(text)
        self.append_stream(text)

    def print_html(self, text):
        self._append_html(text, True)

    def execute_command(self, command, hidden=True):
        """
        Execute a command in the frame of the console widget
        """
        self._execute(command, hidden)


class Console(QDockWidget):
    """
    Console window used for advanced commands, debugging,
    logging, and profiling.
    """

    _instance = None

    def __init__(self, app, parentWindow):
        QDockWidget.__init__(self, parentWindow)
        self.app = app
        self.setTitleBarWidget(QWidget())
        self.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.setVisible(False)

        banner = f"""Simple Spectra Manipulator console based on IPython, version {spectramanipulator.__version__}. Numpy package was imported as np and matplotlib.pyplot as plt. Three variables are setup:
    item - this is used to interact with spectra in TreeWidget and perform various calculations
    tree_widget - instance of TreeWidget
    main - instance of Main (Main Window)

Enjoy.
        
"""
        Console._instance = self

        self.banner = banner
        self.console_widget = None
        self._plain_text_buffer = []
        self._html_buffer = []
        self._cmd_buffer = []
        self._vars_buffer = []
        self.setWidget(QWidget())

    def _ensure_console_widget(self):
        if self.console_widget is not None:
            return

        self.console_widget = ConsoleWidget(self.app, self.banner)
        self.setWidget(self.console_widget)

        for txt in self._plain_text_buffer:
            self.console_widget.print_text(txt)
        for html in self._html_buffer:
            self.console_widget.print_html(html)
        for cmd, hidden in self._cmd_buffer:
            self.console_widget.execute_command(cmd, hidden)
        for var_dict in self._vars_buffer:
            self.console_widget.push_vars(var_dict)

        self._plain_text_buffer.clear()
        self._html_buffer.clear()
        self._cmd_buffer.clear()
        self._vars_buffer.clear()

    def setVisible(self, visible):
        """
        Override to set proper focus.
        """
        if visible:
            self._ensure_console_widget()
        QDockWidget.setVisible(self, visible)
        if visible and self.console_widget is not None:
            self.console_widget.setFocus()

    @staticmethod
    def showMessage(message, add_new_line=True):
        if Console._instance is not None:
            string = '\n' + message if add_new_line else message
            if Console._instance.console_widget is None:
                Console._instance._plain_text_buffer.append(string)
            else:
                Console._instance.console_widget.print_text(string)

    def print_html(self, text, add_new_line=True):
        string = '\n' + text if add_new_line else text
        if self.console_widget is None:
            self._html_buffer.append(string)
            return
        self.console_widget.print_html(string)

    @staticmethod
    def execute_command(cmd, hidden=True):
        if Console._instance is not None:
            if Console._instance.console_widget is None:
                Console._instance._cmd_buffer.append((cmd, hidden))
            else:
                Console._instance.console_widget.execute_command(cmd, hidden)

    @staticmethod
    def push_variables(variable_dict):
        if Console._instance is not None:
            if Console._instance.console_widget is None:
                Console._instance._vars_buffer.append(variable_dict)
            else:
                Console._instance.console_widget.push_vars(variable_dict)
