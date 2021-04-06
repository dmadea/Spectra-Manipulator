from PyQt5 import QtCore

from PyQt5.QtWidgets import QDockWidget, QWidget

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
import spectramanipulator


# great help from here https://stackoverflow.com/questions/11513132/embedding-ipython-qt-console-in-a-pyqt-application/12375397#12375397
# just copied, works like charm :)

# qtconsole package is not yet compatible with pyqt6

class ConsoleWidget(RichJupyterWidget):

    def __init__(self, customBanner=None, *args, **kwargs):
        super(ConsoleWidget, self).__init__(*args, **kwargs)

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
            guisupport.get_app_qt().exit()

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

    def __init__(self, parentWindow):
        QDockWidget.__init__(self, parentWindow)
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

        # Add console window
        self.console_widget = ConsoleWidget(banner)
        self.setWidget(self.console_widget)

    def setVisible(self, visible):
        """
        Override to set proper focus.
        """
        QDockWidget.setVisible(self, visible)
        if visible:
            self.console_widget.setFocus()

    @staticmethod
    def showMessage(message, add_new_line=True):
        if Console._instance is not None:
            string = '\n' + message if add_new_line else message
            Console._instance.console_widget.print_text(string)

    def print_html(self, text, add_new_line=True):
        string = '\n' + text if add_new_line else text
        self.console_widget.print_html(string)

    @staticmethod
    def execute_command(cmd, hidden=True):
        if Console._instance is not None:
            Console._instance.console_widget.execute_command(cmd, hidden)

    @staticmethod
    def push_variables(variable_dict):
        if Console._instance is not None:
            Console._instance.console_widget.push_vars(variable_dict)
