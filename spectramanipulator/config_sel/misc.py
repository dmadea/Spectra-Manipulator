
from PyQt5 import QtCore, QtWidgets

def getWidgetState(qWindow):
    """ Gets the QWindow or QWidget state as a QByteArray.

        Since Qt does not provide this directly we hack this by saving it to the QSettings
        in a temporary location and then reading it from the QSettings.

        :param widget: A QWidget that has a saveState() methods
    """
    settings = QtCore.QSettings()
    settings.beginGroup('temp_conversion')
    try:
        settings.setValue("winState", qWindow.saveState())
        return bytes(settings.value("winState"))
    finally:
        settings.endGroup()



def setWidgetSizePolicy(widget, horPolicy=None, verPolicy=None):
    """ Sets the size policy of a widget.
    """
    sizePolicy = widget.sizePolicy()

    if horPolicy is not None:
        sizePolicy.setHorizontalPolicy(horPolicy)

    if verPolicy is not None:
        sizePolicy.setVerticalPolicy(verPolicy)

    widget.setSizePolicy(sizePolicy)
    return sizePolicy





def widgetSubCheckBoxRect(widget, option):
    """ Returns the rectangle of a check box drawn as a sub element of widget
    """
    opt = QtWidgets.QStyleOption()
    opt.initFrom(widget)
    style = widget.style()
    return style.subElementRect(QtWidgets.QStyle.SE_ViewItemCheckIndicator, opt, widget)
