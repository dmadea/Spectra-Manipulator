import numpy as np
from PyQt5.QtGui import QColor


def int_default_color_scheme(counter):
    colors = [
        (255, 0, 0, 255),  # red
        (0, 255, 0, 255),  # green
        (0, 0, 255, 255),  # blue
        (0, 0, 0, 255),  # black
        (255, 255, 0, 255),  # yellow
        (255, 0, 255, 255),  # magenta
        (0, 255, 255, 255),  # cyan
        (155, 155, 155, 255),  # gray
        (155, 0, 0, 255),  # dark red
        (0, 155, 0, 255),  # dark green
        (0, 0, 155, 255),  # dark blue
        (155, 155, 0, 255),  # dark yellow
        (155, 0, 155, 255),  # dark magenta
        (0, 155, 155, 255)  # dark cyan
    ]

    return QColor(*colors[counter % len(colors)])


def intColorGradient(index, hues, grad_mat, reversed=False):
    """User defined gradient"""

    ind = int(index) % hues

    pos = ((hues - ind) / hues) if reversed else (ind / hues)

    positions = grad_mat[:, 0].flatten()
    idx_pos = np.searchsorted(positions, pos, side="right")
    idx_pos -= 1 if idx_pos > 0 else 0
    idx_pos -= 1 if idx_pos == len(positions) - 1 else 0

    # position within the interval of colors
    x = (pos - positions[idx_pos]) / (positions[idx_pos + 1] - positions[idx_pos])

    # calculate the resulting color as a linear combination of colors in the interval
    color_vector = (1 - x) * grad_mat[idx_pos, 1:] + x * grad_mat[idx_pos + 1, 1:]  # RGBA

    color = QColor(*color_vector)
    # color.setAlpha(int(color_vector[-1]))
    return color


def intColor(index, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255,
             reversed=False):
    """
    Creates a QColor from a single index. Useful for stepping through a predefined list of colors.

    The argument *index* determines which color from the set will be returned. All other arguments determine what the set of predefined colors will be

    Colors are chosen by cycling across hues while varying the value (brightness).
    By default, this selects from a list of 9 hues."""
    hues = int(hues)
    values = int(values)
    ind = int(index) % (hues * values)
    indh = ind % hues
    indv = ind // hues
    if values > 1:
        v = minValue + indv * ((maxValue - minValue) / (values - 1))
    else:
        v = maxValue

    if reversed:
        h = minHue + ((hues - indh - 1) * (maxHue - minHue)) / hues
    else:
        h = minHue + (indh * (maxHue - minHue)) / hues

    c = QColor()
    c.setHsv(h, sat, v)
    c.setAlpha(alpha)
    return c