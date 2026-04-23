# -*- coding: utf-8 -*-
import types

from pyqtgraph.graphicsItems.AxisItem import AxisItem as PgAxisItem


def patch_axisitem_respect_disable_auto_si_prefix(axis) -> None:
    """
    When ``autoSIPrefix`` is off, pyqtgraph can still change tick scaling for axes that
    have a visible label (left/bottom), so tick numbers differ from mirrored top/right
    axes. Keep ``autoSIPrefixScale`` at 1.0 when disabled, matching the unlabeled axes.
    """
    if getattr(axis, '_sm_si_respect_patched', False):
        return

    def update_auto_si_prefix(self):
        if not self.autoSIPrefix:
            self.autoSIPrefixScale = 1.0
            self.labelUnitPrefix = ''
            self._updateLabel()
        else:
            PgAxisItem.updateAutoSIPrefix(self)

    axis.updateAutoSIPrefix = types.MethodType(update_auto_si_prefix, axis)
    axis._sm_si_respect_patched = True
