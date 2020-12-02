from PyQt5.QtCore import Qt
from spectrum import Spectrum, SpectrumList
from user_namespace import add_to_list, update_view, redraw_all_spectra


class GenericItem:

    def __init__(self, name, info, parent=None):

        self.check_state = Qt.Unchecked

        self.name = str(name)
        self.info = str(info)

        self.parent = parent
        self.children = []

        self.setParent(parent)

    def is_root(self):
        return self.parent is None

    def isChecked(self):
        return self.check_state == Qt.Checked or self.check_state == Qt.PartiallyChecked

    def setParent(self, parent, row=None):
        if parent != None:
            self.parent = parent
            self.parent.appendChild(self, row)
        else:
            self.parent = None

    def appendChild(self, child, row=None):
        self.children.insert(row if row is not None else len(self.children), child)
        # self.children.append(child)

    def childAtRow(self, row):
        try:
            return self.children[row]
        except IndexError:
            # print("index error")
            return

    def move_child(self, row_of_child: int, to_row=0):
        self.children.insert(to_row, self.children.pop(row_of_child))

    def rowOfChild(self, child):
        for i, item in enumerate(self.children):
            if item == child:
                return i
        return -1

    def removeChildAtRow(self, row):
        # try:

        del self.children[row]

        # value = self.children[row]
        # self.children.remove(value)
        # except:
        #     pass

        return True

    def row(self):
        if self.parent is not None:
            return self.parent.rowOfChild(self)

    def removeChild(self, child):
        self.children.remove(child)

    def root(self):
        parent = self.parent
        while parent.parent is not None:
            parent = parent.parent
        return parent

    def __len__(self):
        return len(self.children)

    def __getitem__(self, item):
        return self.children[item]

    def __iter__(self):
        return iter(self.children)

    def add_to_list(self, spectra=None):
        if self.__class__ == SpectrumItem or self.__class__ == SpectrumItemGroup:
            add_to_list(self if spectra is None else spectra)
            # print('add_to_list - generic')

    def _redraw_all_spectra(self):
        redraw_all_spectra()

        # print('_redraw_all_spectra - generic')

    def _update_view(self):
        update_view()
        # print('_update_view - generic')


class SpectrumItem(GenericItem, Spectrum):

    @classmethod
    def init(cls, spectrum, parent=None):
        spectrum.__class__ = cls
        GenericItem.__init__(spectrum, spectrum.name, '', parent=parent)
        return spectrum

    def is_in_group(self):
        return not self.parent.is_root()

    def is_top_level(self):
        return self.parent.is_root()


class SpectrumItemGroup(GenericItem, SpectrumList):

    def __init__(self, name, info='', parent=None):
        GenericItem.__init__(self, name, info, parent=parent)
        # super(SpectrumItemGroup, self).__init__(name, info, parent=parent)

    def is_top_level(self):
        return True
