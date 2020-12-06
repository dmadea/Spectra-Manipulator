

from PyQt5.QtCore import QMimeData
from pickle import dumps, loads


class PyMimeData(QMimeData):
    """ The PyMimeData wraps a Python instance as MIME data.
    """
    # The MIME type for instances.
    MIME_TYPE = 'application/x-ssm-dragged_items'

    def __init__(self, item_list=None):
        """ Initialise the instance.
        """
        QMimeData.__init__(self)
        # print('init PyMimeData')

        # Keep a local reference to be returned if possible.
        self._local_instance = item_list

        if item_list is not None:
            # We may not be able to pickle the data.
            try:
                pdata = dumps(item_list)
            except Exception as ex:
                print("dumps not successfull", ex.__str__())
                return

            # This format (as opposed to using a single sequence) allows the
            # type to be extracted without unpickling the data itself.
            # self.setData(self.MIME_TYPE, dumps(item_list.__class__) + pdata)

            self.setData(self.MIME_TYPE, pdata)

    @classmethod
    def coerce(cls, md):
        """ Coerce a QMimeData instance to a PyMimeData instance if
        possible.
        """
        # See if the data is already of the right type.  If it is then
        # print('coarse PyMimeData')

        if isinstance(md, cls):
            return md

        # See if the data type is supported.
        if not md.hasFormat(cls.MIME_TYPE):
            return None

        nmd = cls()
        nmd.setData(cls.MIME_TYPE, md.data(cls.MIME_TYPE))

        return nmd

    def instance(self):
        """ Return the instance.
        """
        if self._local_instance is not None:
            return self._local_instance

        # io = StringIO(str(self.data(self.MIME_TYPE)))

        try:
            items_list = loads(self.data(self.MIME_TYPE))

            self._local_instance = items_list

            # Recreate the instance.
            return items_list
        except:
            pass

        return None

        # io = StringIO(str(self.data(self.MIME_TYPE)))
        #
        # try:
        #     # Skip the type.
        #     load(io)
        #
        #     # Recreate the instance.
        #     return load(io)
        # except:
        #     pass
        #
        # return None

    # what is this for??
    def instanceType(self):
        """ Return the type of the instance.
        """
        if self._local_instance is not None:
            return self._local_instance.__class__

        try:
            return loads(str(self.data(self.MIME_TYPE)))
        except:
            pass

        return None