Installation
============

First, the Python 3 (ver. >=3.7, compatibility with lower versions cannot be guaranteed, because it was not tested)
is needed. The latest Python distribution can be downloaded from https://www.python.org/downloads/. It is recommended to
install 64 bit version. During installation, be sure to check **Add Python to PATH**. This will add paths of your new
python and ./Scripts directories to environmental variables.

After installation of latest Python distribution, some additional packages have to be installed. Packages needed for this 
program to work are listed below:


* `PyQt5 <https://pypi.org/project/PyQt5/>`_ - cross-platform toolkit for creating GUI (python binding)
* `LMFIT <https://lmfit.github.io/lmfit-py/index.html>`_ - library used for fitting
* `matplotlib <https://matplotlib.org/>`_ - library used for making beautiful static plots
* `QtConsole <https://ipython.org/ipython-doc/dev/interactive/qtconsole.html>`_ - IPython console widget for PyQt
* `Numba <http://numba.pydata.org/>`_ - speeds up the low-level python code by compiling it to the machine code
  just in time (when it is needed)
* `(PyQtGraph) <http://www.pyqtgraph.org/>`_ - for interactive plots, **not need to install** (already part of
  this program, because some changes had to be made to the original code of the library)
* `(NumPy) <https://www.numpy.org/>`_ - numerical python, used for all calculations, installed automatically with LMFIT
* `(SciPy) <https://scipy.org/scipylib/index.html>`_ - Scientific Python, installed automatically with LMFIT

The common way is installation through ``pip``::

	$ pip install [--name-of-the-package--]
	

Or, easily (for Windows), launching the ``install_packages.bat`` file will install all packages automatically.

When working on computer with different versions of Python (2.x vs 3.x), be sure that the commands are executed from the
directory of latest Python 3 distribution, usually located in (well it depends on the setup in PATH in environmental variables,
if `python` command corresponds to latest Python distribution, the above installation will work with no problem, for
more info, see this `YouTube video <https://www.youtube.com/watch?v=OdIHeg4jj2c>`_)::

	$ cd "C:\Users\[--username--]\AppData\Local\Programs\Python\Python37\Scripts"
	
So, for Windows users, open the ``install_packages_multiple_py_versions.bat`` file and change the path to your Python 3.x **Scripts**
folder location and execute.
	
Also, make sure that *.py* and *.pyw* files are associated with Python launchers ``py.exe`` and ``pyw.exe``, respectively, located in C:\\Windows\\.
This should be done automatically with Python 3 installation. Python launcher reads this directive at the beginning of the *.py* or *.pyw* file to distinguish,
what version of Python should be used::

	#!python3

So, when some programs that run on Python 2.7 stop working after installation of new Python 3.x, be sure to add this directive at the begining of
the *.py* or *.pyw* file::
	
	#!python2
	

  
  
