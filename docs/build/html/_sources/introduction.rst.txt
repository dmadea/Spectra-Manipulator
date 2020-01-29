Introduction
============


What is Simple Spectra Manipulator?
-----------------------------------

Simple Spectra Manipulator (SSM) is a small program with GUI that enables to import and export
various types of spectral (not only) files, performs simple or advanced manipulation with them
and visualize them. The program is intended to be intuitive, user-friendly, easy to use and fast,
thus, to make the work more effective. The easy and simple import and export capabilities are emphasized,
such as transferring data to and from Excel/Origin. The program includes running IPython console which 
can be used for low-level interaction with the program and non-trivial data manipulation and calculations.
This program is not intended to replace `Origin <https://www.originlab.com/>`_ or similar software packages.
For publication quality figures, use Origin, even though nice graphs can be made with this software as well.
These can be used for presentations purposes.


This program uses PyQtGraph library for plotting, because it is fast for interaction, in the contrary of
for example, matplotlib library, which is commonly used for python. Matplotlib produces nice plots, 
but is not quite usable for interactive applications because of its slow speed. Matplotlib library is
useful for static plots. In this program, it is used to plot 2D confidence intervals though the console
though (see tutorial).



What can it do?
---------------

Basic features for data manipulation:

* Baseline correction, normalization, cutting the spectra, interpolation and 
  extending by zeros
  
* Curve fitting based on lmfit module with correct confidence intervals estimation


Intermediate features for data manipulation (only accessible through console):

* Arithmetic operations among numbers, spectra and group of spectra

* Integration, differentiation, Savitzky-Golay filter, power spectrum (FFT),
  transposition, finding maximum at selected range, etc...

  
  
