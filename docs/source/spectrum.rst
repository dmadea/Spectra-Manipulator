Class Spectrum and SpectrumList
===============================


All operations return *self*, so they can be easily chained together:

	>>> item[0].interpolate(spacing=1, kind='linear').baseline_correct(400, 500).cut(220, 500)

If *item[0]* is group of spectra, this line of code interpolates all spectra by linear intepolation with 1 nm spacing, 
performs a baseline correction (subtracts the average in 400 - 500 range) and cuts the spectra to 200 - 500 nm region.



Spectrum
--------


.. autoclass:: spectrum.Spectrum
	:members:
	
	
	
SpectrumList
------------


.. autoclass:: spectrum.SpectrumList
	:members:
	

