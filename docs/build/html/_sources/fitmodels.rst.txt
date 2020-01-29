Fitting models
==============

The file ``fitmodels.py`` contains generic models used for fitting.
Available models are listed below. In order to create own model, please add your model to ``userfitmodels.py``. 
The other file was added so that the original file is not modified. For creating your own model, you can get inspiration
from generic models in ``fitmodels.py``.

In classical kinetic models, the concentration at time zero (:math:`c(0) = c_0`) is omitted as a fitting 
parameter for :math:`1^{\mathrm{st}}` order kinetics, because the rate does not depend on :math:`c_0`. This parameter
is included for variable order and special models (eg. Photobleaching), because the rate depends on :math:`c_0`.



A→B (1st order)
---------------

.. autoclass:: fitmodels.AB1stModel
	:members:
	
	
A→B (var order)
---------------
	
	
.. autoclass:: fitmodels.ABVarOrderModel
	:members: 
	

A→B (Mixed 1st and 2nd order)
-----------------------------
	
.. autoclass:: fitmodels.AB_mixed12Model
	:members: 
	
A→B→C (1st order)
-----------------
	
.. autoclass:: fitmodels.ABC1stModel
	:members: 
	
A→B→C (var order)
-----------------

.. autoclass:: fitmodels.ABCvarOrderModel
	:members: 


A→B→C→D (1st order)
-------------------

.. autoclass:: fitmodels.ABCD1stModel
	:members: 
	

A→B→C→D (var order)
-------------------

.. autoclass:: fitmodels.ABCDvarOrderModel
	:members: 
	
	
A→B, C→D (1st order)
--------------------

.. autoclass:: fitmodels.ABCD_seqModel
	:members: 

	
Linear
------

.. autoclass:: fitmodels.LinearModel
	:members: 

	
Photobleaching
--------------

.. autoclass:: fitmodels.Photobleaching_Model
	:members: 

	
	
	
	
	
	
	
	
	
	

	
	
	
	
	
	
	
	
	
	
	