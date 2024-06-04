import fabrik
import numpy as np
a=np.array(((0,0,0),(3,4,0),(7,7,0),(10,7,4),(10,7,8)))
b=np.array((5,5,5,4))

model=fabrik.Structure(a,b)
model.plot()