from scipy import linalg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import ndimage, misc
# define square matrix
two_d_array = np.array([ [4,5], [3,2] ])
#pass value to function inv()
linalg.inv( two_d_array )
print(linalg.inv( two_d_array ))
eg_val, eg_vect = linalg.eig(two_d_array)
#get eigenvalues
print(eg_val)
#get eigenvectors
print(eg_vect)


#Frequency in terms of Hertz
fre  = 5 
#Sample rate
fre_samp = 50
t = np.linspace(0, 2, 2 * fre_samp, endpoint = False )
a = np.sin(fre  * 2 * np.pi * t)
figure, axis = plt.subplots()
axis.plot(t, a)
axis.set_xlabel ('Time (s)')
axis.set_ylabel ('Signal amplitude')
plt.show()

from scipy import fftpack

A = fftpack.fft(a)
frequency = fftpack.fftfreq(len(a)) * fre_samp
figure, axis = plt.subplots()

axis.stem(frequency, np.abs(A))
axis.set_xlabel('Frequency in Hz')
axis.set_ylabel('Frequency Spectrum Magnitude')
axis.set_xlim(-fre_samp / 2, fre_samp/ 2)
axis.set_ylim(-5, 110)
plt.show()


def function(a):
       return   a*2 + 20 * np.sin(a)
plt.plot(a, function(a))
plt.show()
#use BFGS algorithm for optimization
optimize.fmin_bfgs(function, 0) 

#get face image of panda from misc package
panda = misc.face()
#plot or show image of face
plt.imshow( panda )
plt.show()
#Flip Down using scipy misc.face image  
flip_down = np.flipud(misc.face())
plt.imshow(flip_down)
plt.show()
panda_rotate = ndimage.rotate(panda, 135)
plt.imshow(panda_rotate)
plt.show()
