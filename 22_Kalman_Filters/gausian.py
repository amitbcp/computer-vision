from math import *

def gaussian(mu,sigma2,x) :
    return 1/sqrt(2.*pi*sigma2) * exp(-.5 * (x-mu)**2/sigma2)

print("Gaussian for mu = 10, sigma2 = 4. x = 8. : {}".format(gaussian(10.,4.,8.)))

def gaussian_update(mean1,var1,mean2,var2):
    new_mean=  (var2*mean1 + var1*mean2) / (var1 + var2)
    new_var = (var1 * var2)/(var1+var2)
    return [new_mean,new_var]

print("Updated Gausuian Parameters : {}".format(gaussian_update(10.,8.,13., 2.)))
print("Updated Gausuian Parameters : {}".format(gaussian_update(10.,4.,12., 4.)))

def gaussian_predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

print("Predicted Gausuian Parameters : {}".format(gaussian_predict(10.,8.,13., 2.)))
print("Predicted Gausuian Parameters : {}".format(gaussian_predict(10.,4.,12., 4.)))

measurements = [5., 6., 7., 9., 10.]
motion = [1., 1., 2., 1., 1.]
measurement_sig = 4.
motion_sig = 2.
mu = 0.
sig = 10000.

for n in range(len(measurements)) :
    [mu,sig] = gaussian_update(mu,sig,measurements[n],measurement_sig)
    print("Update : {}".format([mu,sig]))
    [mu,sig] = gaussian_predict(mu,sig,motion[n],motion_sig)
    print("Update : {}".format([mu,sig]))
