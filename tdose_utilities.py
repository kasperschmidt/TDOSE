# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import os
import sys
import pyfits
import scipy.ndimage
import matplotlib.pylab as plt
import pdb
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def build_2D_cov_matrix(sigmax,sigmay,angle,verbose=True):
    """
    Build a covariance matrix for a 2D multivariate Gaussian

    --- INPUT ---
    sigmax        Standard deviation of the x-compoent of the multivariate Gaussian
    sigmay        Standard deviation of the y-compoent of the multivariate Gaussian
    angle         Angle to rotate matrix by in degrees (clockwise) to populate covariance cross terms

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    covmatrix = tu.build_2D_cov_matrix(3,1,35)

    """
    if verbose: print ' - Build 2D covariance matrix with varinaces (x,y)=('+str(sigmax)+','+str(sigmay)+\
                      ') and then rotated '+str(angle)+' degrees'
    cov_orig      = np.zeros([2,2])
    cov_orig[0,0] = sigmay**2.0
    cov_orig[1,1] = sigmax**2.0

    angle_rad     = angle * np.pi/180.0
    c, s          = np.cos(angle_rad), np.sin(angle_rad)
    rotmatrix     = np.matrix([[c, -s], [s, c]])

    cov_rot       = np.dot(np.dot(rotmatrix,cov_orig),np.transpose(rotmatrix))  # performing rot * cov * rot^T

    return cov_rot
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_noisy_cube(cube,type='poisson',gauss_std=0.5,verbose=True):
    """
    Generate noisy cube based on input cube.

    --- INPUT ---
    cube
    type        Type of noise to generate
                  poisson    Generates poissonian (integer) noise
                  gauss      Generates gaussian noise for a gaussian with standard deviation gauss_std=0.5
    gauss_std   Standard deviation of noise if type='gauss'
    verbose     Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    datacube = np.ones(([3,3,3])); cube[0,1,1]=5; cube[1,1,1]=6; cube[2,1,1]=8
    cube_with_noise = tu.gen_noisy_cube(datacube,type='gauss',gauss_std='0.5')

    """
    if verbose: print ' - Generating "'+type+'" noise on data cube'
    if type == 'poisson':
        cube_with_noise = np.random.poisson(lam=cube, size=None)
    elif type == 'gauss':
        cube_with_noise = cube + np.random.normal(loc=np.zeros(cube.shape),scale=gauss_std, size=None)
    else:
        sys.exit(' ---> type="'+type+'" is not valid in call to mock_cube_sources.generate_cube_noise() ')

    return cube_with_noise
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_2Dgauss(size,cov,scale,verbose=True,show2Dgauss=False):
    """
    Generating a 2D gaussian with specified parameters

    --- INPUT ---
    size    The dimensions of the array to return. Expects [y-size,x-size].
            The 2D gauss will be positioned in the center of a (+/-x-size/2., +/-y-size/2) sized array
    cov     Covariance matrix of gaussian, i.e., variances and rotation
            Can be build with cov = build_2D_cov_matrix(stdx,stdy,angle)

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    covmatrix   = tu.build_2D_cov_matrix(4,1,5)
    gauss2Dimg  = tu.gen_2Dgauss([20,40],covmatrix,5,show2Dgauss=True)

    covmatrix   = tu.build_2D_cov_matrix(4,1,0)
    gauss2Dimg  = tu.gen_2Dgauss([20,40],covmatrix,25,show2Dgauss=True)

    """
    if verbose: print ' - Generating multivariate_normal object for generating 2D gauss'
    mvn = multivariate_normal([0, 0], cov)

    if verbose: print ' - Setting up grid to populate with 2D gauss PDF'
    x, y = np.mgrid[-np.round(size[0]/2.):np.round(size[0]/2.):1.0, -np.round(size[1]/2.):np.round(size[1]/2.):1.0]
    pos = np.zeros(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    gauss2D = mvn.pdf(pos)

    if verbose: print ' - Scaling 2D gaussian by a factor '+str(scale)
    gauss2D = gauss2D*scale

    if show2Dgauss:
        if verbose: print ' - Displaying resulting image of 2D gaussian'
        plt.imshow(gauss2D,interpolation='none')
        plt.title('Generated 2D Gauss')
        plt.show()

    return gauss2D
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def plot_matrix_array():
    return None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def build_fits_wcs():
    return None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =