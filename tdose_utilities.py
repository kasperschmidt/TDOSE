# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import os
import datetime
import sys
from astropy import wcs
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from astropy.nddata import Cutout2D
import pyfits
import scipy.ndimage
import tdose_utilities as tu
import tdose_model_FoV as tmf
import astropy.convolution as ac # convolve, convolve_fft, Moffat2DKernel, Gaussian2DKernel
from scipy.stats import multivariate_normal
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

    angle_rad     = (180.0-angle) * np.pi/180.0 # The (90-angle) makes sure the same convention as DS9 is used
    c, s          = np.cos(angle_rad), np.sin(angle_rad)
    rotmatrix     = np.matrix([[c, -s], [s, c]])

    cov_rot       = np.dot(np.dot(rotmatrix,cov_orig),np.transpose(rotmatrix))  # performing rot * cov * rot^T

    return cov_rot
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def normalize_2D_cov_matrix(covmatrix,verbose=True):
    """
    Calculate the normalization foctor for a multivariate gaussian from it's covariance matrix
    However, not that gaussian returned by tu.gen_2Dgauss() is normalized for scale=1

    """
    detcov  = np.linalg.det(covmatrix)
    normfac = 1.0 / (2.0 * np.pi * np.sqrt(detcov) )

    return normfac
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_noisy_cube(cube,type='poisson',gauss_std=0.5,verbose=True):
    """
    Generate noisy cube based on input cube.

    --- INPUT ---
    cube        Data cube to be smoothed
    type        Type of noise to generate
                  poisson    Generates poissonian (integer) noise
                  gauss      Generates gaussian noise for a gaussian with standard deviation gauss_std=0.5
    gauss_std   Standard deviation of noise if type='gauss'
    verbose     Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    datacube        = np.ones(([3,3,3])); datacube[0,1,1]=5; datacube[1,1,1]=6; datacube[2,1,1]=8
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
def gen_psfed_cube(cube,type='gauss',type_param=[0.5,1.0],use_fftconvolution=False,verbose=True):
    """
    Smooth cube with a 2D kernel provided by 'type', i.e., applying a model PSF smoothing to cube

    --- INPUT ---
    cube        Data cube to be smoothed
    type        Type of smoothing kernel to apply
                  gauss      Use 2D gaussian smoothing kernel
                             type_param expected:   [stdev,(stdev_wave_scale)]
                  moffat     Use a 2D moffat profile to represent the PSF
                             type_param expected:   [gamma,alpha,(gamma_wave_scale,alpha_wave_scale)]

                NB: If *wave_scale inputs are provided a list of scales to apply at each wavelength layer
                    (z-direction) of data cube is expected, hence, adding a wavelength dependence to the kernels.


    type_param  List of parameters for the smoothing kernel.
                For expected paramters see notes in description of "type" keyword above.
    verbose     Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    datacube      = np.ones(([3,3,3])); datacube[0,1,1]=5; datacube[1,1,1]=6; datacube[2,1,1]=8
    cube_smoothed = tu.gen_psfed_cube(datacube,type='gauss',type_param=[10.0,[1.1,1.3,1.5]])

    --- EXAMPLE OF USE ---

    """
    if verbose: print ' - Applying a '+type+' PSF to data cube'
    Nparam  = len(type_param)
    Nlayers = cube.shape[0]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if type == 'gauss':
        if Nparam == 1:
            if verbose: print '   No wavelength dependence; duplicating kernel for all layers'
            kernel  = ac.Gaussian2DKernel(type_param[0])
            kernels = [kernel]*Nlayers
        elif Nparam == 2:
            if verbose: print '   Wavelength dependence; looping over layers to generate kernels'
            if Nlayers != len(type_param[1]):
                sys.exit(' ---> The number of wavelength scalings provided ('+str(len(type_param[1]))+
                         ') is different from the number of layers in cube ('+str(Nlayers)+')')
            kernels = []
            for ll in xrange(Nlayers):
                kernel  = ac.Gaussian2DKernel(type_param[0]*type_param[1][ll])
                kernels.append(kernel)
        else:
            sys.exit(' ---> Invalid number of paramters provided ('+str(Nparam)+')')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif type == 'moffat':
        if Nparam == 2:
            if verbose: print '   No wavelength dependence; duplicating kernel for all layers'
            kernel  = ac.Moffat2DKernel(type_param[0],type_param[1])
            kernels = [kernel]*Nlayers
        elif Nparam == 4:
            if verbose: print '   Wavelength dependence; looping over layers to generate kernels'
            if (Nlayers != len(type_param[2])) or (Nlayers != len(type_param[3])):
                sys.exit(' ---> The number of wavelength scalings provided ('+str(len(type_param[2]))+
                         ' and '+str(len(type_param[3]))+
                         ') are different from the number of layers in cube ('+str(Nlayers)+')')
            kernels = []
            for ll in xrange(Nlayers):
                kernel  = ac.Moffat2DKernel(type_param[0]*type_param[2][ll],type_param[1]*type_param[3][ll])
                kernels.append(kernel)
        else:
            sys.exit(' ---> Invalid number of paramters provided ('+str(Nparam)+')')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    else:
        sys.exit(' ---> type="'+type+'" is not valid in call to mock_cube_sources.gen_smoothed_cube() ')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if verbose: print ' - Applying convolution kernel ('+type+') to each wavelength layer '
    cube_psfed = tu.perform_2Dconvolution(cube,kernels,use_fftconvolution=use_fftconvolution,verbose=True)

    return cube_psfed
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def perform_2Dconvolution(cube,kernels,use_fftconvolution=False,verbose=True):
    """
    Perform 2D convolution in data cube layer by layer

    --- INPUT ---
    cube                 Data cube to convolve
    kernels              List of (astropy) kernels to apply on each (z/wavelengt)layer of the cube
    use_fftconvolution   To convolve in FFT space set this keyword to True

    --- EXAMPLE OF USE ---
    # see tdose_utilities.gen_psfed_cube()

    """
    csh = cube.shape
    cube_convolved = np.zeros(csh)

    for zz in xrange(csh[0]): # looping over wavelength layers of cube
        layer = cube[zz,:,:]
        if use_fftconvolution:
            layer_convolved = ac.convolve_fft(layer, kernels[zz], boundary='fill')
        else:
            layer_convolved = ac.convolve(layer, kernels[zz], boundary='fill')

        cube_convolved[zz,:,:] = layer_convolved

    return cube_convolved

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_2Dgauss(size,cov,scale,verbose=True,show2Dgauss=False):
    """
    Generating a 2D gaussian with specified parameters

    --- INPUT ---
    size    The dimensions of the array to return. Expects [y-size,x-size].
            The 2D gauss will be positioned in the center of a (+/-x-size/2., +/-y-size/2) sized array
    cov     Covariance matrix of gaussian, i.e., variances and rotation
            Can be build with cov = build_2D_cov_matrix(stdx,stdy,angle)
    scale   Scaling the 2D gaussian. By default scale = 1 returns normalized 2D Gaussian.
            I.e.,  np.trapz(np.trapz(gauss2D,axis=0),axis=0) = 1

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    covmatrix   = tu.build_2D_cov_matrix(4,1,5)
    gauss2Dimg  = tu.gen_2Dgauss([20,40],covmatrix,5,show2Dgauss=True)

    sigmax          = 3.2
    sigmay          = 1.5
    covmatrix       = tu.build_2D_cov_matrix(sigmax,sigmay,0)
    scale           = 1 # returns normalized gaussian
    Nsigwidth       = 15
    gauss2DimgNorm  = tu.gen_2Dgauss([sigmay*Nsigwidth,sigmax*Nsigwidth],covmatrix,scale,show2Dgauss=True)

    """
    if verbose: print ' - Generating multivariate_normal object for generating 2D gauss'
    mvn = multivariate_normal([0, 0], cov)

    if verbose: print ' - Setting up grid to populate with 2D gauss PDF'
    x, y = np.mgrid[-np.ceil(size[0]/2.):np.floor(size[0]/2.):1.0, -np.ceil(size[1]/2.):np.floor(size[1]/2.):1.0]
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
def shift_2Dprofile(profile,position,padvalue=0.0,showprofiles=False):
    """
    Shift 2D profile to given position in array by rolling it in x and y.
    Can move by sub-pixel amount using interpolation

    --- INPUT ---
    position      position to move center of image (profile) to:  [ypos,xpos]
                  NB! assumes position value starts from 0, i.e., if providing pixel values subtract 1.

    --- EXAMPLE OF USE ---

    """
    profile_dim = profile.shape

    yshift = position[0]-profile_dim[0]/2.
    xshift = position[1]-profile_dim[1]/2.
    profile_shifted = scipy.ndimage.interpolation.shift(profile, [yshift,xshift], output=None, order=3,
                                                        mode='constant', cval=0.0, prefilter=True)

    #profile_shifted = np.roll(np.roll(profile,yroll,axis=0),xroll,axis=1)

    if showprofiles:
        vmaxval = np.max(profile_shifted)
        plt.imshow(profile_shifted,interpolation='none',vmin=-vmaxval, vmax=vmaxval)
        plt.title('Positioned Source')
        plt.show()

    return profile_shifted
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def roll_2Dprofile(profile,position,padvalue=0.0,showprofiles=False):
    """
    Move 2D profile to given position in array by rolling it in x and y.

    --- INPUT ---
    position      position to move center of image (profile) to:  [ypos,xpos]
                  NB! assumes position value starts from 0, i.e., if providing pixel values subtract 1.

    --- EXAMPLE OF USE ---
    tu.roll_2Dprofile(gauss2D,)

    """
    profile_dim = profile.shape

    yroll = np.int(np.round(position[0]-profile_dim[0]/2.))
    xroll = np.int(np.round(position[1]-profile_dim[1]/2.))
    profile_shifted = np.roll(np.roll(profile,yroll,axis=0),xroll,axis=1)

    if showprofiles:
        vmaxval = np.max(profile_shifted)
        plt.imshow(profile_shifted,interpolation='none',vmin=-vmaxval, vmax=vmaxval)
        plt.title('Positioned Source')
        plt.show()

    if yroll < 0:
        profile_shifted[yroll:,:] = padvalue
    else:
        profile_shifted[:yroll,:] = padvalue

    if xroll < 0:
        profile_shifted[:,xroll:] = padvalue
    else:
        profile_shifted[:,:xroll] = padvalue

    if showprofiles:
        plt.imshow(profile_shifted,interpolation='none',vmin=-vmaxval, vmax=vmaxval)
        plt.title('Positioned Source with 0s inserted')
        plt.show()

    return profile_shifted
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def get_now_string(withseconds=False):
    """
    Retruning a string containing a formated version of the current data and time
    """
    if withseconds:
        nowstr  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        nowstr  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return nowstr
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_gridcomponents(imgsize):
    """
    Generate grid compoents, i.e. x and y indecese for a given image size
    """
    x = np.linspace(0, imgsize[1]-1, imgsize[1])
    y = np.linspace(0, imgsize[0]-1, imgsize[0])
    x,y = np.meshgrid(x, y)
    return x,y
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def analytic_convolution_gaussian(mu1,covar1,mu2,covar2):
    """
    The analytic vconvolution of two Gaussians is simply the sum of the two mean vectors
    and the two convariance matrixes

    """
    muconv    = mu1+mu2
    covarconv = covar1+covar2
    return muconv, covarconv

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def maxlikelihood_multivariateguass(datapoints,mean,covar):
    """
    Return the mean vector and co-variance matrix for the analytic maximum likelihood estimate of
    a multivariate gaussian distribution given a set of datapoints from the distribution
    See https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf

    (this does not match the scale, i.e., the amplitude of the distribution. For this a Chi2 estimate is needed)

    --- INPUT ---


    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    std       = np.array([3,1])
    covmatrix = tu.build_2D_cov_matrix(mean[1],mean[0],35)
    mean      = np.array([120,100])
    dataimg   = pyfits.open('/Users/kschmidt/work/TDOSE/mock_cube_sourcecat161213_oneobj.fits')[0].data[0,:,:]

    MLmean, MLcovar = maxlikelihood_multivariateguass(dataimg,mean,covmatrix)

    """
    datapoints = data.ravel()
    Npix       = len(datapoints)

    MLmean  = 1.0/Npix * np.sum( datapoints )
    MLcovar = 1.0/Npix * np.sum( (datapoints-mean) * np.transpose(datapoints-mean) )

    return MLmean, MLcovar
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def convert_paramarray(paramarray,hdr,hdr_new,verbose=True):
    """
    Function to convert the pixel-based paramter array from one wcs frame to another

    --- INFO ---
    paramarray    Parameter array (e.g., loaded with build_paramarray)
    hdr           Header (wcs) information the parameter array referes to
    hdr_new       The header (wcs) information to us for transforming parameters to new reference frame

    """
    paramconv = np.zeros(paramarray.shape)

    wcs_in  = wcs.WCS(tu.strip_header(hdr.copy()))
    wcs_out = wcs.WCS(tu.strip_header(hdr_new.copy()))

    if wcs_out.to_header()['WCSAXES'] == 3:
        wcs_out = tu.WCS3DtoWCS2D(wcs_out)

    Nobj      = len(paramarray)/6
    scale_in  = wcs.utils.proj_plane_pixel_scales(wcs_in)*3600.0   # pix scale in arcsec
    scale_out = wcs.utils.proj_plane_pixel_scales(wcs_out)*3600.0  # pix scale in arcsec

    for oo in xrange(Nobj):
        ypix      = paramarray[oo*6+0]-1.
        xpix      = paramarray[oo*6+1]-1.
        skycoord  = wcs.utils.pixel_to_skycoord(xpix,ypix,wcs_in)
        pixcoord  = wcs.utils.skycoord_to_pixel(skycoord,wcs_out)# + np.array([1,1])
        paramconv[oo*6+0] = pixcoord[1]+1
        paramconv[oo*6+1] = pixcoord[0]+1
        paramconv[oo*6+2] = paramarray[oo*6+2]
        paramconv[oo*6+3] = paramarray[oo*6+3]*scale_in[0]/scale_out[0]
        paramconv[oo*6+4] = paramarray[oo*6+4]*scale_in[1]/scale_out[1]
        paramconv[oo*6+5] = paramarray[oo*6+5]

    return paramconv
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def build_paramarray(fitstable,verbose=True):
    """
    Build parameter array (list) expected by tdose_model_cube.gen_fullmodel()
    based on output parameter fits file from tdose_model_FoV.gen_fullmodel()

    --- INPUT ---
    fitstable       fits table containing the fitted and intial source parameters
                    outputted by tdose_model_FoV.gen_fullmodel()

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    path = '/Users/kschmidt/work/TDOSE/'
    file = 'mock_cube_sourcecat161213_all_tdose_mock_cube_NOISEgauss_v170207_modelimage_nosigma_objparam.fits'
    paramarray = tu.build_paramarray(path+file,verbose=True)
    """
    tabdat     = pyfits.open(fitstable)[1].data
    Nobj       = len(tabdat['obj'])
    paramarray = np.zeros([Nobj*6])

    for oo in xrange(Nobj):
        paramarray[oo*6+0] = tabdat['ypos'][oo]
        paramarray[oo*6+1] = tabdat['xpos'][oo]
        paramarray[oo*6+2] = tabdat['fluxscale'][oo]
        paramarray[oo*6+3] = tabdat['ysigma'][oo]
        paramarray[oo*6+4] = tabdat['xsigma'][oo]
        paramarray[oo*6+5] = tabdat['angle'][oo]

    return paramarray
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def WCS3DtoWCS2D(wcs3d,verbose=True):
    """
    Removing the wavelength component of a WCS object, i.e., turning converting
    the WCS from 3D (lambda,ra,dec) to 2D (ra,dec)
    """
    hdr3D = wcs3d.to_header()
    for key in hdr3D.keys():
        if '3' in key:
            del hdr3D[key]

    hdr3D['WCSAXES'] = 2
    wcs2d = wcs.WCS(hdr3D)

    return wcs2d
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_subcube(cubefile,ra,dec,cutoutsize,outname,cubeext=['DATA','STAT'],
                    clobber=False,imgfiles=None,imgexts=None,verbose=True):
    """
    Function for cropping/extracting sub data cube (and potentially corresponding image)

    --- INPUT ---
    cubefile       Data cube to extract sub-cube from
    ra             The right ascension of center of sub-cube
    dec            The declination of the center of the sub-cube
    cutoutsize     RA and Dec size of cutout (in arc sec).
    outname        Name of file to save extracted sub-cube to
    clobber        If true existing fits image will be overwritten
    imgfiles       List of file names to extract sub-images for corresponding to sub-cube's spacial extent
                   Will save images to same directory as sub-cub outname
    verbose        Toggle verbosity

    --- EXAMPLE OF USE ---
    cubefile    = '/Users/kschmidt/work/TDOSE/musecubetestdata/candels-cdfs-15/DATACUBE_candels-cdfs-15_v1.0.fits'
    imgfile     = '/Users/kschmidt/work/images_MAST/hlsp_candels_hst_wfc3_gs-tot_f125w_v1.0_drz.fits'
    ra          = 53.12437322
    dec         = -27.85161087
    cutoutsize  = [10,7]
    outname     = '/Users/kschmidt/work/TDOSE/musecubetestdata/DATACUBE_candels-cdfs-15_v1p0_cutout_MUSEWide11503085_'+str(cutoutsize[0])+'x'+str(cutoutsize[1])+'arcsec.fits'
    cutouts     = tu.extract_subcube(cubefile,ra,dec,cutoutsize,outname,cubeext=['DATA','STAT'],clobber=True,imgfiles=[imgfile],imgexts=[0])

    """
    if verbose: print ' - Extracting sub data cube from :\n   '+cubefile
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if os.path.isfile(outname) & (clobber == False):
        sys.exit(outname+' already exists and clobber=False ')
    skyc      = SkyCoord(ra, dec, frame='icrs', unit=(units.deg,units.deg))
    size      = units.Quantity((  cutoutsize[1], cutoutsize[0]), units.arcsec)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Ncubes    = len(cubeext)
    hdrs_all  = []
    for cc, cx in enumerate(cubeext):
        if verbose: print '\n - Cutting out wavelength layes of cube in extension '+cx
        cubedata = pyfits.open(cubefile)[cx].data
        cubehdr  = pyfits.open(cubefile)[cx].header
        Nlayers  = cubedata.shape[0]

        if verbose: print ' - Removing comments and history as well as "section title entries" ' \
                          'from fits header as "newline" is non-ascii character'
        striphdr   = tu.strip_header(cubehdr.copy())
        cubewcs    = wcs.WCS(striphdr)
        cubewcs_2D = tu.WCS3DtoWCS2D(cubewcs.copy())

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print ' - Extracting sub-image in eachlayer'
        for ll in xrange(Nlayers):
            if verbose:
                infostr = '   cutting out from layer '+str("%6.f" % (ll+1))+' / '+str("%6.f" % Nlayers)
                sys.stdout.write("%s\r" % infostr)
                sys.stdout.flush()

            cutout_layer  = Cutout2D(cubedata[ll,:,:], skyc, size, wcs=cubewcs_2D)

            if ll == 0:
                cutout_cube = np.zeros([Nlayers,cutout_layer.data.shape[0],cutout_layer.data.shape[1]])
                for key in cutout_layer.wcs.to_header().keys():
                    striphdr[key] = cutout_layer.wcs.to_header()[key]
                hdrs_all.append(striphdr)

            cutout_cube[ll,:,:] = cutout_layer.data

        if cc == 0:
            cutouts = [cutout_cube]
        else:
            cutouts.append(cutout_cube)

        if verbose: print '\n   done'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Saving sub-cubes to '+outname
    hducube = pyfits.PrimaryHDU()  # creating default fits header
    hdulist = [hducube]

    for cc, cx in enumerate(cubeext):
        if verbose: print '   Add clean version of cube to extension               '+cx
        hducutout        = pyfits.ImageHDU(cutouts[cc])
        for key in hdrs_all[cc]:
            if not key in hducutout.header.keys():
                hducutout.header.append((key,hdrs_all[cc][key],hdrs_all[cc][key]),end=True)

        hducutout.header.append(('EXTNAME ',cx            ,''),end=True)
        hdulist.append(hducutout)

    hdulist = pyfits.HDUList(hdulist)       # turn header into to hdulist
    hdulist.writeto(outname,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if imgfiles is not None:
        Nimg = len(imgfiles)
        if verbose: print ' - Extracting images corresponding to cube from '+str(Nimg)+' images in imgfiles provided:\n'
        if imgexts is None:
            imgexts = [0]*Nimg

        for ii, imgfile in enumerate(imgfiles):
            imgname = imgfile.split('/')[-1]
            outname = outname.replace('.fits','_CUTOUT'+str(cutoutsize[0])+'x'+str(cutoutsize[1])+'arcsec_From_'+imgname)
            cutout  = tu.extract_subimage(imgfile,ra,dec,cutoutsize,outname=outname,
                                          imgext=imgexts[ii],clobber=clobber,verbose=verbose)

    return cutouts
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_subimage(imgfile,ra,dec,cutoutsize,outname=None,clobber=False,imgext=0,verbose=True):
    """
    Crop a fits image and save the extract subimage to file

    --- INPUT ---
    imgfile        File name to extract sub-image from
    ra             The right ascension of center of sub-image
    dec            The declination of the center of the sub-image
    cutoutsize     RA and Dec size of cutout (in arc sec).
    outname        Name of file to save extracted sub-image to; if None the cutout will just be returned
    clobber        If true existing fits image will be overwritten
    imgext         Image fits extension
    verbose        Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    imgfile     = '/Users/kschmidt/work/images_MAST/hlsp_candels_hst_wfc3_gs-tot_f125w_v1.0_drz.fits'
    ra          = 53.12437322
    dec         = -27.85161087
    cutoutsize  = [15,30]
    outname     = '/Users/kschmidt/work/TDOSE/hlsp_candels_hst_wfc3_gs-tot_f125w_v1.0_drz_MUSEWide11503085_'+str(cutoutsize[0])+'X'+str(cutoutsize[1])+'arcseccut.fits'
    cutout      = tu.extract_subimage(imgfile,ra,dec,cutoutsize,outname=outname,clobber=False)

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Will extract '+str(cutoutsize[0])+'X'+str(cutoutsize[1])+\
                      ' arcsec subimage centered on ra,dec='+str(ra)+','+str(dec)+' from:\n   '+imgfile

    imgdata  = pyfits.open(imgfile)[imgext].data
    imghdr   = pyfits.open(imgfile)[imgext].header

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Removing comments and history as well as "section title entries" ' \
                      'from fits header as "newline" is non-ascii character'
    striphdr = tu.strip_header(imghdr.copy())
    imgwcs   = wcs.WCS(striphdr)
    skyc     = SkyCoord(ra, dec, frame='icrs', unit=(units.deg,units.deg))
    size     = units.Quantity((  cutoutsize[1], cutoutsize[0]), units.arcsec)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Cutting out data and updating WCS info'
    cutout  = Cutout2D(imgdata, skyc, size, wcs=imgwcs)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if outname is not None:
        cuthdr  = cutout.wcs.to_header()
        if verbose: print ' - Update fits header and save file to \n   '+outname
        imghdrkeys = imghdr.keys()
        for key in cuthdr.keys():
            if key in imghdrkeys:
                imghdr[key] = cuthdr[key]

        hdulist = pyfits.PrimaryHDU(data=cutout.data,header=imghdr)
        hdulist.writeto(outname,clobber=clobber)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return cutout
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def strip_header(header,verbose=True):
    """
    Removing all COMMENT, "TITLE" and HISTORY parameters in a fits header to avoid non-ascii characters
    """
    del header['COMMENT']
    del header['HISTORY']
    for key in header.keys():
        if key == '':
            del header[key]
    return header
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_ds9region(fitstable,outputfile,wcsinfo,color='red',width=2,Nsigma=2,textlist=None,fontsize=12,
                    clobber=False,verbose=True):
    """
    Generate a basic DS9 region file based on a model parameters file.

    """
    if verbose: print ' - Generating DS9 region file from model paramter file'
    paramarray = tu.build_paramarray(fitstable,verbose=True)
    Nobj       = len(paramarray)/6
    scale      = wcs.utils.proj_plane_pixel_scales(wcsinfo)*3600.0   # pix scale in arcsec

    if not clobber:
        if os.path.isfile(outputfile):
            sys.exit(' ---> File already exists and clobber = False')
    fout = open(outputfile,'w')
    fout.write("# Region file format: DS9 version 4.1 \nfk5\n")

    if textlist is None:
        textstrings = pyfits.open(fitstable)[1].data['obj'].astype(int).astype(str)
    else:
        textstrings = pyfits.open(fitstable)[1].data['obj'].astype(int).astype(str)
        for tt, obj in enumerate(textstrings):
            textstrings[tt] = obj+': '+textlist[tt]

    if verbose: print ' - Converting to wcs coordinates'
    for oo in xrange(Nobj):
        ypix      = paramarray[oo*6+0]
        xpix      = paramarray[oo*6+1]
        skycoord  = wcs.utils.pixel_to_skycoord(xpix,ypix,wcsinfo)
        dec       = skycoord.dec.value
        ra        = skycoord.ra.value
        fluxscale = paramarray[oo*6+2]
        sigmay    = paramarray[oo*6+3]*scale[0]/3600.0
        sigmax    = (paramarray[oo*6+4]*scale[1]*np.cos(np.deg2rad(dec)))/3600.0
        angle     = paramarray[oo*6+5]
        #if oo == 4: pdb.set_trace()
        string = 'ellipse('+str(ra)+','+str(dec)+','+str(Nsigma*sigmax)+','+str(Nsigma*sigmay)+','+str(angle)+') '

        string = string+' # color='+color+' width='+str(width)+\
                 ' font="times '+str(fontsize)+' bold roman" text={'+textstrings[oo]+'}'

        fout.write(string+' \n')

    fout.close()
    if verbose: print ' - Saved region file to '+outputfile
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_sourcecat_from_SExtractorfile(sextractorfile,outname='./tdose_sourcecat.txt',clobber=False,
                                      idcol=0,racol=2,deccol=3,fluxcol=22,fluxfactor=100.,verbose=True):
    """
    Generate source catalog for modeling image with tdose_model_FoV.gen_fullmodel()

    """
    if verbose: print ' - Generating TDOSE source catalog for FoV modeling'
    if sextractorfile.endswith('.fits'):
        sexdat  = pyfits.open(sextractorfile)[1].data
        ids     = sexdat[idcol]
        ras     = sexdat[racol]
        decs    = sexdat[deccol]
        fluxes  = sexdat[fluxcol]*fluxfactor
    else:
        sexdat  = np.genfromtxt(sextractorfile,names=None,dtype=None,comments='#')
        ids     = sexdat['f'+str(idcol)]
        ras     = sexdat['f'+str(racol)]
        decs    = sexdat['f'+str(deccol)]
        fluxes  = sexdat['f'+str(fluxcol)]*fluxfactor

    if (clobber == False) & os.path.isfile(outname):
        if verbose: print ' - WARNING: Output ('+outname+') already exists and clobber=False, hence returning None'
        return None
    else:
        if verbose: print ' - Will save source catalog to '+outname+' (overwriting any existing file)'
        fout = open(outname,'w')
        fout.write('# TDOSE Source catalog generated with tdose_utilities.gen_sourcecat_from_SExtractorfile() from:\n')
        fout.write('# '+sextractorfile+'\n')
        fout.write('# id xpos  ypos  fluxscale \n')

        for ii, id in enumerate(ids):
            fout.write(str(ids[ii])+' '+str(ras[ii])+' '+str(decs[ii])+' '+str(fluxes[ii])+'  \n')

        fout.close()
        return outname
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_paramlist_from_SExtractorfile(sextractorfile,pixscale=0.06,imgheader=None,clobber=False,
                                      idcol='ID',racol='RA',deccol='DEC',aimg='A_IMAGE',bimg='B_IMAGE',
                                      angle='THETA_IMAGE',fluxscale='FLUX_ISO_F814W',fluxfactor=100.,Nsigma=3,
                                      saveDS9region=True,ds9color='red',ds9width=2,ds9fontsize=12,
                                      savefitsimage=False,verbose=True):
    """
    Generate source catalog for modeling image with tdose_model_FoV.gen_fullmodel()

    --- INPUT ---

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    sexfile   = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/catalog_photometry_candels-cdfs-02.fits'
    imgheader = pyfits.open('/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/acs_814w_candels-cdfs-02_cut_v1.0.fits')[0].header
    paramlist = tu.gen_paramlist_from_SExtractorfile(sexfile,imgheader=imgheader,Nsigma=8,savefitsimage=True)

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Loading SExtractor catalog'
    try:
        sourcedat = pyfits.open(sextractorfile)[1].data
    except:
        sys.exit(' ---> Problems loading fits catalog')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if imgheader is None:
        if verbose: print ' - Image header not provided; assuming ra and dec col are in pixel units'
    else:
        if verbose: print ' - Image header provided; converting ra and dec values using wcs info from header'
        striphdr = tu.strip_header(imgheader.copy())
        wcs_in   = wcs.WCS(striphdr)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Nobjects   = len(sourcedat)
    if verbose: print ' - Assembling paramter list for '+str(Nobjects)+' sources found in catalog'
    paramlist = []
    for oo in xrange(Nobjects):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if imgheader is None:
            xpos       = sourcedat[racol][oo]
            ypos       = sourcedat[deccol][oo]
        else:
            skycoord   = SkyCoord(sourcedat[racol][oo], sourcedat[deccol][oo], frame='icrs', unit='deg')
            pixcoord   = wcs.utils.skycoord_to_pixel(skycoord,wcs_in)
            xpos, ypos = pixcoord[0], pixcoord[1]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        xpos = xpos
        ypos = ypos
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if fluxscale is not None:
            fs = sourcedat[fluxscale][oo]*fluxfactor
        else:
            fs = 1.0
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        sigy  = sourcedat[bimg][oo]*Nsigma
        sigx  = sourcedat[aimg][oo]*Nsigma
        ang   = sourcedat[angle][oo]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        objlist    = [ypos,     xpos,     fs,       sigy,  sigx,  ang]
        paramlist  = paramlist + objlist
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    paramlist_arr = np.asarray(paramlist)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if saveDS9region:
        if verbose: print ' - Generating DS9 region file with object parameters'
        ds9region_file = sextractorfile.replace('.fits','_tdose.reg')
        if os.path.isfile(ds9region_file) & (clobber == False):
            if verbose: print ' - ds9 region file '+ds9region_file+' already exists and clobber=False so skipping'
        else:
            fout = open(ds9region_file,'w')
            fout.write("# Region file format: DS9 version 4.1 \nimage\n")

            for oo in xrange(Nobjects):
                objparam = np.resize(paramlist_arr,(Nobjects,6))[oo]

                string = 'ellipse('+str(objparam[1])+','+str(objparam[0])+','+str(objparam[4])+','+\
                         str(objparam[3])+','+str(objparam[5])+') '

                string = string+' # color='+ds9color+' width='+str(ds9width)+\
                         ' font="times '+str(ds9fontsize)+' bold roman" text={'+str(sourcedat[idcol][oo])+'}'

                fout.write(string+' \n')
            fout.close()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if savefitsimage:
        if verbose: print ' - Generating image with object models from object parameters'
        fitsimage = sextractorfile.replace('.fits','_tdose_modelimage.fits')
        if imgheader is None:
            if verbose: print ' - No image header provided for fits file ' \
                              '(to get wcs and image model image dimensions) so skipping'
        else:
            imgsize   = np.array([striphdr['NAXIS2'],striphdr['NAXIS1']])
            tmf.save_modelimage(fitsimage,paramlist_arr,imgsize,param_init=False,clobber=clobber,
                                outputhdr=imgheader,verbose=verbose,verbosemodel=verbose)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return paramlist_arr
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def plot_matrix_array():
    return None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def build_fits_wcs():
    return None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =