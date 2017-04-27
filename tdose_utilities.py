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
import subprocess
import glob
import shutil
import scipy.ndimage
import tdose_utilities as tu
import tdose_model_FoV as tmf
import astropy.convolution as ac # convolve, convolve_fft, Moffat2DKernel, Gaussian2DKernel
from scipy.stats import multivariate_normal
import matplotlib.pylab as plt
import pdb
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def load_setup(setupfile='./tdose_setup_template.txt',verbose=True):
    """
    Return dictionary with the setups found in 'setupfile'

    --- INPUT ---
    setupfile    The name of the txt file containing the TDOSE setup to load
                 A template for this setup file can be generated with
                 tdose_load_setup.generate_setup_template()

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    setup = tu.load_setup(setupfile='./tdose_setup_template.txt')

    """
    if verbose: print ' --- tdose_utilities.load_setup() --- '
    #------------------------------------------------------------------------------------------------------
    if verbose: print ' - Loading setup for TDOSE in '+setupfile
    setup_arr = np.genfromtxt(setupfile,dtype=None,names=None)
    setup_dic = {}
    for ii in xrange(setup_arr.shape[0]):
        try:
            val = float(setup_arr[ii,1])
        except:
            val = str(setup_arr[ii,1])

        # - - - treatment of individual paramters - - -
        if ('extension' in setup_arr[ii,0]) & (type(val) == float): val = int(val)
        if setup_arr[ii,1].lower() == 'none':  val = None
        if setup_arr[ii,1].lower() == 'true':  val = True
        if setup_arr[ii,1].lower() == 'false': val = False

        dirs = ['sources_to_extract','model_cube_layers','cutout_sizes']
        if (setup_arr[ii,0] in dirs) & ('/' in setup_arr[ii,1]):
            val = setup_arr[ii,1]
            setup_dic[setup_arr[ii,0]] = val
            continue

        lists = ['modify_sources_list','model_cube_layers','sources_to_extract','plot_1Dspec_xrange','plot_1Dspec_yrange',
                 'plot_S2Nspec_xrange','plot_S2Nspec_yrange','cutout_sizes']
        if (setup_arr[ii,0] in lists) & (setup_arr[ii,1] != 'all'):
            val = [float(vv) for vv in val.split('[')[-1].split(']')[0].split(',')]
            setup_dic[setup_arr[ii,0]] = val
            continue

        if ('psf_sigma' in setup_arr[ii,0]) & (type(val) == str):
            if  '/' in val:
                sigmasplit = val.split('/')
                if len(sigmasplit) != 2:
                    pass
                else:
                    val = float(sigmasplit[0]) / float(sigmasplit[1])
            setup_dic[setup_arr[ii,0]] = val
            continue

        setup_dic[setup_arr[ii,0]] = val
    if verbose: print ' - Returning dictionary containing setup parameters'
    return setup_dic
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def generate_setup_template(outputfile='./tdose_setup_template.txt',clobber=False,verbose=True):
    """
    Generate setup text file template

    --- INPUT ---
    outputfile    The name of the output which will contain the TDOSE setup template

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu

    filename = './tdose_setup_template_new.txt'
    tu.generate_setup_template(outputfile=filename,clobber=True)
    setup    = tu.load_setup(setupfile=filename)

    """
    if verbose: print ' --- tdose_utilities.generate_setup_template() --- '
    #------------------------------------------------------------------------------------------------------
    if os.path.isfile(outputfile) & (clobber == False):
        sys.exit(' ---> Outputfile already exists and clobber=False ')
    else:
        if verbose: print ' - Will store setup template in '+outputfile
        if os.path.isfile(outputfile) & (clobber == True):
            if verbose: print ' - Output already exists but clobber=True so overwriting it '

        setuptemplate = """
#-------------------------------------------------START OF TDOSE SETUP-------------------------------------------------
#
# Template for TDOSE (http://github.com/kasperschmidt/TDOSE) setup file
# Generated with tdose_utilities.generate_setup_template() on 2017-04-11 14:03
# The spectral extraction using this setup is run with tdose.perform_extraction()
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - DATA INPUT  - - - - - - - - - - - - - - - - - - - - - - - - - - -
data_cube              /Volumes/DATABCKUP2/MUSE-Wide/datacubes_dcbgc/DATACUBE_candels-cdfs-02_v1.0_dcbgc.fits                    # Path and name of fits file containing data cube to extract spectra from
cube_extension         DATA_DCBGC                         # Name or number of fits extension containing data cube

variance_cube          /Volumes/DATABCKUP2/MUSE-Wide/datacubes_dcbgc/DATACUBE_candels-cdfs-02_v1.0_dcbgc.fits                   # Path and name of fits file containing variance cube to use for extraction
variance_extension     STAT                              # Name or number of fits extension containing noise cube

ref_image              /Volumes/DATABCKUP2/MUSE-Wide/hst_cutouts/acs_814w_candels-cdfs-02_cut_v1.0.fits              # Path and name of fits file containing image to use as reference when creating source model
img_extension          0                                  # Name or number of fits extension containing reference image

source_catalog         /Volumes/DATABCKUP2/TDOSEextractions/tdose_sourcecats/catalog_photometry_candels-cdfs-02_tdose_sourcecat.fits               # Path and name of source catalog containing sources to extract spectra for
sourcecat_IDcol        id                                 # Column containing source IDs in source_catalog
sourcecat_xposcol      x_image                            # Column containing x pixel position in source_catalog
sourcecat_yposcol      y_image                            # Column containing y pixel position in source_catalog
sourcecat_racol        ra                                 # Column containing ra  position in source_catalog (used to position cutouts if model_cutouts = True)
sourcecat_deccol       dec                                # Column containing dec position in source_catalog (used to position cutouts if model_cutouts = True)
sourcecat_fluxcol      fluxscale                          # Column containing dec position in source_catalog (used to position cutouts if model_cutouts = True)
sourcecat_parentIDcol  None                               # Column containing parent source IDs grouping source IDs into objects. Set to None to used id column
                                                          # corresponding to assigning each source to a single object
                                                          # if not None the parentid is used to group source models when storing 1D spectra. All models keep sources separate.
# - - - - - - - - - - - - - - - - - - - - - - - - OUTPUT DIRECTORIES  - - - - - - - - - - - - - - - - - - - - - - - - -

models_directory       /Volumes/DATABCKUP2/TDOSEextractions/tdose_models/                  # Directory to store the modeling output from TDOSE in
cutout_directory       /Volumes/DATABCKUP2/TDOSEextractions/tdose_cutouts/                 # Directory to store image and cube cutouts in if model_cutouts=True
spec1D_directory       /Volumes/DATABCKUP2/TDOSEextractions/tdose_spectra/                 # Output directory to store spectra in.

# - - - - - - - - - - - - - - - - - - - - - - - - SOURCE MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
model_image_ext        tdose_modelimage                   # Name extension of fits file containing reference image model. To ignored use None
model_param_reg        tdose_modelimage_ds9               # Name extension of DS9 region file for reference image model. To ignored use None
model_image_cube_ext   tdose_modelimage_cubeWCS           # Name extension of fits file containing model image after conversion to cube WCS. To ignored use None.

source_model           gauss                              # The source model to use for sources: [gauss, galfit, mog (not enabled)]

# - - - - - - - - - - - - - - - - - - - - - - - - GAUSS MODEL SETUP - - - - - - - - - - - - - - - - - - - - - - - - - -
gauss_guess            /Volumes/DATABCKUP2/MUSE-Wide/catalogs_photometry/catalog_photometry_candels-cdfs-02.fits                               # To base initial guess of gaussian parameters on a SExtractor output provide SExtractor output fits file here
                                                          # If gauss_initguess=None the positions and flux scale provided in source_catalog will be used.
gauss_guess_idcol      ID                                 # Column of IDs in gauss_guess SExtractor file
gauss_guess_racol      RA                                 # Column of RAs in gauss_guess SExtractor file
gauss_guess_deccol     DEC                                # Column of Decs in gauss_guess SExtractor file
gauss_guess_aimg       A_IMAGE                            # Column of major axis in gauss_guess SExtractor file
gauss_guess_bimg       B_IMAGE                            # Column of minor axis in gauss_guess SExtractor file
gauss_guess_angle      THETA_IMAGE                        # Column of angle in gauss_guess SExtractor file
gauss_guess_fluxscale  ACS_F814W_FLUX                     # Column of flux in gauss_guess SExtractor file to us for scaling
gauss_guess_fluxfactor 3                                  # Factor to apply to flux scale in initial Gauss parameter guess
gauss_guess_Nsigma     1                                  # Number of sigmas to include in initial Gauss parameter guess

# - - - - - - - - - - - - - - - - - - - - - - - - GALFIT MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
galfit_result          None                               # If source_model = galfit provide the path and name of fits file containing galfit results
galfit_model_extension 2                                  # Fits extension containing galfit model with model parameters of each source in header

# - - - - - - - - - - - - - - - - - - - - - - - - - - CUTOUT SETUP  - - - - - - - - - - - - - - - - - - - - - - - - - -
model_cutouts          True                               # Perform modeling and spectral extraction on small cutouts of the cube and images to reduce run-time
cutout_sizes           /Users/kschmidt/work/TDOSE/tdose_setup_cutoutsizes.txt                             # Size of cutouts [ra,dec] in arcsec around each source to model.
                                                          # To use source-specific cutouts provide ascii file containing ID xsize[arcsec] and ysize[arcsec].

# - - - - - - - - - - - - - - - - - - - - - - - - - PSF MODEL SETUP - - - - - - - - - - - - - - - - - - - - - - - - - -
psf_type               gauss                              # Select PSF model to build. Choices are:
                                                          #   gauss      Model the PSF as a symmetric Gaussian with sigma = FWHM/2.35482
psf_FWHM_evolve        linear                             # Evolution of the FWHM from blue to red end of data cube. Choices are:
                                                          #   linear     FWHM wavelength dependence described as FWHM(lambda) = p0[''] + p1[''/A] * (lambda - 7000A)
psf_FWHMp0             0.940                              # p0 parameter to use when determining wavelength dependence of PSF
psf_FWHMp1             -3.182e-5                          # p1 parameter to use when determining wavelength dependence of PSF
psf_savecube           True                               # To save fits file containing the PSF cube set psf_savecube = True

# - - - - - - - - - - - - - - - - - - - - - - - - - CUBE MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
model_cube_layers      /Users/kschmidt/work/TDOSE/tdose_setup_layers.txt                  # Layers of data cube to model [both end layers included]. If 'all' the full cube will be modeled.
                                                          # To model source-specific layers provide ascii file containing ID layerlow and layerhigh.
                                                          # If layerlow=all and layerhigh=all all layers will be modeled for particular source

model_cube_optimizer   matrix                             # The optimizer to use when matching flux levels in cube layers: [matrix,curvet,lstsq]

model_cube_ext         tdose_modelcube                    # Name extension of fits file containing model data cube.
residual_cube_ext      tdose_modelcube_residual           # Name extension of fits file containing residual between model data cube and data. To ignored use None.
source_model_cube      tdose_source_modelcube             # Name extension of fits file containing source model cube (used to modify data cube).

# - - - - - - - - - - - - - - - - - - - - - - - - SPECTRAL EXTRACTION - - - - - - - - - - - - - - - - - - - - - - - - -
sources_to_extract     /Users/kschmidt/work/TDOSE/tdose_setup_objects.txt # [8685,10195,29743]                 # Sources in source_catalog to extract 1D spectra for.
                                                          # If sourcecat_parentIDcol os not None all associated spectra are included in stored object spectra
                                                          # If set to 'all', 1D spectra for all sources in source_catalog is produced (without grouping according to parents).
                                                          # For long list of objects provide ascii file containing containing ids (here parent grouping will be performed)
spec1D_name            tdose_spectrum                     # Name extension to use for extracted 1D spectra

# - - - - - - - - - - - - - - - - - - - - - - - - - - - PLOTTING  - - - - - - - - - - - - - - - - - - - - - - - - - - -
plot_generate          True                               # Indicate whether to generate plots or not
plot_1Dspec_ext        fluxplot                           # Name extension of pdf file containing plot of 1D spectrum
plot_1Dspec_xrange     [4800,9300]                        # Range of x-axes (wavelength) for plot of 1D spectra
plot_1Dspec_yrange     [-100,1500]                        # Range of y-axes (flux) for plot of 1D spectra
plot_1Dspec_shownoise  True                               # Indicate whether to show the noise envelope in plot or not

plot_S2Nspec_ext       S2Nplot                            # Name extension of pdf file containing plot of S/N spectrum
plot_S2Nspec_xrange    [4800,9300]                        # Range of x-axes (wavelength) for plot of S2N spectra
plot_S2Nspec_yrange    [-1,60]                            # Range of y-axes (S2N) for plot of S2N spectra
#--------------------------------------------------END OF TDOSE SETUP--------------------------------------------------

""" % (tu.get_now_string())
        fout = open(outputfile,'w')
        fout.write(setuptemplate)
        fout.close()

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def generate_setup_template_modify(outputfile='./tdose_setup_template_modify.txt',clobber=False,verbose=True):
    """
    Generate setup text file template for modifying data cubes

    --- INPUT ---
    outputfile    The name of the output which will contain the TDOSE setup template

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu

    filename = './tdose_setup_template_modify_new.txt'
    tu.generate_setup_template_modify(outputfile=filename,clobber=True)
    setup    = tu.load_setup(setupfile=filename)

    """
    if verbose: print ' --- tdose_utilities.generate_setup_template_modify() --- '
    #------------------------------------------------------------------------------------------------------
    if os.path.isfile(outputfile) & (clobber == False):
        sys.exit(' ---> Outputfile already exists and clobber=False ')
    else:
        if verbose: print ' - Will store setup template in '+outputfile
        if os.path.isfile(outputfile) & (clobber == True):
            if verbose: print ' - Output already exists but clobber=True so overwriting it '

        setuptemplate = """
#---------------------------------------------START OF TDOSE MODIFY SETUP---------------------------------------------
#
# Template for TDOSE (http://github.com/kasperschmidt/TDOSE) setup file for modifyinf data cubes
# Generated with tdose_utilities.generate_setup_template_modify() on %s
# Cube modifications are run independent of tdose.perform_extraction() with tdose.modify_cube()
#
# - - - - - - - - - - - - - - - - - - - - - - - - -  MODIFYING CUBE - - - - - - - - - - - - - - - - - - - - - - - - - -
data_cube              /Volumes/DATABCKUP2/TDOSEextractions/tdose_cutouts/DATACUBE_candels-cdfs-02_v1.0_dcbgc_id8685_cutout9p0x12p0arcsec.fits                    # Path and name of fits file containing data cube to modify
cube_extension         DATA_DCBGC                         # Name or number of fits extension containing data cube
source_model_cube      /Volumes/DATABCKUP2/TDOSEextractions/tdose_models/DATACUBE_candels-cdfs-02_v1.0_dcbgc_id8685_cutout9p0x12p0arcsec_tdose_source_modelcube.fits                    # Path and name of fits file containing source model cube
source_extension       DATA_DCBGC                         # Name or number of fits extension containing source model cube

modyified_cube         tdose_modified_datacube            # Name extension of file containing modified data cube.

                                                          # should be removed ( in "objects" will be removed.
modify_sources_list    [1,2,5]                            # List of IDs of sources to remove from data cube using source model cube.
                                                          # For long list of IDs provide path and name of file containing IDs (only)
sources_action         remove                             # Indicate how to modify the data cube. Chose between:
                                                          #    'remove'     Sources in modify_sources_list are removed from data cube (default)
                                                          #    'keep'       All sources except the sources in modify_sources_list are removed from data cube
#----------------------------------------------END OF TDOSE MODIFY SETUP----------------------------------------------

""" % (tu.get_now_string())
        fout = open(outputfile,'w')
        fout.write(setuptemplate)
        fout.close()

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
def gen_2Dgauss(size,cov,scale,method='scipy',show2Dgauss=False,verbose=True):
    """
    Generating a 2D gaussian with specified parameters

    --- INPUT ---
    size          The dimensions of the array to return. Expects [y-size,x-size].
                  The 2D gauss will be positioned in the center of a (+/-x-size/2., +/-y-size/2) sized array
    cov           Covariance matrix of gaussian, i.e., variances and rotation
                  Can be build with cov = build_2D_cov_matrix(stdx,stdy,angle)
    scale         Scaling the 2D gaussian. By default scale = 1 returns normalized 2D Gaussian.
                  I.e.,  np.trapz(np.trapz(gauss2D,axis=0),axis=0) = 1
    method        Method to use for generating 2D gaussian:
                   'scipy'    Using the class multivariate_normal from the scipy.stats library
                   'matrix'   Use direct matrix expression for PDF of 2D gaussian               (slow!)
    show2Dgauss   display image of generated 2D gaussian
    verbose       Toggler verbosity

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
    if verbose: print ' - Generating multivariate_normal object for generating 2D gauss using ',
    if method == 'scipy':
        if verbose: print ' scipy.stats.multivariate_normal.pdf() '
        mvn     = multivariate_normal([0, 0], cov)

        if verbose: print ' - Setting up grid to populate with 2D gauss PDF'
        x, y = np.mgrid[-np.ceil(size[0]/2.):np.floor(size[0]/2.):1.0, -np.ceil(size[1]/2.):np.floor(size[1]/2.):1.0]
        pos = np.zeros(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y

        gauss2D = mvn.pdf(pos)
    elif method == 'matrix':
        if verbose: print ' loop over matrix expression '
        gauss2D = np.zeros([np.int(np.ceil(size[0])),np.int(np.ceil(size[1]))])
        mean    = np.array([np.floor(size[0]/2.),np.floor(size[1]/2.)])
        norm    = 1/np.linalg.det(np.sqrt(cov))/2.0/np.pi
        for xpix in np.arange(size[1]):
            for ypix in np.arange(size[0]):
                coordMmean                   = np.array([int(ypix),int(xpix)]) - mean
                MTXexpr                      = np.dot(np.dot(np.transpose(coordMmean),np.linalg.inv(cov)),coordMmean)
                gauss2D[int(ypix),int(xpix)] = norm * np.exp(-0.5 * MTXexpr)

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
def build_paramarray(fitstable,returninit=False,verbose=True):
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
        if returninit:
            paramarray[oo*6+0] = tabdat['ypos_init'][oo]
            paramarray[oo*6+1] = tabdat['xpos_init'][oo]
            paramarray[oo*6+2] = tabdat['fluxscale_init'][oo]
            paramarray[oo*6+3] = tabdat['ysigma_init'][oo]
            paramarray[oo*6+4] = tabdat['xsigma_init'][oo]
            paramarray[oo*6+5] = tabdat['angle_init'][oo]
        else:
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
                    clobber=False,imgfiles=None,imgexts=None,imgnames=None,verbose=True):
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
    imgexts
    imgnames
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
    if verbose: print ' --- tdose_utilities.extract_subcube() --- '
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
        if verbose: print ' - Extracting sub-cube based on cutout bounding box of first layer'
        firstlayer    = 0
        cutout_layer  = Cutout2D(cubedata[firstlayer,:,:], skyc, size, wcs=cubewcs_2D, mode='partial')

        for key in cutout_layer.wcs.to_header().keys():
            striphdr[key] = cutout_layer.wcs.to_header()[key]
        hdrs_all.append(striphdr)

        manualcutting = False # always use quick solution (results are identical)
        if manualcutting:
            cutout_cube         = np.zeros([Nlayers,cutout_layer.data.shape[0],cutout_layer.data.shape[1]])
            for ll in xrange(Nlayers):
                if verbose:
                    infostr = '   cutting out from layer '+str("%6.f" % (ll+1))+' / '+str("%6.f" % Nlayers)
                    sys.stdout.write("%s\r" % infostr)
                    sys.stdout.flush()
                cutout_layer        = Cutout2D(cubedata[ll,:,:], skyc, size, wcs=cubewcs_2D)
                cutout_cube[ll,:,:] = cutout_layer.data
            if verbose: print '\n   done'
        else:
            cutout_cube = cubedata[:,cutout_layer.bbox_original[0][0]:cutout_layer.bbox_original[0][1]+1,
                          cutout_layer.bbox_original[1][0]:cutout_layer.bbox_original[1][1]+1]

        if cc == 0:
            cutouts = [cutout_cube]
        else:
            cutouts.append(cutout_cube)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Saving sub-cubes to '+outname
    hducube = pyfits.PrimaryHDU()  # creating default fits header
    hdulist = [hducube]

    for cc, cx in enumerate(cubeext):
        if verbose: print '   Add clean version of cube to extension               '+cx
        hducutout        = pyfits.ImageHDU(cutouts[cc])
        for key in hdrs_all[cc]:
            if not key in hducutout.header.keys():
                keyvalue = hdrs_all[cc][key]
                if type(keyvalue) == str:
                    keycomment = keyvalue.replace('Angstrom','A')
                hducutout.header.append((key,keyvalue,keycomment),end=True)
        hducutout.header.append(('EXTNAME ',cx            ,' '),end=True)
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
            if imgnames is None:
                imgname = imgfile.split('/')[-1]
                outname = outname.replace('.fits','_CUTOUT'+str(cutoutsize[0])+'x'+str(cutoutsize[1])+'arcsec_From_'+imgname)
            else:
                outname = imgnames[ii]

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
def gen_sourcecat_from_SExtractorfile(sextractorfile,outname='./tdose_sourcecat.txt',clobber=False,imgheader=None,
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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if imgheader is None:
        if verbose: print ' - Image header not provided; assuming ra and dec col are in pixel units'
    else:
        if verbose: print ' - Image header provided; converting ra and dec values using wcs info from header'
        striphdr   = tu.strip_header(imgheader.copy())
        wcs_in     = wcs.WCS(striphdr)
        skycoord   = SkyCoord(ras, decs, frame='icrs', unit='deg')
        pixcoord   = wcs.utils.skycoord_to_pixel(skycoord,wcs_in)
        xpos       = pixcoord[0]
        ypos       = pixcoord[1]

    if (clobber == False) & os.path.isfile(outname):
        if verbose: print ' - WARNING: Output ('+outname+') already exists and clobber=False, hence returning None'
        return None
    else:
        if verbose: print ' - Will save source catalog to '+outname+' (overwriting any existing file)'
        fout = open(outname,'w')
        fout.write('# TDOSE Source catalog generated with tdose_utilities.gen_sourcecat_from_SExtractorfile() from:\n')
        fout.write('# '+sextractorfile+'\n')
        if imgheader:
            fout.write('# id ra dec x_image y_image fluxscale \n')
            for ii, id in enumerate(ids):
                fout.write(str(ids[ii])+' '+str(ras[ii])+' '+str(decs[ii])+' '+str(xpos[ii])+' '+str(ypos[ii])+' '+
                           str(fluxes[ii])+'  \n')
        else:
            fout.write('# id x_image y_image fluxscale \n')
            for ii, id in enumerate(ids):
                fout.write(str(ids[ii])+' '+str(ras[ii])+' '+str(decs[ii])+' '+str(fluxes[ii])+'  \n')

        fout.close()
        return outname
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_paramlist_from_SExtractorfile(sextractorfile,pixscale=0.06,imgheader=None,clobber=False,objects='all',
                                      idcol='ID',racol='RA',deccol='DEC',aimg='A_IMAGE',bimg='B_IMAGE',
                                      angle='THETA_IMAGE',fluxscale='FLUX_ISO_F814W',fluxfactor=100.,Nsigma=3,
                                      saveDS9region=True,ds9regionname=None,ds9color='red',ds9width=2,ds9fontsize=12,
                                      savefitsimage=False,fitsimagename=None,
                                      savefitstable=False,fitstablename=None,verbose=True):
    """
    Generate parameter list for Gaussian source modeling with tdose_model_FoV.gen_fullmodel() based on SEctractor catalog

    --- INPUT ---

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    sexfile   = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/catalog_photometry_candels-cdfs-02.fits'
    imgheader = pyfits.open('/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/acs_814w_candels-cdfs-02_cut_v1.0.fits')[0].header
    paramlist = tu.gen_paramlist_from_SExtractorfile(sexfile,imgheader=imgheader,Nsigma=8,savefitsimage=False,objects=[10195])

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

    if objects == 'all':
        Nobjects_convert = Nobjects
    else:
        Nobjects_convert = len(objects)

    if verbose: print ' - Assembling paramter list for '+str(Nobjects_convert)+' sources found in catalog'
    paramlist = []
    for oo in xrange(Nobjects):
        if objects != 'all':
            if sourcedat[idcol][oo] not in objects: continue # skipping objects not in objects list provided
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
        if ds9regionname is None:
            ds9region_file = sextractorfile.replace('.fits','_tdose.reg')
        else:
            ds9region_file = ds9regionname

        if os.path.isfile(ds9region_file) & (clobber == False):
            if verbose: print ' - ds9 region file '+ds9region_file+' already exists and clobber=False so skipping'
        else:
            fout = open(ds9region_file,'w')
            fout.write("# Region file format: DS9 version 4.1 \nimage\n")

            for oo in xrange(Nobjects):
                if objects != 'all':
                    if sourcedat[idcol][oo] not in objects: continue # skipping objects not in objects list provided

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
        if fitsimagename is None:
            fitsimage = sextractorfile.replace('.fits','_tdose_modelimage.fits')
        else:
            fitsimage = fitsimagename

        if imgheader is None:
            if verbose: print ' - No image header provided for fits file ' \
                              '(to get wcs and image model image dimensions) so skipping'
        else:
            imgsize   = np.array([striphdr['NAXIS2'],striphdr['NAXIS1']])
            tmf.save_modelimage(fitsimage,paramlist_arr,imgsize,param_init=False,clobber=clobber,
                                outputhdr=imgheader,verbose=verbose,verbosemodel=verbose)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if savefitstable:
        if objects != 'all':
            Nobj = len(objects)
        else:
            Nobj = Nobjects

        objparam = np.resize(paramlist_arr,(Nobj,6))

        if verbose: print ' - Storing fitted source paramters as fits table and returning output'
        if fitstablename is None:
            tablename = sextractorfile.replace('.fits','_tdose_paramlist.fits')
        else:
            tablename = fitstablename

        objnumbers = np.arange(Nobj)+1
        c01 = pyfits.Column(name='obj',            format='D', unit='',       array=objnumbers)
        c02 = pyfits.Column(name='xpos',           format='D', unit='PIXELS', array=paramlist_arr[1::6])
        c03 = pyfits.Column(name='ypos',           format='D', unit='PIXELS', array=paramlist_arr[0::6])
        c04 = pyfits.Column(name='fluxscale',      format='D', unit='',       array=paramlist_arr[2::6])
        c05 = pyfits.Column(name='xsigma',         format='D', unit='PIXELS', array=paramlist_arr[4::6])
        c06 = pyfits.Column(name='ysigma',         format='D', unit='PIXELS', array=paramlist_arr[3::6])
        c07 = pyfits.Column(name='angle',          format='D', unit='DEGREES',array=paramlist_arr[5::6])

        coldefs = pyfits.ColDefs([c01,c02,c03,c04,c05,c06,c07])

        th = pyfits.new_table(coldefs) # creating default header

        tbHDU  = pyfits.new_table(coldefs, header=th.header)
        tbHDU.writeto(tablename, clobber=clobber)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return paramlist_arr
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_fitsextension(fitsfile,extension,outputname='default',conversion='None',useheader4output=False,clobber=False,
                          verbose=True):
    """
    Extract and extension from a fits file and save it as seperate fitsfile.
    Useful for preparing fits images for GALFIT run.

    --- EXAMPLE OF USE ---
    fitsfile = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/imgblock_6475_acs_814w.fits'
    tu.extract_fitsextension(fitsfile,2)

    fitsfile = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/acs_814w_candels-cdfs-02_wht_cut_v2.0.fits'
    tu.extract_fitsextension(fitsfile,0,conversion='ivar2sigma',useheader4output=True)


    """
    dataarr = pyfits.open(fitsfile)[extension].data

    if outputname.lower() == 'default':
        outputname = fitsfile.replace('.fits','_extension'+str(extension)+'.fits')

    if conversion == 'ivar2sigma':
        if verbose: print ' - Converting extension from units of inverse variance to sigma (standard deviation)'
        dataarr    = np.sqrt(1.0/dataarr)
        outputname = outputname.replace('.fits','_ivar2sigma.fits')

    if useheader4output:
        if verbose: print ' - Using header of input image for output'
        hdr = pyfits.open(fitsfile)[extension].header
    else:
        hdr = None

    if verbose: print ' - Saving extracted extension to '+outputname
    if os.path.isfile(outputname) & (clobber == False):
        sys.exit(' ----> Output file '+outputname+' already exists and clobber=False')
    else:
        pyfits.writeto(outputname,dataarr,header=hdr)
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def galfit_buildinput_fromssextractoroutput(filename,sexcatalog,image,imgext=0,sigmaimg='none',objecttype='gaussian',
                                            Nsigma=3,fluxfactor=100.,saveDS9region=True,savefitsimage=False,
                                            magzeropoint=26.5,platescale=[0.03,0.03],convolvebox=[100,100],psfimg='none',
                                            clobber=False,verbose=True):
    """
    Assemble a galfit input file from a SExtractor catalog.

    --- INPUT ---


    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    import pyfits
    image       = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/acs_814w_candels-cdfs-02_cut_v2.0.fits'
    sigmaimg    = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/acs_814w_candels-cdfs-02_wht_cut_v2.0_extension0_ivar2sigma.fits'
    psfimg      = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/imgblock_6475_acs_814w_extension2.fits'
    sexcatalog  = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/catalog_photometry_candels-cdfs-02.fits'
    fileout     = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/galfit_inputfile_acs_814w_candels-cdfs-02-sextractor.txt'
    tu.galfit_buildinput_fromssextractoroutput(fileout,sexcatalog,image,objecttype='gaussian',magzeropoint=25.947,sigmaimg=sigmaimg,platescale=[0.03,0.03],psfimg=psfimg,convolvebox=[500,500])

    """

    imgheader   = pyfits.open(image)[imgext].header

    param_sex = tu.gen_paramlist_from_SExtractorfile(sexcatalog,imgheader=imgheader,Nsigma=Nsigma,
                                                     fluxfactor=fluxfactor,saveDS9region=saveDS9region,
                                                     savefitsimage=savefitsimage,clobber=clobber,verbose=verbose)

    tu.galfit_buildinput_fromparamlist(filename,param_sex,image,objecttype=objecttype,sigmaimg=sigmaimg,psfimg=psfimg,
                                       platescale=platescale,magzeropoint=magzeropoint,convolvebox=convolvebox,verbose=True)
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def galfit_buildinput_fromparamlist(filename,paramlist,dataimg,sigmaimg='none',psfimg='none',badpiximg='none',
                                    objecttype='gaussian',ids=None,clobber=False,
                                    imgregion='full',imgext=0,convolvebox=[100,100],magzeropoint=26.5,
                                    platescale=[0.06,0.06],verbose=True):
    """
    Assemble a galfit input file from a TDOSE parameter list

    --- INPUT ---


    """
    if verbose: print(' - Assembling input file for GALFIT modeling based on TDOSE source parameter list')
    if os.path.isfile(filename) & (clobber==False):
        if verbose: print(' - '+filename+' already exists and clobber=False so not generating new version')
    else:
        if verbose: print(' - Will write setups to:\n   '+filename)
        fout =  open(filename, 'w')

        # NB Using no absolute paths as this can cause "Abort trap: 6" crash
        image2model      = dataimg.split('/')[-1]
        outputimg        = filename.replace('.txt','_galfitoutput.fits').split('/')[-1]
        sigmaimage       = sigmaimg.split('/')[-1]
        psfimage         = psfimg.split('/')[-1]
        psfsampling      = '1'
        badpiximage      = badpiximg.split('/')[-1]
        paramconstraints = 'none'

        if imgregion.lower() == 'full':
            imgshape         = pyfits.open(dataimg)[imgext].data.shape
            imageregion      = ' '.join([str(l) for l in [1,imgshape[1],1,imgshape[0]]])
        else:
            imageregion      = ' '.join([str(l) for l in imgregion])

        convolvebox      = ' '.join([str(l) for l in convolvebox])
        magzeropoint     = str(magzeropoint)
        platescale       = ' '.join([str(l) for l in platescale])
        displaytype      = 'regular'
        choice           = '0'


        headerstr = """#===============================================================================
# GALFIT input file generated with tdose_utilities.galfit_buildinput_fromparamlist() on %s
#===============================================================================
# IMAGE and GALFIT CONTROL PARAMETERS
A) %s               # Input data image (FITS file)
B) %s               # Output data image block
C) %s               # Sigma image name (made from data if blank or "none")
D) %s               # Input PSF image and (optional) diffusion kernel
E) %s               # PSF fine sampling factor relative to data
F) %s               # Bad pixel mask (FITS image or ASCII coord list)
G) %s               # File with parameter constraints (ASCII file)
H) %s               # Image region to fit (xmin xmax ymin ymax)
I) %s               # Size of the convolution box (x y)
J) %s               # Magnitude photometric zeropoint
K) %s               # Plate scale (dx dy)    [arcsec per pixel]
O) %s               # Display type (regular, curses, both)
P) %s               # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps

# INITIAL FITTING PARAMETERS
#
#   For object type, the allowed functions are:
#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat,
#       ferrer, powsersic, sky, and isophote.
#
#   Hidden parameters will only appear when they're specified:
#       C0 (diskyness/boxyness),
#       Fn (n=integer, Azimuthal Fourier Modes),
#       R0-R10 (PA rotation, for creating spiral structures).
#
# -----------------------------------------------------------------------------
#   par)    par value(s)    fit toggle(s)    # parameter description
# -----------------------------------------------------------------------------
""" % (tu.get_now_string(),
       image2model,
       outputimg,
       sigmaimage,
       psfimage,
       psfsampling,
       badpiximage,
       paramconstraints,
       imageregion,
       convolvebox,
       magzeropoint,
       platescale,
       displaytype,
       choice)

        fout.write(headerstr)
        if verbose: print('   wrote header to file')

        Nobj = len(paramlist)/6
        paramarr = np.resize(np.asarray(paramlist),(Nobj,6))

        if ids is None:
            ids = ['None provided']*Nobj

        for oo in xrange(Nobj):
            if verbose:
                infostr = '   writing setup for object '+str("%6.f" % (oo+1))+' / '+str("%6.f" % Nobj)
                sys.stdout.write("%s\r" % infostr)
                sys.stdout.flush()

            objparam = paramarr[oo] # [yposition,xposition,fluxscale,sigmay,sigmax,angle]
            fout.write('######### Object number: '+str(oo+1)+' (ID = '+ids[oo]+') #########')

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if objecttype == 'gaussian':
                position  = '   '.join([str(l) for l in [objparam[1],objparam[0],1,1]])
                mag       = '   '.join([str(l) for l in [-2.5*np.log10(objparam[2])+float(magzeropoint),1]])#GALFIT readme eq 34
                fwhm      = str(2.355*np.max(objparam[3:5]))+'  1  '
                axisratio = str(np.min(objparam[3:5])/np.max(objparam[3:5]))+' 1  '
                posangle  = '   '.join([str(l) for l in [objparam[5]-90,1]])
                Zval      = '0 '

                fout.write("""
 0) %s          #  object type
 1) %s          #  position x, y [pixel]
 3) %s          #  total magnitude
 4) %s          #  FWHM [pixels]
 9) %s          #  axis ratio (b/a)
10) %s          #  position angle (PA)  [Degrees: Up=0, Left=90]
 Z) %s          #  leave in [1] or subtract [0] this comp from data? \n \n""" %
                           (objecttype,position,mag,fwhm,axisratio,posangle,Zval) )
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            elif objecttype == 'sersic':
                sys.exit(' ---> Sersic galfit model setups not enabled ')
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print '\n   done; closing output file'
        fout.close()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def galfit_run(galfitinputfile,verbose=True,galfitverbose=False,noskyest=False):
    """
    Run galfit using a galfit input file (expects extension '.txt' for naming output file)
    and save the received command line output to file.

    NB: If the 'galfit' command is not available in the shell being launched with 'sh' add an executable
        file with the content:
            #!/bin/sh
            /Users/username/path/to/galfit/galfit "$@"
        to /usr/local/bin/

    --- INPUT ---
    galfitinputfile    GALFIT input file to run. Provide full path, as that is used to locate the
                       working (data) directory. It is assumed that only relative paths to all input
                       files are used in the input. Preferably they all liuve in the same directory as
                       the galfitinputfile. Cases of "Abort trap: 6" crashes of GALFIT have been while
                       using absolute paths in galfitinputfile run from different working directory

    --- EXAMPLE OF USE ---
    galfitinput  = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/galfit_inputfile_acs_814w_candels-cdfs-02-sextractor.txt'
    galfitoutput = tu.galfit_run(galfitinput,noskyest=False)


    """
    currentdir = os.getcwd()
    if verbose: print ' - Spawning GALFIT command to shell using the input file:\n   '+galfitinputfile
    datapath = '/'.join(os.path.abspath(galfitinputfile).split('/')[0:-1])+'/'
    if verbose: print '   (moving to '+datapath+' and working from there)'
    os.chdir(datapath)

    runcmd = 'galfit  '
    if noskyest:
        runcmd = runcmd+' -noskyest '
    runcmd = runcmd + galfitinputfile
    if verbose: print ' - Will run the GALFIT command:\n   '+runcmd

    outputfile = galfitinputfile.replace('.txt','_cmdlineoutput.txt')
    if verbose: print ' - Will save command line output from GALFIT to \n   '+outputfile
    fout = open(outputfile,'w')
    fout.write('####### output from GALFIT run with tdose_utilities.galfit_run() on '+tu.get_now_string()+' #######\n')
    fout.close()

    if verbose: print '   ----------- GALFIT run started on '+tu.get_now_string()+' ----------- '
    process = subprocess.Popen(runcmd, stdout=subprocess.PIPE, shell=True, bufsize=1, universal_newlines=True)
    while True:
        line = process.stdout.readline()
        #for line in iter(process.stdout.readline, ''):
        if line != b'':
            if galfitverbose:
                sys.stdout.write(line)

            fout = open(outputfile,'a')
            fout.write(line)
            fout.close()
        else:
            break
    if verbose: print '   ----------- GALFIT run finished on '+tu.get_now_string()+' ----------- '

    if verbose: print ' - Renaming and moving output files to image directory :'
    if os.path.isfile('./fit.log'):
        if verbose: print '   moving  ./fit.log'
        shutil.move('./fit.log',galfitinputfile.replace('.txt','_galfit.log'))

    fitoutfiles = glob.glob('./galfit.*')
    if len(fitoutfiles) > 0:
        for ff in fitoutfiles:
            if verbose: print '   moving  '+ff
            shutil.move(ff,galfitinputfile.replace('.txt','_'+ff.split('/')[-1].replace('.',''))+'result.txt')

    if verbose: print ' - Moving back to '+currentdir
    os.chdir(datapath)

    return outputfile
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def galfit_results2paramlist(galfitresults,verbose=True):
    """
    Load result file from GALFIT run and save it as a TDOSE object parameter files

    --- EXAMPLE OF USE ---
    file   = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/galfit_inputfile_acs_814w_candels-cdfs-02-sextractor_galfit01result.txt'
    param  = tu.galfit_results2paramlist(file)

    """

    paramlist = []

    fin       = open(galfitresults, 'r')
    modeltype = ''
    objno     = ''
    for ll, line in enumerate(fin):

        if line.startswith('J) '):
            magzp      = float( line.split('J) ')[-1].split('#')[0].split(' ')[0] )

        if line.startswith('# Component number'):
            objno      = int(line.split('number: ')[-1])
            if verbose: print ' - extracting infor for object number '+str(objno)

        if line.startswith(' 0) '):
            modeltype = line.split(' 0) ')[-1].split('#')[0].strip()

        # paramlist = [yposition,xposition,fluxscale,sigmay,sigmax,angle]
        if modeltype == 'gaussian':
            if line.startswith(' 1) '):
                posstrs   = line.split(' 1) ')[-1].split('#')[0].split(' ')
                paramlist.append(float(posstrs[1]))
                paramlist.append(float(posstrs[0]))


            if line.startswith(' 3) '):
                mag       = float( line.split(' 3) ')[-1].split('#')[0].split(' ')[0] )
                flux      = 10**( (mag+magzp)/-2.5 )
                paramlist.append(flux)

            if line.startswith(' 4) '):
                fwhm      = float( line.split(' 4) ')[-1].split('#')[0].split(' ')[0] )
                sigma     = fwhm/2.355

            if line.startswith(' 9) '):
                axisrat   = float( line.split(' 9) ')[-1].split('#')[0].split(' ')[0] )

            if line.startswith('10) '):
                angle     = float( line.split('10) ')[-1].split('#')[0].split(' ')[0] )

                sigmax    = sigma
                sigmay    = sigma*axisrat

                paramlist.append(sigmay)
                paramlist.append(sigmax)
                paramlist.append(angle)
                modeltype = ''
                objno     = ''

        elif (objno != '') & (modeltype != ''):
            if verbose: print ' - WARNING modeltype='+modeltype+' for object '+str(objno)+' is unknonw; not added to paramlist'
    fin.close()
    return paramlist
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =