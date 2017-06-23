# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import os
import datetime
import sys
from astropy import wcs
from astropy import units
from astropy import convolution
import astropy.convolution as ac # convolve, convolve_fft, Moffat2DKernel, Gaussian2DKernel
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
from scipy.stats import multivariate_normal
import matplotlib.pylab as plt
import pdb
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def load_setup(setupfile='./tdose_setup_template.txt',verbose=True):
    """
    Return dictionary with the setups found in 'setupfile'

    --- INPUT ---
    setupfile       The name of the txt file containing the TDOSE setup to load
                    A template for this setup file can be generated with
                    tdose_load_setup.generate_setup_template()
    verbose         Toggle verbosity

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
    outputfile      The name of the output which will contain the TDOSE setup template
    clobber         Overwrite files if they exist
    verbose         Toggle verbosity

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
# Generated with tdose_utilities.generate_setup_template() on %s
# The spectral extraction using this setup is run with tdose.perform_extraction()
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - DATA INPUT  - - - - - - - - - - - - - - - - - - - - - - - - - - -
data_cube              /path/datacube.fits                # Path and name of fits file containing data cube to extract spectra from
cube_extension         DATA_DCBGC                         # Name or number of fits extension containing data cube

variance_cube          /path/variancecube.fits            # Path and name of fits file containing variance cube to use for extraction
variance_extension     VARCUBE                            # Name or number of fits extension containing noise cube

ref_image              /path/referenceimage.fits          # Path and name of fits file containing image to use as reference when creating source model
img_extension          0                                  # Name or number of fits extension containing reference image

wht_image              /path/refimage_wht.fits            # Path and name of fits file containing weight map of reference image (only cut out; useful for galfit modeling)
wht_extension          0                                  # Name or number of fits extension containing weight map

ref_image_model        None                               # If a model of the reference image exists provide it here.
                                                          # If a model is provided the PSF convolution and flux optimization is done numerically.
model_extension        0                                  # Name or number of fits extension containing reference image model

source_catalog         /path/tdose_sourcecat.fits         # Path and name of source catalog containing sources to extract spectra for
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

models_directory       /path/tdose_models/                # Directory to store the modeling output from TDOSE in
cutout_directory       /path/tdose_cutouts/               # Directory to store image and cube cutouts in if model_cutouts=True
spec1D_directory       /path/tdose_spectra/               # Output directory to store spectra in.

# - - - - - - - - - - - - - - - - - - - - - - - - - - CUTOUT SETUP  - - - - - - - - - - - - - - - - - - - - - - - - - -
model_cutouts          True                               # Perform modeling and spectral extraction on small cutouts of the cube and images to reduce run-time
cutout_sizes           /path/tdose_setup_cutoutsizes.txt  # Size of cutouts [ra,dec] in arcsec around each source to model.
                                                          # To use source-specific cutouts provide ascii file containing ID xsize[arcsec] and ysize[arcsec].

# - - - - - - - - - - - - - - - - - - - - - - - - SOURCE MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
model_image_ext        tdose_modelimage                   # Name extension of fits file containing reference image model. To ignored use None
model_param_reg        tdose_modelimage_ds9               # Name extension of DS9 region file for reference image model. To ignored use None
model_image_cube_ext   tdose_modelimage_cubeWCS           # Name extension of fits file containing model image after conversion to cube WCS. To ignored use None.

source_model           gauss                              # The source model to use for sources. Choices are:
                                                          #   gauss          Each source is modeled as a multivariate gaussian using the source_catalog input as starting point
                                                          #   galfit         The sources in the field-of-view are defined based on GALFIT header parameters; if all components are        # Not enabled yet
                                                          #                  Gaussians an analytical convolution is performed. Otherwise numerical convolution is used.                   # Not enabled yet
                                                          #   modelimg       A model image exists, e.g., obtained with Galfit, in modelimg_directory. This prevents dis-entangling of     # Not enabled yet
                                                          #                  different objects, i.e., the provided model image is assumed to represent the 1 object in the field-of-view. # Not enabled yet
                                                          #                  If the model image is not found a gaussian model of the FoV (source_model=gauss) is performed instead        # Not enabled yet
                                                          #   aperture       A simple aperture extraction on the datacubes is performed, i.e., no modeling of sources.

# - - - - - - - - - - - - - - - - - - - - - - - - GAUSS MODEL SETUP - - - - - - - - - - - - - - - - - - - - - - - - - -
gauss_guess            /path/sextractoroutput.fits        # To base initial guess of gaussian parameters on a SExtractor output provide SExtractor output fits file here
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
galfit_directory       /path/models_galfit/               # If source_model = galfit provide path to directory containing galfit models.
                                                          # TDOSE will look for galfit_*ref_image*_output.fits (incl. the cutout string if model_cutouts=True)
                                                          # If no model is found a source_model=gauss run on the object will be performed instead.
galfit_model_extension 2                                  # Fits extension containing galfit model with model parameters of each source in header.

# - - - - - - - - - - - - - - - - - - - - - - - - MODEL IMAGE SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
modelimg_directory    /path/models_cutouts/               # If source_model = modelimg provide the path to directory containing the individual source models
                                                          # TDOSE will look for model_*ref_image*.fits (incl. the cutout string if model_cutouts=True)
                                                          # If no model is found (and no galfit model is found) a source_model=gauss run on the object will be performed instead.
modelimg_extension     2                                  # Fits extension containing model

# - - - - - - - - - - - - - - - - - - - - - - - - APERTURE MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - -
aperture_size          0.5                                # Radius of apertures to use given in arc seconds

# - - - - - - - - - - - - - - - - - - - - - - - - - PSF MODEL SETUP - - - - - - - - - - - - - - - - - - - - - - - - - -
psf_type               gauss                              # Select PSF model to build. Choices are:
                                                          #   gauss      Model the PSF as a symmetric Gaussian with sigma = FWHM/2.35482
psf_FWHM_evolve        linear                             # Evolution of the FWHM from blue to red end of data cube. Choices are:
                                                          #   linear     FWHM wavelength dependence described as FWHM(lambda) = p0[''] + p1[''/A] * (lambda - 7000A)
psf_FWHMp0             0.940                              # p0 parameter to use when determining wavelength dependence of PSF
psf_FWHMp1             -3.182e-5                          # p1 parameter to use when determining wavelength dependence of PSF
psf_savecube           True                               # To save fits file containing the PSF cube set psf_savecube = True

# - - - - - - - - - - - - - - - - - - - - - - - - - CUBE MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
model_cube_layers      'all'                              # Layers of data cube to model [both end layers included]. If 'all' the full cube will be modeled.
                                                          # To model source-specific layers provide ascii file containing ID layerlow and layerhigh.
                                                          # If layerlow=all and layerhigh=all all layers will be modeled for particular source
model_cube_optimizer   matrix                             # The optimizer to use when matching flux levels in cube layers:
                                                          #   matrix      Optimize fluxes analytically using matrix algebra to minimize chi squared of the equation set comparing model and data in each layer.
                                                          #   curvefit    Optimize fluxes numerically using least square fitting from scipy.optimize.curve_fit().
                                                          #               Only enabled for analytic convolution of Gaussian source models.
                                                          #   lstsq       Optimize fluxes analytically using scipy.linalg.lstsq().

model_cube_ext         tdose_modelcube                    # Name extension of fits file containing model data cube.
residual_cube_ext      tdose_modelcube_residual           # Name extension of fits file containing residual between model data cube and data. To ignored use None.
source_model_cube_ext  tdose_source_modelcube             # Name extension of fits file containing source model cube (used to modify data cube).

# - - - - - - - - - - - - - - - - - - - - - - - - SPECTRAL EXTRACTION - - - - - - - - - - - - - - - - - - - - - - - - -
sources_to_extract     [8685,9262,10195,29743]            # Sources in source_catalog to extract 1D spectra for.
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
plot_S2Nspec_yrange    [-1,15]                            # Range of y-axes (S2N) for plot of S2N spectra
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
    outputfile      The name of the output which will contain the TDOSE setup template
    clobber         Overwrite files if they exist
    verbose         Toggle verbosity

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
data_cube              /path/datacube.fits                # Path and name of fits file containing data cube to modify
cube_extension         DATA_DCBGC                         # Name or number of fits extension containing data cube
source_model_cube      /path/tdose_source_modelcube.fits  # Path and name of fits file containing source model cube
source_extension       DATA_DCBGC                         # Name or number of fits extension containing source model cube

modyified_cube         tdose_modified_datacube            # Name extension of file containing modified data cube.

modify_sources_list    [1,2,5]                            # List of IDs of sources to remove from data cube using source model cube.
                                                          # For long list of IDs provide path and name of file containing IDs (only)
sources_action         remove                             # Indicate how to modify the data cube. Chose between:
                                                          #    'remove'     Sources in modify_sources_list are removed from data cube
                                                          #    'keep'       All sources except the sources in modify_sources_list are removed from data cube
#----------------------------------------------END OF TDOSE MODIFY SETUP----------------------------------------------

""" % (tu.get_now_string())
        fout = open(outputfile,'w')
        fout.write(setuptemplate)
        fout.close()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def duplicate_setup_template(outputdirectory,infofile,infohdr=2,infofmt="S250",
                             loopcols=['data_cube','cube_extension'],
                             namebase='MUSEWide_tdose_setup',clobber=False,verbose=True):
    """
    ~ ~ ~ ~ STILL UNDER CONSTRUCTION/TESTING ~ ~ ~ ~

    Take a setup template generated with generate_setup_template() and duplicate it filling
    it with information from a provided infofile, e.g., fill update PSF info, field names,
    image names, source lists, etc.

    --- INPUT ---
    outputdirectory     Directory to store setup templates in
    infofile            File containing info to replace values in template setup with
    infohdr             Number of header (comment) lines in infofile before the expected list of column names
    infofmt             Format of columns in infofile (format for all columns are needed; not just loopcols)
                        If just a single format string is provided, this will be used for all columns.
    loopcols            The name of the columns in the loopcols to perform replacements for. The columns should
                        correspond to keywords in the TDOSE setup file. The first column of the file should be
                        named 'setupname' and will be used to name the duplicated setup file (appending it to namebase).
                        if 'all', all columns in infofile will be attempted replaced.
    namebase            Name base to use for the setup templates
    clobber             Overwrite files if they exist
    verbose             Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu

    outputdir = '/Users/kschmidt/work/TDOSE/muse_tdose_setups/'
    infofile  = outputdir+'musewide_infofile.txt'
    tu.duplicate_setup_template(outputdir,infofile,namebase='MUSEWide_tdose_setup',clobber=False,loopcols=['setupname','data_cube','cube_extension'])

    """
    if verbose: print ' --- tdose_utilities.duplicate_setup_template_MUSEWide() --- '

    filename = outputdirectory+namebase+'.txt'
    tu.generate_setup_template(outputfile=filename,clobber=clobber)

    if ',' not in infofmt: #if a single common format is given count columns in infofile
        copen     = np.genfromtxt(infofile,skip_header=infohdr,names=True)
        Ncol      = len(copen[0])
        infofmt   = ','.join([infofmt]*Ncol)

    copen     = np.genfromtxt(infofile,skip_header=infohdr,names=True,dtype=infofmt)

    if loopcols == 'all':
        if verbose: print ' - loopcals="all" so will attempt replacement of all columns in infofile'
        loopcols = np.asarray(copen.dtype.names).tolist()

    Nfiles    = len(copen[loopcols[0]])
    if verbose: print ' - Performing replacements and generating the '+str(Nfiles)+' TDOSE setup templates ' \
                                                                                   'described in \n   '+infofile

    for setupnumber in xrange(Nfiles):
        replacements = copen[setupnumber]
        newsetup     = outputdirectory+namebase+'_'+replacements['setupname']+'.txt'
        if os.path.isfile(newsetup) & (clobber == False):
            if verbose: print ' - File '+newsetup+' already exists and clobber = False so moving on to next duplication '
            continue
        else:
            fout = open(newsetup,'w')

            with open(filename,'r') as fsetup:
                for setupline in fsetup:
                    if setupline.startswith('#'):
                        if "Generated with tdose_utilities.generate_setup_template()" in setupline:
                            nowstring = tu.get_now_string()
                            fout.write("# Generated with tdose_utilities.duplicate_setup_template() on "+nowstring+' \n')
                        else:
                            fout.write(setupline)
                    elif setupline == '\n':
                        fout.write(setupline)
                    else:
                        vals = setupline.split()

                        if vals[0] in loopcols:
                            replaceline = setupline.replace(' '+vals[1]+' ',' '+copen[vals[0]][setupnumber]+' ')
                        else:
                            replaceline = setupline.replace(' '+vals[1]+' ',' NO_REPLACEMENT ')

                        newline     = replaceline.split('#')[0]+'#'+\
                                      '#'.join(setupline.split('#')[1:]) # don't include comment replacements
                        fout.write(newline)
        fout.close()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def build_2D_cov_matrix(sigmax,sigmay,angle,verbose=True):
    """
    Build a covariance matrix for a 2D multivariate Gaussian

    --- INPUT ---
    sigmax          Standard deviation of the x-compoent of the multivariate Gaussian
    sigmay          Standard deviation of the y-compoent of the multivariate Gaussian
    angle           Angle to rotate matrix by in degrees (clockwise) to populate covariance cross terms
    verbose         Toggle verbosity
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

    --- INPUT ---
    covmatrix       covariance matrix to normaliz
    verbose         Toggle verbosity

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
    cube                Data cube to be smoothed
    type                Type of smoothing kernel to apply
                            gauss      Use 2D gaussian smoothing kernel
                                        type_param expected:   [stdev,(stdev_wave_scale)]
                            moffat      Use a 2D moffat profile to represent the PSF
                                        type_param expected:   [gamma,alpha,(gamma_wave_scale,alpha_wave_scale)]
                        NB: If *wave_scale inputs are provided a list of scales to apply at each wavelength layer
                            (z-direction) of data cube is expected, hence, adding a wavelength dependence to the kernels.
    type_param          List of parameters for the smoothing kernel.
                        For expected paramters see notes in description of "type" keyword above.
    use_fftconvolution  Perform convolution in Foruire space with FFT
    verbose             Toggle verbosity

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
    verbose              Toggle verbosity

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
def gen_aperture(imgsize,ypos,xpos,radius,pixval=1,showaperture=False,verbose=True):
    """
    Generating an aperture image

    --- INPUT ---
    imgsize       The dimensions of the array to return. Expects [y-size,x-size].
                  The aperture will be positioned in the center of a (+/-x-size/2., +/-y-size/2) sized array
    ypos          Pixel position in the y direction
    xpos          Pixel position in the x direction
    radius        Radius of aperture in pixels
    showaperture  Display image of generated aperture
    verbose       Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    apertureimg  = tu.gen_aperture([20,40],10,5,10,showaperture=True)
    apertureimg  = tu.gen_aperture([2000,4000],900,1700,150,showaperture=True)

    """
    if verbose: print ' - Generating aperture in image (2D array)'
    y , x    = np.ogrid[-ypos:imgsize[0]-ypos, -xpos:imgsize[1]-xpos]
    mask     = x*x + y*y <= radius**2.
    aperture = np.zeros(imgsize)

    if verbose: print ' - Assigning pixel value '+str(pixval)+' to aperture'
    aperture[mask] = pixval

    if showaperture:
        if verbose: print ' - Displaying resulting image of aperture'
        plt.imshow(aperture,interpolation='none')
        plt.title('Generated aperture')
        plt.show()

    return aperture
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
    profile         profile to shift
    position        position to move center of image (profile) to:  [ypos,xpos]
                    NB! assumes position value starts from 0, i.e., if providing pixel values subtract 1.
    padvalue        the values to padd the images with when shifting profile
    showprofiles    Show profile when shifted?

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
    Note that the roll does not handle sub-pixel moves.
    tu.shift_2Dprofile() does this using interpolation

    --- INPUT ---
    profile         profile to shift
    position        position to move center of image (profile) to:  [ypos,xpos]
                    NB! assumes position value starts from 0, i.e., if providing pixel values subtract 1.
    padvalue        the values to padd the images with when shifting profile
    showprofiles    Show profile when shifted?

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

    --- INPUNT ---
    withseconds     To include seconds in the outputted string set this keyword to True

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

    --- INPUT ---
    imgsize         size of image to generate grid points for (y,x)

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

    --- INPUT ---
    mu1         The mean of the first gaussian
    covar1      The covariance matrix of of the first gaussian
    mu2         The mean of the second gaussian
    covar2      The covariance matrix of of the second gaussian

    """
    muconv    = mu1+mu2
    covarconv = covar1+covar2
    return muconv, covarconv

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def numerical_convolution_image(imgarray,kerneltype,saveimg=True,imgmask=None,fill_value=0.0,
                                norm_kernel=False,convolveFFT=False,verbose=True):
    """
    Perform numerical convolution on numpy array (image)

    --- INPUT ---
    imgarray      numpy array containing image to convolve
    kerneltype    Provide either a numpy array containing the kernel or an astropy kernel
                  to use for the convolution. E.g.,
                      astropy.convolution.Moffat2DKernel()
                      astropy.convolution.Gaussian2DKernel()
    saveimg       Save image of convolved imgarray
    imgmask       Mask of image array to apply during convolution
    fill_value    Fill value to use in convolution
    norm_kernel   To normalize the convolution kernel set this keyword to True
    convolveFFT   To convolve the image in fourier space set convolveFFT=True
    verbose       Toggle verbosity

    """
    if kerneltype is np.array:
        kernel    = kerneltype
        kernelstr = 'numpy array'
    else:
        kernel    = kerneltype
        kernelstr = 'astropy Guass/Moffat'

    if verbose: print ' - Convolving image with a '+kernelstr+' kernel using astropy convolution routines'

    if convolveFFT:
        img_conv = convolution.convolve_fft(imgarray, kernel, boundary='fill',
                                            fill_value=fill_value,normalize_kernel=norm_kernel, mask=imgmask,
                                            crop=True, return_fft=False, fft_pad=None,
                                            psf_pad=None, interpolate_nan=False, quiet=False,
                                            ignore_edge_zeros=False, min_wt=0.0)
    else:
        img_conv = convolution.convolve(imgarray, kernel, boundary='fill',
                                        fill_value=fill_value, normalize_kernel=norm_kernel, mask=imgmask)
    if saveimg:
        sys.exit(' ---> Saving convolved image is not enabled yet')

    return img_conv
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def convert_paramarray(paramarray,hdr,hdr_new,type='gauss',verbose=True):
    """
    Function to convert the pixel-based paramter array from one wcs frame to another

    --- INFO ---
    paramarray      Parameter array (e.g., loaded with build_paramarray)
    hdr             Header (wcs) information the parameter array referes to
    hdr_new         The header (wcs) information to us for transforming parameters to new reference frame
    type            The type of parameters to convert. Choose between
                        gauss       The paramarray contains 6 parameters for each source
                        aperture    The paramarray contains 4 parameters for each source
    verbose         Toggle verbosity

    """
    paramconv = np.zeros(paramarray.shape)

    wcs_in  = wcs.WCS(tu.strip_header(hdr.copy()))
    wcs_out = wcs.WCS(tu.strip_header(hdr_new.copy()))

    if wcs_out.to_header()['WCSAXES'] == 3:
        wcs_out = tu.WCS3DtoWCS2D(wcs_out)

    scale_in  = wcs.utils.proj_plane_pixel_scales(wcs_in)*3600.0   # pix scale in arcsec
    scale_out = wcs.utils.proj_plane_pixel_scales(wcs_out)*3600.0  # pix scale in arcsec

    if type == 'gauss':
        Nparam    = 6
        Nobj      = len(paramarray)/Nparam
        for oo in xrange(Nobj):
            ypix      = paramarray[oo*Nparam+0]-1.
            xpix      = paramarray[oo*Nparam+1]-1.
            skycoord  = wcs.utils.pixel_to_skycoord(xpix,ypix,wcs_in)
            pixcoord  = wcs.utils.skycoord_to_pixel(skycoord,wcs_out)# + np.array([1,1])
            paramconv[oo*Nparam+0] = pixcoord[1]
            paramconv[oo*Nparam+1] = pixcoord[0]
            paramconv[oo*Nparam+2] = paramarray[oo*Nparam+2]
            paramconv[oo*Nparam+3] = paramarray[oo*Nparam+3]*scale_in[0]/scale_out[0]
            paramconv[oo*Nparam+4] = paramarray[oo*Nparam+4]*scale_in[1]/scale_out[1]
            paramconv[oo*Nparam+5] = paramarray[oo*Nparam+5]
    elif type == 'aperture':
        Nparam    = 4
        Nobj      = len(paramarray)/4
        for oo in xrange(Nobj):
            ypix      = paramarray[oo*Nparam+0]-1.
            xpix      = paramarray[oo*Nparam+1]-1.
            skycoord  = wcs.utils.pixel_to_skycoord(xpix,ypix,wcs_in)
            pixcoord  = wcs.utils.skycoord_to_pixel(skycoord,wcs_out)# + np.array([1,1])
            paramconv[oo*Nparam+0] = pixcoord[1]
            paramconv[oo*Nparam+1] = pixcoord[0]
            paramconv[oo*Nparam+2] = paramarray[oo*Nparam+2]*scale_in[0]/scale_out[0]
            paramconv[oo*Nparam+3] = paramarray[oo*Nparam+3]
    else:
        sys.exit(' ---> Invalid type = '+type+' of parameters to convert')

    return paramconv
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def build_paramarray(fitstable,returninit=False,verbose=True):
    """
    Build parameter array (list) expected by tdose_model_cube.gen_fullmodel()
    based on output parameter fits file from tdose_model_FoV.gen_fullmodel()

    --- INPUT ---
    fitstable       fits table containing the fitted and intial source parameters
                    outputted by tdose_model_FoV.gen_fullmodel()
    returninit      Return the intiial parameters
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    path = '/Users/kschmidt/work/TDOSE/'
    file = 'mock_cube_sourcecat161213_all_tdose_mock_cube_NOISEgauss_v170207_modelimage_nosigma_objparam.fits'
    paramarray = tu.build_paramarray(path+file,verbose=True)
    """
    tabdat     = pyfits.open(fitstable)[1].data
    tabhdr     = pyfits.open(fitstable)[1].header
    try:
        paramtype = tabhdr['MODTYPE']
    except:
        if verbose: ' Did not find the keyword "MODTYPE" in the fits header; assuming the parameters are from gaussian models'
        paramtype = 'gauss'

    Nobj       = len(tabdat['obj'])

    if paramtype == 'gauss':
        Nparam     = 6
        paramarray = np.zeros([Nobj*Nparam])
        for oo in xrange(Nobj):
            if returninit:
                paramarray[oo*Nparam+0] = tabdat['ypos_init'][oo]
                paramarray[oo*Nparam+1] = tabdat['xpos_init'][oo]
                paramarray[oo*Nparam+2] = tabdat['fluxscale_init'][oo]
                paramarray[oo*Nparam+3] = tabdat['ysigma_init'][oo]
                paramarray[oo*Nparam+4] = tabdat['xsigma_init'][oo]
                paramarray[oo*Nparam+5] = tabdat['angle_init'][oo]
            else:
                paramarray[oo*Nparam+0] = tabdat['ypos'][oo]
                paramarray[oo*Nparam+1] = tabdat['xpos'][oo]
                paramarray[oo*Nparam+2] = tabdat['fluxscale'][oo]
                paramarray[oo*Nparam+3] = tabdat['ysigma'][oo]
                paramarray[oo*Nparam+4] = tabdat['xsigma'][oo]
                paramarray[oo*Nparam+5] = tabdat['angle'][oo]
    elif paramtype == 'aperture':
        Nparam       = 4
        paramarray = np.zeros([Nobj*Nparam])
        for oo in xrange(Nobj):
            if returninit:
                paramarray[oo*Nparam+0] = tabdat['ypos_init'][oo]
                paramarray[oo*Nparam+1] = tabdat['xpos_init'][oo]
                paramarray[oo*Nparam+2] = tabdat['radius_init'][oo]
                paramarray[oo*Nparam+3] = tabdat['pixvalue_init'][oo]
            else:
                paramarray[oo*Nparam+0] = tabdat['ypos'][oo]
                paramarray[oo*Nparam+1] = tabdat['xpos'][oo]
                paramarray[oo*Nparam+2] = tabdat['radius'][oo]
                paramarray[oo*Nparam+3] = tabdat['pixvalue'][oo]
    else:
        sys.exit(' ---> Unknown MODTYPE = '+paramtype+' in fits header')
    return paramarray
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def WCS3DtoWCS2D(wcs3d,verbose=True):
    """
    Removing the wavelength component of a WCS object, i.e., converting
    the WCS from 3D (lambda,ra,dec) to 2D (ra,dec)

    --- INPUT ---
    wcs3d       The WCS object to convert from (lambda,ra,dec) to (ra,dec)
    verbose     Toggle verbosity

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
    imgexts        The extension of of the images
    imgnames       The names of the images
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

    --- INPUT ---
    header      Header to strip from COMMENT and HISTORY entries
    verbose     Toggle verbosity

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
    Generate a DS9 region file based on a model parameters file.
    Model type represented by parameters is determined from fits header.

    --- INPUT ---
    fitstable     Fits file with information for model parameters in header to generate ds9 region file for
    outputfile    Name of output file to save region file to
    wcsinfo       WCS information to use when converting coordinates to RA and Dec
    color         Color of regions
    width         Width of lines of regions
    Nsigma        Number of sigmas to use as radii when plotting circles/ellipses
    textlist      List of text strings to add to the regions
    fontsize      Font size of region title text
    clobber       Overwrite region file if it already exists
    verbose       Toggle verbosity

    """
    tabhdr = pyfits.open(fitstable)[1].header
    try:
        paramtype = tabhdr['MODTYPE']
    except:
        if verbose: ' Did not find the keyword "MODTYPE" in the fits header; assuming the parameters are from gaussian models'
        paramtype = 'gauss'

    if verbose: print ' - Generating DS9 region file from model paramter file'
    paramarray = tu.build_paramarray(fitstable,verbose=True)
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

    if paramtype == 'gauss':
        Nparam = 6
    elif paramtype == 'aperture':
        Nparam = 4
    else:
        sys.exit(' ---> Invalid MODEL type keyword provide to tu.model_ds9region()')

    Nobj       = len(paramarray)/Nparam
    if verbose: print ' - Converting to wcs coordinates'
    for oo in xrange(Nobj):
        ypix      = paramarray[oo*Nparam+0]
        xpix      = paramarray[oo*Nparam+1]
        skycoord  = wcs.utils.pixel_to_skycoord(xpix,ypix,wcsinfo)
        dec       = skycoord.dec.value
        ra        = skycoord.ra.value

        if paramtype == 'gauss':
            fluxscale = paramarray[oo*Nparam+2]
            sigmay    = paramarray[oo*Nparam+3]*scale[0]/3600.0
            sigmax    = (paramarray[oo*Nparam+4]*scale[1]*np.cos(np.deg2rad(dec)))/3600.0
            angle     = paramarray[oo*Nparam+5]
            string    = 'ellipse('+str(ra)+','+str(dec)+','+str(Nsigma*sigmax)+','+str(Nsigma*sigmay)+','+str(angle)+') '
        elif paramtype == 'aperture':
            radius    = paramarray[oo*Nparam+2]*scale[0]
            string    = 'circle('+str(ra)+','+str(dec)+','+str(radius)+'") '
        else:
            sys.exit(' ---> Invalid MODEL type keyword provide to tu.model_ds9region()')

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

    --- INPUT ---
    sextractorfile      SExtractor file to generate source catalog from.
    outname             Name of source catalog to generate
    clobber             Overwrite files if they exists
    imgheader           Fits header with WCS information to convert ra and dec into pixel values
    idcol               Column number or column name of column containing the source IDs
    racol               Column number or column name of column containing the R.A. (imgheader needs to be provided) or
                        x-direction pixel position of sources
    deccol              Column number or column name of column containing the Dec. (imgheader needs to be provided) or
                        y-direction pixel position of sources
    fluxcol             Column number or column name of column containing flux scaling of sources
    fluxfactor          Factor to apply to fluxcol values
    verbose             Toggle verbosity

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
    sextractorfile      SExtractor output file to generate parameter list from
    pixscale            Pixel scale of image in arcsec/pix
    imgheader           Image header containing WCS to use for converting R.A. and Dec. values to pixel positions
    clobber             Overwrite files if they already exist
    objects             List of objects to generate parameter list for. If 'all', parameter list will contain all objects
    idcol               Name of column containing object IDs
    racol               Name of column containing R.A.s (imgheader needed) or x-pixel positions
    deccol              Name of column containing Dec.s (imgheader needed) or y-pixel positions
    aimg                Name of column containing the major axis of the source from the SExtractor fit
    bimg                Name of column containing the minor axis of the source from the SExtractor fit
    angle               Name of column containing the angle of the source from the SExtractor fit
    fluxscale           Name of column containing fluxes for sources
    fluxfactor          Factor to scale fluxscale values with
    Nsigma              The number of sigma to use in parameter list
    saveDS9region       Generate DS9 region file of sources
    ds9regionname       Name of DS9 region file to generate
    ds9color            Color of DS9 regions
    ds9width            Line widht to draw DS9 regions width
    ds9fontsize         Font size of DS9 region titles
    savefitsimage       Store image to fits file?
    fitsimagename       Name of fits image containign tdose model if it exists and is not
                        sextractorfile.replace('.fits','_tdose_modelimage.fits')
    savefitstable       Store parameter to fits table?
    fitstablename       Name of fits table to stor parameter list to
    verbose             Toggle verbosity

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

    --- INPUT ---
    fitsfile            Fits file to extract extension from
    extension           Extension to extract from fitsfile
    outputname          Name of file to store extracted extension to
    conversion          Conversion to apply to extrension values. Choose between:
                            ivar2sigma      converting inverse variance values to sigma values
    useheader4output    Use header of input image for output image?
    clobber             Overwrite files if they already exist
    verbose             Toggle verbosity

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
    filename            Name of file to store GALFIT input in
    sexcatalog          SExtractor output file to generate parameter list from
    image               Image the SExtractor catalog corresponds to
    imgext              Extention of fits image to use
    sigmaimg            Sigma image to includ in GALFIT setup
    objecttype          The type of objects to model. Choose between:
                            gaussian
                            sersic
    Nsigma              Number of sigma to use for object sizes
    fluxfactor          Factor to scale fluxscale values with
    saveDS9region       Generate DS9 region file of sources
    savefitsimage       Store image to fits file?
    magzeropoint        Zero point to adopt for photometry
    platescale          The pixel sizes in untes of arcsec/pix
    convolvebox         Size of convolution box for GALFIT to use
    psfimg              PSF image to use
    clobber             Overwrite files if they already exist
    verbose             Toggle verbosity

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
    filename            Name of file to store GALFIT input in
    paramlist           TDOSE parameter list to build galfit input from
    dataimg             Image the SExtractor catalog corresponds to
    sigmaimg            Sigma image to include in GALFIT setup
    psfimg              PSF image to include in GALFIT setup
    badpiximg           Bad pixel image to include in GALFIT setup
    objecttype          The type of objects to model. Choose between:
                            gaussian
                            sersic
    ids                 Ids to assign to each source in GALFIT input file
    clobber             Overwrite files if they already exist
    imgregion           Region of image to set as model region in GALFIT file
    imgext              Extention of fits image to use
    convolvebox         Size of convolution box for GALFIT to use
    magzeropoint        Zero point to adopt for photometrymagzeropoint
    platescale          The pixel sizes in untes of arcsec/pix
    verbose             Toggle verbosity

    """
    if verbose: print(' - Assembling input file for GALFIT modeling based on TDOSE source parameter list')
    if os.path.isfile(filename) & (clobber==False):
        if verbose: print(' - '+filename+' already exists and clobber=False so not generating new version')
    else:
        if verbose: print(' - Will write setups to:\n   '+filename)
        fout =  open(filename, 'w')

        # NB Using no absolute paths as this can cause "Abort trap: 6" crash
        image2model      = dataimg #.split('/')[-1]
        outputimg        = filename.replace('.txt','_galfitoutput.fits').split('/')[-1]
        sigmaimage       = sigmaimg #.split('/')[-1]
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
def galfit_buildinput_multiGaussTemplate(filename,dataimg,Ngauss=4,gaussspacing=3,sigmays=[2],sigmaxs=[2],fluxscales=[22],angles=[0.0],
                                         sigmaimg='none',psfimg='none',badpiximg='none',
                                         imgregion='full',imgext=0,convolvebox=[150,150],magzeropoint=26.5,
                                         platescale=[0.03,0.03],clobber=False,verbose=True):
    """
    Assemble a galfit input file for a grid of gaussians ("poor-man gaussian mixture")

    --- INPUT ---
    filename            Name of file to store GALFIT input in
    dataimg             Image the SExtractor catalog corresponds to
    Ngauss              The number of gaussians to place in the grid
    gaussspacing        Spacing between each gaussian
    sigmays             Sigma in x direction of gaussians
    sigmaxs             Sigma in x direction of gaussians
    fluxscales          fluxscales of gaussians
    angles              Angles of gaussians
    sigmaimg            Sigma image to include in GALFIT setup
    psfimg              PSF image to include in GALFIT setup
    badpiximg           Bad pixel image to include in GALFIT setup
    imgregion           Region of image to set as model region in GALFIT file
    imgext              Extention of fits image to use
    convolvebox         Size of convolution box for GALFIT to use
    magzeropoint        Zero point to adopt for photometrymagzeropoint
    platescale          The pixel sizes in untes of arcsec/pix
    clobber             Overwrite files if they already exist
    verbose             Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu

    outputpath  = '/Volumes/DATABCKUP2/MUSE-Wide/galfitresults/'
    cutoutpath  = '/Volumes/DATABCKUP2/TDOSEextractions/tdose_cutouts/'
    dataimage   = cutoutpath+'acs_814w_candels-cdfs-02_cut_v1.0_id8685_cutout7p0x7p0arcsec.fits'
    sigmaimg    = cutoutpath+'acs_814w_candels-cdfs-02_wht_cut_v1.0_id9262_cutout6p0x6p0arcsec_sigma_smooth_gaussSig3pix.fits'
    psfimg      = outputpath+'imgblock_6475_acs_814w_extension2.fits'
    outputfile  = outputpath+'galfit_inputfile_'+dataimage.split('/')[-1].replace('.fits','.txt')

    tu.galfit_buildinput_multiGaussTemplate(outputfile,dataimage,Ngauss=9,gaussspacing=5,sigmaimg=sigmaimg,psfimg=psfimg,clobber=True)

    """
    if verbose: print(' - Building parameter list for '+str(Ngauss)+' gaussian components ')
    paramlist      = np.ones(Ngauss*6)
    imgshape       = pyfits.open(dataimg)[imgext].data.shape

    Ngaussperrow   = np.ceil(np.sqrt(Ngauss))
    positiongrid   = gen_gridcomponents([Ngaussperrow*gaussspacing,Ngaussperrow*gaussspacing])

    ylowcorner     = np.round(imgshape[0]/2.)-(Ngaussperrow*gaussspacing/2.)
    xlowcorner     = np.round(imgshape[1]/2.)-(Ngaussperrow*gaussspacing/2.)


    ids    = []
    Nparam = 6
    for oo in xrange(Ngauss):
        objno = str("%.4d" % (oo+1))
        ids.append(objno)

        yposition = ylowcorner + np.floor(oo/Ngaussperrow)*gaussspacing
        xposition = xlowcorner + (oo/Ngaussperrow - int(oo/Ngaussperrow)) * Ngaussperrow * gaussspacing
        if len(fluxscales) == 1:
            fluxscale = fluxscales[0]
        else:
            fluxscale = fluxscales[oo]

        if len(sigmays) == 1:
            sigmay = sigmays[0]
        else:
            sigmay = sigmays[oo]

        if len(sigmays) == 1:
            sigmax = sigmaxs[0]
        else:
            sigmax = sigmaxs[oo]

        if len(angles) == 1:
            angle = angles[0]
        else:
            angle = angles[oo]

        paramlist[oo*Nparam:oo*Nparam+Nparam] = [yposition,xposition,fluxscale,sigmay,sigmax,angle]

    if verbose: print(' - Build galfit template input using tu.galfit_buildinput_fromparamlist()')
    tu.galfit_buildinput_fromparamlist(filename,paramlist,dataimg,objecttype='gaussian',sigmaimg=sigmaimg,psfimg=psfimg,ids=ids,
                                       imgregion=imgregion,badpiximg=badpiximg,platescale=platescale,magzeropoint=magzeropoint,
                                       convolvebox=convolvebox,imgext=imgext,clobber=clobber,verbose=verbose)

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
    verbose            toggle verbosity
    galfitverbose      toggle verbosity from galfit
    noskyest           Estimate sky when running galfit?

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
    crashed = False
    Niter   = 'None' # resetting counter
    Ncount  = 'None' # resetting counter

    while True:
        line = process.stdout.readline()
        #for line in iter(process.stdout.readline, ''):
        if line != b'':
            if galfitverbose:
                sys.stdout.write(line)

            fout = open(outputfile,'a')
            fout.write(line)
            fout.close()

            if verbose:
                if 'Iteration :' in line:
                    Niter  = line.split('Iteration : ')[1][:3]
                if 'COUNTDOWN =' in line:
                    Ncount = line.split('COUNTDOWN = ')[1][:3]

                if (Niter != 'None') & (Ncount != 'None'):
                    infostr = '   GALFIT Iteration '+str("%6.f" % float(Niter))+' at countdown '+str("%6.f" % float(Ncount))
                    sys.stdout.write("%s\r" % infostr)
                    sys.stdout.flush()
                    Niter  = 'None' # resetting counter
                    Ncount = 'None' # resetting counter

            if 'Doh!  GALFIT crashed' in line:
                crashed = True
        else:
            break

    if crashed:
        if verbose: print '\n WARNING - looks like galfit crashed with a mushroom cloud...'
    if verbose: print '\n   ----------- GALFIT run finished on '+tu.get_now_string()+' ----------- '

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
    os.chdir(currentdir)

    return outputfile
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def galfit_results2paramlist(galfitresults,verbose=True):
    """
    Load result file from GALFIT run and save it as a TDOSE object parameter files

    --- INPUT ---
    galfitresults       Output from running GALFIT
    verbose             Toggle verbosity

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
def reshape_array(array, newsize, pixcombine='sum'):
    """
    Reshape an array to a give size using either the sum, mean or median of the pixels binned

    Note that the old array dimensions have to be multiples of the new array dimensions

    --- INPUT ---
    array           Array to reshape (combine pixels)
    newsize         New size of array
    pixcombine      The method to combine the pixels with. Choices are sum, mean and median

    """
    sh = newsize[0],array.shape[0]//newsize[0],newsize[1],array.shape[1]//newsize[1]
    pdb.set_trace()
    if pixcombine == 'sum':
        reshapedarray = array.reshape(sh).sum(-1).sum(1)
    elif pixcombine == 'mean':
        reshapedarray = array.reshape(sh).mean(-1).mean(1)
    elif pixcombine == 'median':
        reshapedarray = array.reshape(sh).median(-1).median(1)

    return reshapedarray
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def get_datinfo(cutoutid,setupdic):
    """
    Function returning information on file names etc. for both default run and cutout run

    --- INPUT ---
    cutoutid        ID to return information for
    setupdic        Dictionary containing the setup parameters read from the TDOSE setup file

    """
    if cutoutid == -9999:
        cutstr       = None
        imgsize      = setupdic['cutout_sizes']
        refimg       = setupdic['ref_image']
        datacube     = setupdic['data_cube']
        variancecube = setupdic['noise_cube']
        sourcecat    = setupdic['source_catalog']
    else:
        if type(setupdic['cutout_sizes']) == np.str_:
            sizeinfo = np.genfromtxt(setupdic['cutout_sizes'],dtype=None,comments='#')
            objent   = np.where(sizeinfo[:,0] == cutoutid)[0]

            if len(objent) > 1:
                sys.exit(' ---> More than one match in '+setupdic['cutout_sizes']+' for object '+str(cutoutid))
            elif len(objent) == 0:
                sys.exit(' ---> No match in '+setupdic['cutout_sizes']+' for object '+str(cutoutid))
            else:
                imgsize   = sizeinfo[objent,1:][0].astype(float).tolist()
        else:
            imgsize   = setupdic['cutout_sizes']

        cutstr          = ('_id'+str(int(cutoutid))+'_cutout'+str(imgsize[0])+'x'+str(imgsize[1])+'arcsec').replace('.','p')
        img_init_base   = setupdic['ref_image'].split('/')[-1]
        cube_init_base  = setupdic['data_cube'].split('/')[-1]
        var_init_base   = setupdic['variance_cube'].split('/')[-1]

        cut_img         = setupdic['cutout_directory']+img_init_base.replace('.fits',cutstr+'.fits')
        cut_cube        = setupdic['cutout_directory']+cube_init_base.replace('.fits',cutstr+'.fits')
        cut_variance    = setupdic['cutout_directory']+var_init_base.replace('.fits',cutstr+'.fits')
        cut_sourcecat   = setupdic['source_catalog'].replace('.fits',cutstr+'.fits')

        if setupdic['wht_image'] is None:
            refimg          = cut_img
        else:
            wht_init_base   = setupdic['wht_image'].split('/')[-1]
            wht_img         = setupdic['cutout_directory']+wht_init_base.replace('.fits',cutstr+'.fits')
            refimg          = [cut_img,wht_img]

        datacube        = cut_cube
        variancecube    = cut_variance
        sourcecat       = cut_sourcecat


    return cutstr, imgsize, refimg, datacube, variancecube, sourcecat
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =