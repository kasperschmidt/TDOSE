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
from astropy.modeling.models import Sersic1D
from astropy.modeling.models import Sersic2D
from astropy.nddata import Cutout2D
import pyfits
import subprocess
import glob
import shutil
import scipy.ndimage
import scipy.special
import scipy.integrate as integrate
import tdose_utilities as tu
import tdose_model_FoV as tmf
from scipy.stats import multivariate_normal
import matplotlib as mpl
mpl.use('Agg') # prevent pyplot from opening window; enables closing ssh session with detached screen running TDOSE
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

        if setup_arr[ii,0] in setup_dic.keys():
            sys.exit(' Setup parameter "'+setup_arr[ii,0]+'" appears multiple times in the setup file\n             '+
                     setupfile)

        dirs = ['sources_to_extract','model_cube_layers','cutout_sizes']
        if (setup_arr[ii,0] in dirs) & ('/' in setup_arr[ii,1]):
            val = setup_arr[ii,1]
            setup_dic[setup_arr[ii,0]] = val
            continue

        lists = ['modify_sources_list','nondetections','model_cube_layers','sources_to_extract','plot_1Dspec_xrange','plot_1Dspec_yrange',
                 'plot_S2Nspec_xrange','plot_S2Nspec_yrange','cutout_sizes']
        if (setup_arr[ii,0] in lists) & (setup_arr[ii,1] != 'all') & (setup_arr[ii,1].lower() != 'none'):
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
    tu.generate_setup_template(outputfile=filename,clobber=False)
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
# Template for Three Dimensional Optimal Spectral Extracion (TDOSE, http://github.com/kasperschmidt/TDOSE) setup file
# Template was generated with tdose_utilities.generate_setup_template() on %s
# Setup file can be run with tdose.perform_extraction() or tdose.perform_extractions_in_parallel()
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

source_catalog         /path/tdose_sourcecat.fits         # Path and name of source catalog containing sources to extract spectra for
sourcecat_IDcol        id                                 # Column containing source IDs in source_catalog
sourcecat_xposcol      x_image                            # Column containing x pixel position in source_catalog
sourcecat_yposcol      y_image                            # Column containing y pixel position in source_catalog
sourcecat_racol        ra                                 # Column containing ra  position in source_catalog (used to position cutouts if model_cutouts = True)
sourcecat_deccol       dec                                # Column containing dec position in source_catalog (used to position cutouts if model_cutouts = True)
sourcecat_fluxcol      fluxscale                          # Column containing a flux scale used for the modeling if no gauss_guess is provided
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
                                                          #   modelimg       A model image exists, e.g., obtained with Galfit, in modelimg_directory. This prevents dis-entangling of
                                                          #                  different objects, i.e., the provided model image is assumed to represent the 1 object in the field-of-view.
                                                          #                  If the model image is not found a gaussian model of the FoV (source_model=gauss) is performed instead.
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

max_centroid_shift     10                                 # The maximum shift of the centroid of each source allowed in the gaussian modeling. Given in pixels to
                                                          # set bounds ypix_centroid +/- max_centroid_shift and xpix_centroid +/- max_centroid_shift
                                                          # If none, no bounds are put on the centroid position of the sources.
# - - - - - - - - - - - - - - - - - - - - - - - - GALFIT MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
galfit_directory       /path/models_galfit/               # If source_model = galfit provide path to directory containing galfit models.
                                                          # TDOSE will look for galfit_*ref_image*_output.fits (incl. the cutout string if model_cutouts=True)
                                                          # If no model is found a source_model=gauss run on the object will be performed instead.
galfit_model_extension 2                                  # Fits extension containing galfit model with model parameters of each source in header.

# - - - - - - - - - - - - - - - - - - - - - - - - MODEL IMAGE SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
modelimg_directory     /path/models_cutouts/              # If source_model = modelimg provide the path to directory containing the individual source models
                                                          # TDOSE will look for model_*ref_image*.fits (incl. the cutout string if model_cutouts=True). If no model is found the object is skipped
                                                          # If a model image named model_*ref_image*_cube.fits is foound, TDOSE assumes this file contains a cube with the individual model
                                                          # components isolated in individual layers of the cube. TDOSE will instead use this model and expects a file named model_*ref_image*_cube_compkey.txt
                                                          # Defining what components belong to the object of interest (i.e., to extract a spectrum for) and what components are contaminating sources in the field-of-view.

modelimg_extension     0                                  # Fits extension containing model

# - - - - - - - - - - - - - - - - - - - - - - - - APERTURE MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - -
aperture_size          1.5                                # Radius of apertures to use given in arc seconds

# - - - - - - - - - - - - - - - - - - - - - - - - - PSF MODEL SETUP - - - - - - - - - - - - - - - - - - - - - - - - - -
psf_type               gauss                              # Select PSF model to build. Choices are:
                                                          #   gauss      Model the PSF as a symmetric Gaussian with sigma = FWHM/2.35482
                                                          #   kernel_gauss   An astropy.convolution.Gaussian2DKernel() used for numerical convolution                        # Not enabled yet
                                                          #   kernel_moffat  An astropy.convolution.Moffat2DKernel()   used for numerical convolution                        # Not enabled yet
psf_FWHM_evolve        linear                             # Evolution of the FWHM from blue to red end of data cube. Choices are:
                                                          #   linear     FWHM wavelength dependence described as FWHM(lambda) = p0[''] + p1[''/A] * (lambda - 7000A)
psf_FWHMp0             0.940                              # p0 parameter to use when determining wavelength dependence of PSF
psf_FWHMp1             -3.182e-5                          # p1 parameter to use when determining wavelength dependence of PSF
psf_savecube           True                               # To save fits file containing the PSF cube set psf_savecube = True
                                                          # This cube is used for the "source_model = modelimg" numerical PSF convolution

# - - - - - - - - - - - - - - - - - - - - - - - - - - - NON_DETECTIONS  - - - - - - - - - - - - - - - - - - - - - - - -
nondetections          None                               # List of IDs of sources in source_catalog that are not detected in the reference image or which
                                                          # have low flux levels in which cases the Gaussian modeling is likely to be inaccurate.
                                                          # For long list of objects provide ascii file containing ids.
                                                          #     If source_model = gauss    then sources will be extracted by replacing models within ignore_radius
                                                          #                                with a single point source in the reference image model, which will then
                                                          #                                be convolved with the PSF specified when extracting, as usual.
                                                          #     If source_model = modelimg TDOSE assumes that the model already represents the desired extraction model
                                                          #                                of the non-detection. I.e., if the object should be extracted as a (PSF
                                                          #                                convolved) point source, the model image should include a point source.
ignore_radius          0.5                                # Models within a radius of ignore_radius [arcsec] of the non-detection location will be replaced with a
                                                          # point source for extractions with source_model = gauss before convolving with the PSF and adjusting the flux
                                                          # leves in each model cube layer.
# - - - - - - - - - - - - - - - - - - - - - - - - - CUBE MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
model_cube_layers      all                                # Layers of data cube to model [both end layers included]. If 'all' the full cube will be modeled.
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
def gen_2Dgauss(size,cov,scale,method='scipy',show2Dgauss=False,savefits=False,verbose=True):
    """
    Generating a 2D gaussian with specified parameters

    --- INPUT ---
    size          The dimensions of the array to return. Expects [ysize,xsize].
                  The 2D gauss will be positioned in the center of the array
    cov           Covariance matrix of gaussian, i.e., variances and rotation
                  Can be build with cov = build_2D_cov_matrix(stdx,stdy,angle)
    scale         Scaling the 2D gaussian. By default scale = 1 returns normalized 2D Gaussian.
                  I.e.,  np.trapz(np.trapz(gauss2D,axis=0),axis=0) = 1
    method        Method to use for generating 2D gaussian:
                   'scipy'    Using the class multivariate_normal from the scipy.stats library
                   'matrix'   Use direct matrix expression for PDF of 2D gaussian               (slow!)
    show2Dgauss   Save plot of generated 2D gaussian
    savefits      Save generated profile to fits file
    verbose       Toggler verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    covmatrix   = tu.build_2D_cov_matrix(4,1,5)
    gauss2Dimg  = tu.gen_2Dgauss([20,40],covmatrix,5,show2Dgauss=True)
    gauss2Dimg  = tu.gen_2Dgauss([9,9],covmatrix,1,show2Dgauss=True)

    sigmax          = 3.2
    sigmay          = 1.5
    covmatrix       = tu.build_2D_cov_matrix(sigmax,sigmay,0)
    scale           = 1 # returns normalized gaussian
    Nsigwidth       = 15
    gauss2DimgNorm  = tu.gen_2Dgauss([sigmay*Nsigwidth,sigmax*Nsigwidth],covmatrix,scale,show2Dgauss=True,savefits=True)

    covmatrix       = tu.build_2D_cov_matrix(4,2,45)
    scale           = 1 # returns normalized gaussian
    gauss2DimgNorm  = tu.gen_2Dgauss([33,33],covmatrix,scale,show2Dgauss=True,savefits=True)

    """
    if verbose: print ' - Generating multivariate_normal object for generating 2D gauss using ',
    if method == 'scipy':
        if verbose: print ' scipy.stats.multivariate_normal.pdf() '
        mvn     = multivariate_normal([0, 0], cov)

        if verbose: print ' - Setting up grid to populate with 2D gauss PDF'
        #x, y = np.mgrid[-np.ceil(size[0]/2.):np.floor(size[0]/2.):1.0, -np.ceil(size[1]/2.):np.floor(size[1]/2.):1.0] #LT170707
        x, y = np.mgrid[-np.floor(size[0]/2.):np.ceil(size[0]/2.):1.0, -np.floor(size[1]/2.):np.ceil(size[1]/2.):1.0]
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

    if float(size[0]/2.) - float(int(size[0]/2.)) == 0.0:
        ypos   = np.asarray(size[0])/2.0-1.0
    else:
        ypos   = np.floor(np.asarray(size[0])/2.0)

    if float(size[1]/2.) - float(int(size[1]/2.)) == 0.0:
        xpos   = np.asarray(size[1])/2.0-1.0
    else:
        xpos   = np.floor(np.asarray(size[1])/2.0)

    gauss2D = tu.shift_2Dprofile(gauss2D,[ypos,xpos],showprofiles=False,origin=0)

    if verbose: print ' - Scaling 2D gaussian by a factor '+str(scale)
    gauss2D = gauss2D*scale

    if show2Dgauss:
        savename = './Generated2Dgauss.pdf'
        if verbose: print ' - Saving resulting image of 2D gaussian to '+savename
        plt.clf()
        centerdot = gauss2D*0.0
        center    = [int(gauss2D.shape[0]/2.),int(gauss2D.shape[1]/2.)]
        centerdot[center[1],center[0]] = 2.0*np.max(gauss2D)
        print ' - Center of gaussian (pixelized - marked in plot):',center
        print ' - Center of gaussian (subpixel)                  :',[ypos,xpos]
        plt.imshow(gauss2D-centerdot,interpolation=None,origin='lower')
        plt.colorbar()
        plt.title('Generated 2D Gauss')
        plt.savefig(savename)
        plt.clf()

    if savefits:
        fitsname = './Generated2Dgauss.fits'
        hduimg   = pyfits.PrimaryHDU(gauss2D)
        hdus     = [hduimg]
        hdulist  = pyfits.HDUList(hdus)           # turn header into to hdulist
        hdulist.writeto(fitsname,clobber=True)    # write fits file (clobber=True overwrites excisting file)
        if verbose: print ' - Saved image of shifted profile to '+fitsname
    return gauss2D
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_2Dsersic(size,parameters,normalize=False,show2Dsersic=False,savefits=False,verbose=True):
    """
    Generating a 2D sersic with specified parameters using astropy's generator

    --- INPUT ---
    size          The dimensions of the array to return. Expects [ysize,xsize].
                  The 2D gauss will be positioned in the center of the array
    parameters    List of the sersic parameters.
                  Expects [amplitude,effective radius, Sersic index,ellipticity,rotation angle]
                  The amplitude is the central surface brightness within the effective radius (Ftot/2 is within r_eff)
                  The rotation angle should be in degrees, counterclockwise from the positive x-axis.
    normalize     Normalize the profile so sum(profile img) = 1.
    show2Dsersic  Save plot of generated 2D Sersic
    savefits      Save generated profile to fits file
    verbose       Toggler verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    size       = [30,40]
    size       = [31,41]
    parameters = [1,6.7,1.7,1.0-0.67,17.76-90]
    sersic2D   = tu.gen_2Dsersic(size,parameters,show2Dsersic=True,savefits=True)

    size       = [30,30]
    size       = [31,31]
    parameters = [1,5,1.7,0.5,45]
    sersic2D   = tu.gen_2Dsersic(size,parameters,show2Dsersic=True,savefits=True)

    """
    x, y  = np.meshgrid(np.arange(size[1]), np.arange(size[0]))

    if float(size[0]/2.) - float(int(size[0]/2.)) == 0.0:
        ypos   = np.asarray(size[0])/2.0-0.5
    else:
        ypos   = np.floor(np.asarray(size[0])/2.0)

    if float(size[1]/2.) - float(int(size[1]/2.)) == 0.0:
        xpos   = np.asarray(size[1])/2.0-0.5
    else:
        xpos   = np.floor(np.asarray(size[1])/2.0)

    model = Sersic2D(amplitude=parameters[0], r_eff=parameters[1], n=parameters[2], ellip=parameters[3],
                     theta=parameters[4]*np.pi/180., x_0=xpos, y_0=ypos)
    sersic2D = model(x, y)

    if normalize:
        sersic2D = sersic2D / np.sum(sersic2D)

    if show2Dsersic:
        plt.clf()
        savename = './Generated2Dsersic.pdf'
        if verbose: print ' - Displaying resulting image of 2D sersic in '+savename
        centerdot = sersic2D*0.0
        center    = [int(sersic2D.shape[0]/2.),int(sersic2D.shape[1]/2.)]
        # centerdot[center[1],center[0]] = 2.0*np.max(sersic2D)
        print ' - Center of Sersic (pixelized - marked in plot):',center
        plt.imshow(sersic2D,interpolation=None,origin='lower')
        plt.colorbar()
        plt.title('Generated 2D Sersic')
        plt.savefig(savename)
        plt.clf()

    if savefits:
        fitsname = './Generated2Dsersic.fits'
        hduimg   = pyfits.PrimaryHDU(sersic2D)
        hdus     = [hduimg]
        hdulist  = pyfits.HDUList(hdus)           # turn header into to hdulist
        hdulist.writeto(fitsname,clobber=True)    # write fits file (clobber=True overwrites excisting file)
        if verbose: print ' - Saved image of shifted profile to '+fitsname

    return sersic2D
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def get_2DsersicIeff(value,reff,sersicindex,axisratio,boxiness=0.0,returnFtot=False):
    """
    Get the surface brightness value at the effective radius of a 2D sersic profile (given GALFIT Sersic parameters).
    Ieff is calculated using ewuations (4) and (5) in Peng et al. (2010), AJ 139:2097.
    This Ieff is what is referred to as 'amplitude' in astropy.modeling.models.Sersic2D
    used in tdose_utilities.gen_2Dsersic()

    --- INPUT ---
    value         If returnFtot=False "value" corresponds to Ftot of the profile (total flux for profile integrated
                  til r=infty) and Ieff will be returned.
                  If instead returnFtot=True "value" should provide Ieff so Ftot can be returned
    reff          Effective radius
    sersicindex   Sersic index of profile
    axisratio     Ratio between the minor and major axis (0<axisratio<1)
    boxiness      The boxiness of the profile
    returnFtot    If Ftot is not known, but Ieff is, set returnFtot=True to return Ftot instead (providing Ieff to "value")

    --- EXAMPLE OF USE ---
    Ieff        = 1.0
    reff        = 25.0
    sersicindex = 4.0
    axisratio   = 1.0
    Ftot_calc   = tu.get_2DsersicIeff(Ieff,reff,sersicindex,axisratio,returnFtot=True)
    Ieff_calc   = tu.get_2DsersicIeff(Ftot_calc,reff,sersicindex,axisratio)

    size = 1000
    x,y = np.meshgrid(np.arange(size), np.arange(size))
    mod = Sersic2D(amplitude = Ieff, r_eff = reff, n=sersicindex, x_0=size/2.0, y_0=size/2.0, ellip=1-axisratio, theta=-1)
    img = mod(x, y)
    hducube  = pyfits.PrimaryHDU(img)
    hdus = [hducube]
    hdulist = pyfits.HDUList(hdus)
    hdulist.writeto('/Volumes/DATABCKUP2/TDOSEextractions/models_cutouts/model_sersic_spherical.fits',clobber=True)

    """
    gam2n  = scipy.special.gamma(2.0*sersicindex)
    kappa  = scipy.special.gammaincinv(2.0*sersicindex,0.5)
    Rfct   = np.pi * (boxiness + 2.) / (4. * scipy.special.beta(1./(boxiness+2.),1.+1./(boxiness+2.)) )
    factor = 2.0 * np.pi * reff**2.0 * np.exp(kappa) * sersicindex * kappa**(-2*sersicindex) * gam2n * axisratio / Rfct

    if returnFtot:
        Ftot = value * factor
        return Ftot
    else:
        Ieff  = value / factor
        return Ieff
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def shift_2Dprofile(profile,position,padvalue=0.0,showprofiles=False,origin=1,splineorder=3,savefits=False,verbose=True):
    """
    Shift 2D profile to given position in array by rolling it in x and y.
    Can move by sub-pixel amount using interpolation

    --- INPUT ---
    profile         profile to shift
    position        position to move center of image (profile) to:  [ypos,xpos]
    padvalue        the values to padd the images with when shifting profile
    origin          The orging of the position values. If 0-based pixels postions the
                    center calculation is updated to refelect this.
    showprofiles    Save plot of profile when shifted?
    splineorder     Order of spline interpolation to use when shifting
    savefits        Save a fitsfile of the shifted profile
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    profile = np.ones([35,35])
    profile[17,17] = 5.0
    fitsname = './Shifted2Dprofile_initial.fits'
    hduimg   = pyfits.PrimaryHDU(profile)
    hdus     = [hduimg]
    hdulist  = pyfits.HDUList(hdus)
    hdulist.writeto(fitsname,clobber=True)

    profile_shifted = tu.shift_2Dprofile(profile,[20.5,20.5],padvalue=0.0,showprofiles=False,origin=1,splineorder=3,savefits=True)

    """
    profile_dim = profile.shape
    yposition   = np.asarray(position[0])
    xposition   = np.asarray(position[1])

    if origin == 1:
        yposition = yposition - 1.0
        xposition = xposition - 1.0

    ycenter_img     = profile_dim[0]/2.-0.5 # sub-pixel center to use as reference when estimating shift
    xcenter_img     = profile_dim[1]/2.-0.5 # sub-pixel center to use as reference when estimating shift

    yshift = np.float(yposition)-ycenter_img
    xshift = np.float(xposition)-xcenter_img

    profile_shifted = scipy.ndimage.interpolation.shift(profile, [yshift,xshift], output=None, order=splineorder,
                                                        mode='constant', cval=0.0, prefilter=True)

    if showprofiles:
        plt.clf()
        savename = './Shifted2Dprofile.pdf'
        vmaxval = np.max(profile_shifted)
        plt.imshow(profile_shifted,interpolation=None,origin='lower') # ,vmin=-vmaxval, vmax=vmaxval
        plt.colorbar()
        plt.title('Positioned Source')
        plt.savefig(savename)
        plt.clf()
        if verbose: print ' - Saved image of shifted profile to '+savename

    if savefits:
        fitsname = './Shifted2Dprofile.fits'
        hduimg   = pyfits.PrimaryHDU(profile_shifted)
        hdus     = [hduimg]
        hdulist  = pyfits.HDUList(hdus)           # turn header into to hdulist
        hdulist.writeto(fitsname,clobber=True)    # write fits file (clobber=True overwrites excisting file)
        if verbose: print ' - Saved image of shifted profile to '+fitsname

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
def numerical_convolution_image(imgarray,kerneltype,saveimg=False,clobber=False,imgmask=None,fill_value=0.0,
                                norm_kernel=False,convolveFFT=False,use_scipy_conv=False,verbose=True):
    """
    Perform numerical convolution on numpy array (image)

    --- INPUT ---
    imgarray        numpy array containing image to convolve
    kerneltype      Provide either a numpy array containing the kernel or an astropy kernel
                    to use for the convolution. E.g.,
                        astropy.convolution.Moffat2DKernel()
                        astropy.convolution.Gaussian2DKernel()
    saveimg         Save image of convolved imgarray
    clobber         Overwrite existing files?
    imgmask         Mask of image array to apply during convolution
    fill_value      Fill value to use in convolution
    norm_kernel     To normalize the convolution kernel set this keyword to True
    convolveFFT     To convolve the image in fourier space set convolveFFT=True
    use_scipy_conv  Whenever the kernel and imgarray has odd dimensions, default is to use the
                    Astropy convolution where NaNs are treated with interpolation. To force a
                    scipy.ndimage convolution set use_scipy_conv=True (this is the convolution
                    used if any of the kernel (and imgarray) dimensions are even).
    verbose         Toggle verbosity

    """
    if (type(kerneltype) is np.ndarray):
        kernel    = kerneltype
        kernelstr = 'numpy array'
    else:
        kernel    = kerneltype
        kernelstr = 'astropy Guass/Moffat'

    if verbose: print ' - Convolving image with a '+kernelstr+' kernel using astropy convolution routines'


    if (np.float(kernel.shape[0]/2.0)-np.int(kernel.shape[0]/2.0) == 0) or \
       (np.float(kernel.shape[1]/2.0)-np.int(kernel.shape[1]/2.0) == 0) or use_scipy_conv:
        if verbose: print ' - Convolving using scipy.ndimage.filters.convolve() as at least one dimension of kernel is even; ' \
                          'no interpolation over NaN values'
        if norm_kernel & (np.sum(kernel) != 1.0):
            kernel = kernel/np.sum(kernel)

        # shift to sub-pixel center for even dimensions
        intpixcen = [kernel.shape[0]/2.0+0.5,kernel.shape[1]/2.0+0.5]
        kernel    = tu.shift_2Dprofile(kernel,intpixcen,showprofiles=False,origin=0)

        img_conv  = scipy.ndimage.filters.convolve(imgarray,kernel,cval=fill_value,origin=0)
    else:
        if (kernel.shape[0] < imgarray.shape[0]) or (kernel.shape[1] < imgarray.shape[1]):
            sys.exit(' ---> Astropy convolution requires kernel to have same size as image (but at least one size is smaller)')

        if (kernel.shape[0] > imgarray.shape[0]) or (kernel.shape[1] > imgarray.shape[1]):
            if verbose: print ' - Astropy convolution requires kernel to have same size as image (but it is larger); '
            if verbose: print '   Extracting center of kernel to use for convolution'
            kernel_use = tu.get_kernelcenter(imgarray.shape,kernel,useMaxAsCenter=True,verbose=False)
        else:
            kernel_use = kernel

        if convolveFFT:
            if verbose: print ' - Convolving using astropy.convolution.convolve_fft(); interpolation over NaN values'
            img_conv = convolution.convolve_fft(imgarray, kernel_use, boundary='fill',
                                                fill_value=fill_value,normalize_kernel=norm_kernel, mask=imgmask,
                                                crop=True, return_fft=False, fft_pad=None,
                                                psf_pad=None, interpolate_nan=False, quiet=False,
                                                ignore_edge_zeros=False, min_wt=0.0)
        else:
            if verbose: print ' - Convolving using astropy.convolution.convolve(); interpolation over NaN values'
            img_conv = convolution.convolve(imgarray, kernel_use, boundary='fill',
                                            fill_value=fill_value, normalize_kernel=norm_kernel, mask=imgmask)

    if saveimg:
        hdulist = pyfits.PrimaryHDU(data=img_conv)
        hdulist.writeto(saveimg,clobber=clobber)

    return img_conv
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def get_kernelcenter(shape,kernel,useMaxAsCenter=False,verbose=True):
    """
    Cutting out kernel center (with a given shape).
    Used to ensure that kernels have the right size for numerical convolution where they are required to have
    the same shape as the image to be convolved.

    NB! Assumes that the kernel is _larger_ than the image. In the other case, e.g.,
        add zeros around kernel to grow it's size

    --- INFO ---
    shape           Shape of center of kernel to cut out
    kernel          Kernel to extract central region from
    useMaxAsCenter  The default is to extract kernel around center of kjernelshape. To use the maximum value
                    of the kernel to define the extraction center set useMaxAsCenter=True
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    img     = np.ones([61,61])
    kernel  = np.ones([121,121])
    kernel[60,60] = 10.0
    kcenter = tu.get_kernelcenter(img.shape,kernel,useMaxAsCenter=True)

    img     = np.ones([40,30])
    kernel  = np.ones([190,190])
    kernel[60,60] = 10.0
    kcenter = tu.get_kernelcenter(img.shape,kernel,useMaxAsCenter=True)

    """
    if useMaxAsCenter:
        cenpix = np.where(kernel == np.max(kernel))
        if len(cenpix[0]) > 1:
            print ' WARNING: '+str(len(cenpix[0]))+' pixels with value max(Kernel). Using the first as center'
        xcen   = cenpix[1][0]
        ycen   = cenpix[0][0]
    else:
        xcen = np.floor(kernel.shape[1]/2.)
        ycen = np.floor(kernel.shape[0]/2.)

    dx   = np.floor(shape[1]/2.)
    dy   = np.floor(shape[0]/2.)

    if (np.floor(shape[0]/2.) != shape[0]/2.) & (np.floor(shape[1]/2.) != shape[1]/2.):
        kernelcen = kernel[int(ycen)-int(dy):int(ycen)+int(dy)+1, int(xcen)-int(dx):int(xcen)+int(dx)+1]
    elif (np.floor(shape[0]/2.) != shape[0]/2.) & (np.floor(shape[1]/2.) == shape[1]/2.):
        kernelcen = kernel[int(ycen)-int(dy):int(ycen)+int(dy)+1, int(xcen)-int(dx):int(xcen)+int(dx)]
    elif (np.floor(shape[0]/2.) == shape[0]/2.) & (np.floor(shape[1]/2.) != shape[1]/2.):
        kernelcen = kernel[int(ycen)-int(dy):int(ycen)+int(dy),   int(xcen)-int(dx):int(xcen)+int(dx)+1]
    elif (np.floor(shape[0]/2.) == shape[0]/2.) & (np.floor(shape[1]/2.) == shape[1]/2.):
        kernelcen = kernel[int(ycen)-int(dy):int(ycen)+int(dy),   int(xcen)-int(dx):int(xcen)+int(dx)]
    else:
        kernelcen = None

    if verbose: print ' - Input kernel shape:                     ',kernel.shape
    if verbose: print ' - Returned kernel center shape:           ',kernelcen.shape

    if verbose: print ' - Max value of input kernel:              ',np.max(kernel)
    if verbose: print ' - Max value of returned kernel center:    ',np.max(kernelcen)

    if verbose: print ' - Location of max value in input kernel:  ',np.where(kernel == np.max(kernel))
    if verbose: print ' - Location of max value in kernel center: ',np.where(kernelcen == np.max(kernelcen))

    return kernelcen
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
            ypix      = paramarray[oo*Nparam+0]
            xpix      = paramarray[oo*Nparam+1]
            skycoord  = wcs.utils.pixel_to_skycoord(xpix,ypix,wcs_in,origin=1)
            pixcoord  = wcs.utils.skycoord_to_pixel(skycoord,wcs_out,origin=1)
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
            ypix      = paramarray[oo*Nparam+0]
            xpix      = paramarray[oo*Nparam+1]
            skycoord  = wcs.utils.pixel_to_skycoord(xpix,ypix,wcs_in,origin=1)
            pixcoord  = wcs.utils.skycoord_to_pixel(skycoord,wcs_out,origin=1)
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
    skyc      = SkyCoord(ra, dec, frame='fk5', unit=(units.deg,units.deg))
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
    skyc     = SkyCoord(ra, dec, frame='fk5', unit=(units.deg,units.deg))
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
        skycoord  = wcs.utils.pixel_to_skycoord(xpix,ypix,wcsinfo,origin=1)
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
                                      idcol=0,racol=2,deccol=3,fluxcol=22,fluxfactor=100.,generateDS9reg=True,
                                      verbose=True):
    """
    Generate a txt (and fits) source catalog for modeling images with tdose_model_FoV.gen_fullmodel()

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
    generateDS9reg      Generate a DS9 region file showing the location of the sources?
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
        striphdr   = tu.strip_header(imgheader.copy(),verbose=verbose)
        wcs_in     = wcs.WCS(striphdr)
        skycoord   = SkyCoord(ras, decs, frame='fk5', unit='deg')
        pixcoord   = wcs.utils.skycoord_to_pixel(skycoord,wcs_in,origin=1)
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

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if generateDS9reg:
            regionfile = outname.replace('.txt','.reg')
            if verbose: print ' - Storing DS9 region file to '+regionfile
            idsstr     = [str(id) for id in ids]
            tu.create_simpleDS9region(regionfile,ras,decs,color='red',circlesize=0.5,textlist=idsstr,clobber=clobber)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        outnamefits = outname.replace('.txt','.fits')
        if verbose: print ' - Save fits version of source catalog to '+outnamefits
        fitsfmt       = ['D','D','D','D','D','D']
        sourcecatfits = tu.ascii2fits(outname,asciinames=True,skip_header=2,fitsformat=fitsfmt,verbose=verbose)

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
            skycoord   = SkyCoord(sourcedat[racol][oo], sourcedat[deccol][oo], frame='fk5', unit='deg')
            pixcoord   = wcs.utils.skycoord_to_pixel(skycoord,wcs_in,origin=1)
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
def ascii2fits(asciifile,asciinames=True,skip_header=0,outpath=None,fitsformat='D',verbose=True):
    """
    Convert ascii file into fits file

    --- INPUT ---
    asciifile        Ascii file to convert
    asciinames       Do the ascii file contain the column names in the header?
    skip_header      The number of header lines to skip when reading the ascii file.
    outpath          Alternative destination for the resulting fits file.
    fitsformat       Format of the entries to store in the fits file. Default assumes 'D' for all
                     Otherwise, provide list of fits formats
    verbose          Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    outpath = '/Users/johndoe/tdose_sourcecats/'
    catfile = 'sources.cat'
    outputfile = tu.ascii2fits(catfile,asciinames=True,skip_header=2,outpath=outpath,verbose=True)

    """
    #-------------------------------------------------------------------------------------------------------------
    if verbose: print ' - Reading ascii file ',asciifile
    data    = np.genfromtxt(asciifile,names=asciinames,skip_header=skip_header,comments='#',dtype=None)
    keys    = data.dtype.names
    #-------------------------------------------------------------------------------------------------------------
    if verbose: print ' - Initialize and fill dictionary with data'
    datadic = {}
    for kk in keys:
        datadic[kk] = []
        try:
            lenarr = len(np.asarray(data[kk]))
            datadic[kk] = np.asarray(data[kk])
        except: # if only one row of data is to be written
            datadic[kk] = np.asarray([data[kk]])

    if verbose: print ' - found the columns '+','.join(keys)

    if len(fitsformat) != len(keys):
        fitsformat = np.asarray([fitsformat]*len(keys))
    #-------------------------------------------------------------------------------------------------------------
    # writing to fits table
    tail = asciifile.split('.')[-1]# reomove extension
    outputfile = asciifile.replace('.'+tail,'.fits')
    if outpath != None:
        outputfile = outpath+outputfile.split('/')[-1]

    columndefs = []
    for kk, key in enumerate(keys):
        try:
            columndefs.append(pyfits.Column(name=key  , format=fitsformat[kk], array=datadic[key]))
        except:
            sys.exit(' ---> ERROR in defining columns for fits file '+outputfile+' in call to tdose_utilities.ascii2fits()')

    cols     = pyfits.ColDefs(columndefs)
    tbhdu    = pyfits.new_table(cols)          # creating table header
    hdu      = pyfits.PrimaryHDU()             # creating primary (minimal) header
    thdulist = pyfits.HDUList([hdu, tbhdu])    # combine primary and table header to hdulist
    thdulist.writeto(outputfile,clobber=True)  # write fits file (clobber=True overwrites excisting file)
    #-------------------------------------------------------------------------------------------------------------
    if verbose: print ' - Wrote the data to: ',outputfile
    return outputfile
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def SExtractorCat2fits(sextractorfiles,stringcols=[1],header=73,verbose=True):
    """
    Converting an ascii catalog with columns defined in header in the SExtractor format, i.e. one column
    name per row preceeded by a "#" and a column numner, and followed by a description (or any ascii file
    with the given setup) to a fits binary table

    --- INPUT ---
    sextractorfiles   List of ascii files to convert to fits
    stringcols        Columns to use a string format for (all other columns will be set to double float)
    header            Header containing the column names of the catalogs following the "SExtractor notation"
    verbose           Toggle verbosity

    --- EXAMPLE OF USE ---
    import glob
    import tdose_utilities as tu
    catalogs = glob.glob('/Volumes/DATABCKUP2/MUSE-Wide/catalogs_photometry/catalog_photometry_candels-cdfs-*.cat')
    tu.SExtractorCat2fits(catalogs,stringcols=[1],header=73,verbose=True)

    """
    for sexcat_ascii in sextractorfiles:
        asciiinfo = open(sexcat_ascii,'r')
        photcols = []
        for line in asciiinfo:
            if line.startswith('#'):
                colname = line.split()[2]
                photcols.append(colname)

        photfmt = ['D']*len(photcols)
        for stringcol in stringcols:
            photfmt[stringcol] = 'A60'

        sexcat_fits   = tu.ascii2fits(sexcat_ascii,asciinames=photcols,skip_header=header,fitsformat=photfmt,verbose=verbose)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def create_simpleDS9region(outputfile,ralist,declist,color='red',circlesize=0.5,textlist=None,clobber=False):
    """
    Generate a basic DS9 region file with circles around a list of coordinates

    --- INPUT ---
    outputfile   Path and name of file to store reigion file to
    ralist       List of R.A. to position circles at
    declist      List of Dec. to position circles at
    color        Color of circles
    size         Size of circles (radius in arcsec)
    text         Text string for each circle
    clobber      Overwrite existing files?

    """

    if not clobber:
        if os.path.isfile(outputfile):
            sys.exit('File already exists and clobber = False --> ABORTING')
    fout = open(outputfile,'w')

    fout.write("# Region file format: DS9 version 4.1 \nfk5\n")

    for rr, ra in enumerate(ralist):
        string = 'circle('+str(ra)+','+str(declist[rr])+','+str(circlesize)+'") # color='+color+' width=3 '

        if textlist is not None:
            string = string+' font="times 10 bold roman" text={'+textlist[rr]+'}'

        fout.write(string+' \n')

    fout.close()
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
def galfit_convertmodel2cube(galfitmodelfiles,includewcs=True,savecubesumimg=False,convkernels=None,sourcecat_compinfo=None,
                             normalizecomponents=False,includesky=False,clobber=True,verbose=True):
    """
    Convert a GALFIT model output file into a model cube, where each model component occupy a different layer in the cube.

    NB! The absolute flux scales of the indvidual Sersic components are off.
        To accommodate this, set normalizecomponents=True and manually scale
        either the indiviodual components or the 'cubesum' image.
        TDOSE works with normalized components so this offsets does not affect the spectral extractions.

    --- INPUT ---
    galfitmodelfiles    Output from running GALFIT
    includewcs          Include WCS information in header? If False no WCS information is included, if True the WCS
                        information from the reference image extension in the GALFIT model (extension 1) is used.
    savecubesumimg      Save image of of sum over cube components (useful for comparison with input GALFIT models)
    convkernels         List of numpy arrays or astropy kernels to use for convolution. This can be used to apply a PSF
                        to the model re-generated from the GALFIT parameters. This is useful as GALFIT is modeling the
                        component paramters before PSF convolution
    sourcecat_compinfo  A source catalog with parent IDs, i.e. IDs assigning individual model components to the main
                        object and to contaminants, is needed for TDOSE to succesfully disentangle and deblend sources
                        based on the model cube. To save a TDOSE-like source catalog with parent ids, provide a file
                        containing "compoent info" to the sourcecat_compinfo keyword. A piece of compoent info is a string
                        including the model name followed by an object ID and 'X:Y' strings indicating what each component
                        (X; starting from 1 corresponding to the COMP_X GALFIT keyword) corresponds to. In the latter
                        Y indicates whether a model component is
                           Y=1     a source in the main object
                           Y=2     a contaminating source
                           Y=3     a sky model
                        Hence, the following "component info" line indicates a model with components 1 and 3 being
                        sources in the object of interest (ID=12345), contaminating sources modeled by components 2 and 4,
                        and a sky model in the 5th model component:

                             name_of_GALFIT_model_for_ID12345.fits  12345  1:1  2:2  3:1  4:2  5:3

    includesky          To include sky-components from the GALFIT header when building the component cube and putting
                        together the source catalog, set includesky=True
    normalizecomponents Normalize each individual components so sum(component image) = 1?
                        TDOSE will normalize cube for extraction optimization irrespective of input
    clobber             Overwrite existing files
    verbose             Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    import pyfits

    fileG   = '/Volumes/DATABCKUP2/TDOSEextractions/models_cutouts/model8685multicomponent/model_acs_814w_candels-cdfs-02_cut_v1.0_id8685_cutout7p0x7p0arcsec.fits' # Gauss components
    fileS   = '/Volumes/DATABCKUP2/TDOSEextractions/models_cutouts/model9262GALFITdoublecomponent/model_acs_814w_candels-cdfs-02_cut_v1.0_id9262_cutout2p0x2p0arcsec.fits' # Sersic components
    fileS   = '/Volumes/DATABCKUP2/TDOSEextractions/models_cutouts/model9262GALFITsinglecomponent/model_acs_814w_candels-cdfs-02_cut_v1.0_id9262_cutout2p0x2p0arcsec.fits' # Sersic components

    PSFmodel = pyfits.open('/Volumes/DATABCKUP2/TDOSEextractions/models_cutouts/F814Wpsfmodel_imgblock_6475.fits')[2].data

    tu.galfit_convertmodel2cube([fileS,fileG],savecubesumimg=True,includewcs=True,convkernels=[PSFmodel,None],normalizecomponents=False)

    tu.test_sersicprofiles(Ieff=0.0184304803665,reff=1.72,sersicindex=1.0,axisratio=0.3,size=67,angle=-177.98)

    """
    if verbose: print ' - Will convert the '+str(len(galfitmodelfiles))+' GALFIT models into cubes '
    if verbose: print '\n'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if sourcecat_compinfo is not None:
        compinfo = {}
        with open(sourcecat_compinfo) as scci:
            for componentinfo in scci:
                if componentinfo.startswith('#'):
                    pass
                else:
                    cisplit = componentinfo.split()
                    compinfo[cisplit[0]] = {}
                    compinfo[cisplit[0]]['id'] = cisplit[1]
                    for cinfo in cisplit[2:]:
                        if ':' in cinfo:
                            cno   = cinfo.split(':')[0]
                            cinfo = cinfo.split(':')[1]
                            compinfo[cisplit[0]]['COMP_'+cno] = cinfo
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for gg, galfitmodel in enumerate(galfitmodelfiles):
        if verbose: print ' - Extracting number of components from model extenstion (ext=2) of GALFIT model:\n   '+galfitmodel
        headerinfo = pyfits.open(galfitmodel)[2].header
        modelarr   = pyfits.open(galfitmodel)[2].data

        if sourcecat_compinfo is not None:
            sourcecat = galfitmodel.replace('.fits','_sourcecatalog')+'.txt'
            if verbose: print ' - Component info provided so opening source catalog file:\n   '+sourcecat
            scat = open(sourcecat,'w')
            scat.write('# Source catalog for '+galfitmodel+'\n')
            scat.write('# Based on component info in '+sourcecat_compinfo+'\n')
            scat.write('# Generated with tdose_utilities.galfit_convertmodel2cube() on '+tu.get_now_string()+'\n# \n')
            scat.write('# parent_id id ra dec x_image y_image fluxscale \n')

            refimg_hdr = pyfits.open(galfitmodel)[1].header
            modelwcs   = wcs.WCS(tu.strip_header(refimg_hdr.copy()))

            cutrange_low_x = int(float(refimg_hdr['OBJECT'].split(':')[0].split('[')[-1]))
            cutrange_low_y = int(float(refimg_hdr['OBJECT'].split(',')[-1].split(':')[0]))

            xpix_mod, ypix_mod, ra_mod, dec_mod = tu.galfit_getcentralcoordinate(galfitmodel,coordorigin=1)
            objdic     = compinfo[galfitmodel.split('/')[-1]]
            objid      = objdic['id']
            modstring  = ' -99 '+objid+' '+str(ra_mod)+' '+str(dec_mod)+' '+\
                         str(xpix_mod-cutrange_low_x+1)+' '+str(ypix_mod-cutrange_low_y+1)+' 1.0'

        compkeys = []
        for key in headerinfo.keys():
            if 'COMP_' in key:
                if headerinfo[key] == 'sky':
                    if includesky:
                        compkeys.append(key)
                    else:
                        pass
                else:
                    compkeys.append(key)

        Ncomp = len(compkeys)

        if Ncomp == 1:
            if verbose: print ' - Only found one component in header so not turning GALFIT model into cube'
        elif Ncomp == 0:
            sys.exit(' ---> Did not find _any_ components in the GALFIT model header')
        else:
            if verbose: print ' - Found '+str(Ncomp)+' components in GALFIT model header to populate cube with'

        cube = np.zeros([Ncomp,modelarr.shape[0],modelarr.shape[1]])
        for cc,component in enumerate(compkeys):
            compnumber    = str(cc+1)

            if headerinfo[component] == 'gaussian':
                xc, xcerr     = tu.galfit_getheadervalue(compnumber,'XC',headerinfo)
                yc, ycerr     = tu.galfit_getheadervalue(compnumber,'YC',headerinfo)
                mag, magerr   = tu.galfit_getheadervalue(compnumber,'MAG',headerinfo)
                ar, arerr     = tu.galfit_getheadervalue(compnumber,'AR',headerinfo)
                pa, paerr     = tu.galfit_getheadervalue(compnumber,'PA',headerinfo)
                magzeropoint  = float(headerinfo['MAGZPT'])
                fluxtot       = 10.0**((mag-magzeropoint)/(-2.5))

                fwhm, fwhmerr = tu.galfit_getheadervalue(compnumber,'FWHM',headerinfo)
                sigma2fwhm    = 2.0*np.sqrt(2.0*np.log(2.0)) # ~ 2.355
                sigmax        = fwhm/sigma2fwhm
                sigmay        = fwhm/sigma2fwhm*ar

                covmatrix     = tu.build_2D_cov_matrix(sigmax,sigmay,pa+90.0,verbose=verbose)
                gauss2Dimg    = tu.gen_2Dgauss(modelarr.shape,covmatrix,fluxtot,show2Dgauss=False,verbose=verbose,method='scipy')
                img_shift     = tu.shift_2Dprofile(gauss2Dimg,[yc,xc],padvalue=0.0,showprofiles=False,origin=1)
                cubelayer     = img_shift

            elif headerinfo[component] == 'sersic':
                xc, xcerr     = tu.galfit_getheadervalue(compnumber,'XC',headerinfo)
                yc, ycerr     = tu.galfit_getheadervalue(compnumber,'YC',headerinfo)
                mag, magerr   = tu.galfit_getheadervalue(compnumber,'MAG',headerinfo)
                ar, arerr     = tu.galfit_getheadervalue(compnumber,'AR',headerinfo)
                pa, paerr     = tu.galfit_getheadervalue(compnumber,'PA',headerinfo)
                magzeropoint  = float(headerinfo['MAGZPT'])
                fluxtot       = 10.0**((mag-magzeropoint)/(-2.5))

                Re, Reerr           = tu.galfit_getheadervalue(compnumber,'RE',headerinfo)
                nsersic, nsersicerr = tu.galfit_getheadervalue(compnumber,'N',headerinfo)

                Ieff        = tu.get_2DsersicIeff(fluxtot,Re,nsersic,ar,boxiness=0.0,returnFtot=False)
                if verbose: print ' -',component,': Ieff=',Ieff,' calculated using Ftot = ',fluxtot,'Reff=',Re,'pix','n=',nsersic,\
                    'axisratio=',ar

                ellipticity = 1.0-ar
                parameters  = [Ieff,Re,nsersic,ellipticity,pa-90]
                if verbose: print ' -',component,': 2D sersic parameters [Ieff,Reff,Sersic index,ellipticity,angle] =',parameters

                if verbose: print ' - central coordinates are',[yc,xc]
                sersic2Dimg = tu.gen_2Dsersic(modelarr.shape,parameters,show2Dsersic=False,normalize=False)
                img_shift   = tu.shift_2Dprofile(sersic2Dimg,[yc,xc],padvalue=0.0,showprofiles=False,origin=1,splineorder=3)
                cubelayer   = img_shift
            elif headerinfo[component] == 'sky':
                xc, xcerr   = tu.galfit_getheadervalue(compnumber,'XC',headerinfo)
                yc, ycerr   = tu.galfit_getheadervalue(compnumber,'YC',headerinfo)
                sky, skyerr = tu.galfit_getheadervalue(compnumber,'SKY',headerinfo)
                fluxtot     = sky
                cubelayer   = np.zeros([modelarr.shape[0],modelarr.shape[1]])+sky
            else:
                sys.exit(' ---> Dealing with a "'+headerinfo[component]+'" GALFIT model component is not implemented yet; sorry. '
                                                                       'Try using "gaussian" or "sersic" components to build your model')

            if convkernels is not None:
                convkernel = convkernels[gg]
                if convkernel is None:
                    if verbose: print ' - Not convoling model with kernel as kernel in "convkernels" list for model '+str(gg)+' was None'
                else:
                    if verbose: print ' - Convoling model with kernel provided'
                    cubelayer = tu.numerical_convolution_image(cubelayer,convkernel,saveimg=False,clobber=clobber,imgmask=None,
                                                               fill_value=0.0,norm_kernel=True,convolveFFT=False,
                                                               use_scipy_conv=False,verbose=verbose)

            if normalizecomponents:
                cubelayer = cubelayer / np.sum(cubelayer)

            cube[cc,:,:] = cubelayer
            cubelayer    = cubelayer*0.0 # resetting cube layer

            if sourcecat_compinfo is not None:
                cval   = objdic[component]
                if cval == '1': # Object component
                    id_parent = objid
                    id_source = objid+str("%.3d" % int(compnumber))
                elif cval == '2': # Contaminant
                    id_parent = objid+str("%.3d" % int(compnumber))#+'002'
                    id_source = objid+str("%.3d" % int(compnumber))#+'002'
                elif cval == '3': # Sky
                    id_parent = objid+str("%.3d" % int(compnumber))#+'003'
                    id_source = objid+str("%.3d" % int(compnumber))#+'003'
                else:
                    id_parent = 'WARNING - invalud value of component info. Component info file provides "'+str(cval)+'" for '+component
                    id_source = ' '

                skycoord   = wcs.utils.pixel_to_skycoord(cutrange_low_x+xc-1,cutrange_low_y+yc-1,modelwcs,origin=1)
                racomp     = skycoord.ra.value
                deccomp    = skycoord.dec.value
                compstring = ' '+id_parent+' '+id_source+' '+str(racomp)+' '+str(deccomp)+' '+str(xc)+' '+str(yc)+' '+str(fluxtot)
                scat.write(compstring+'\n')

        if sourcecat_compinfo is not None:
            scat.write(modstring+'\n') # NB - The string describing the model center has to be put after the component
                                       #      strings to ensure a proper assignment of layers to components when
                                       #      extracting 1D spectra with TDOSE.
            scat.close()
            fitscat = sourcecat.replace('.txt','.fits')
            if verbose: print ' - Save fits version of source catalog to '+fitscat
            fitsfmt       = ['D','D','D','D','D','D','D']
            sourcecatfits = tu.ascii2fits(sourcecat,asciinames=True,skip_header=4,fitsformat=fitsfmt,verbose=verbose)


        # - - - - - - - - - - - - - - - - - - Saving Model Cube - - - - - - - - - - - - - - - - - -
        cubename = galfitmodel.replace('.fits','_cube')+'.fits'
        if verbose: print ' - Saving model cube to \n   '+cubename
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        hducube  = pyfits.PrimaryHDU(cube)       # default HDU with default minimal header

        if includewcs:
            if verbose: print ' - Including WCS information from GALFIT reference image extension   '
            wcsheader = pyfits.open(galfitmodel)[1].header
            # writing hdrkeys:    '---KEY--',                       '----------------MAX LENGTH COMMENT-------------'
            hducube.header.append(('BUNIT  '                      ,'(Ftot/texp)'),end=True)
            hducube.header.append(('CRPIX1 ',wcsheader['CRPIX1']  ,' Pixel coordinate of reference point'),end=True)
            hducube.header.append(('CRPIX2 ',wcsheader['CRPIX2']  ,' Pixel coordinate of reference point'),end=True)
            hducube.header.append(('CD1_1  ',wcsheader['CD1_1 ']  ,' Coordinate transformation matrix element'),end=True)
            hducube.header.append(('CD1_2  ',wcsheader['CD1_2 ']  ,' Coordinate transformation matrix element'),end=True)
            hducube.header.append(('CD2_1  ',wcsheader['CD2_1 ']  ,' Coordinate transformation matrix element'),end=True)
            hducube.header.append(('CD2_2  ',wcsheader['CD2_2 ']  ,' Coordinate transformation matrix element'),end=True)
            hducube.header.append(('CTYPE1 ',wcsheader['CTYPE1']  ,' Right ascension, gnomonic projection'),end=True)
            hducube.header.append(('CTYPE2 ',wcsheader['CTYPE2']  ,' Declination, gnomonic projection'),end=True)
            hducube.header.append(('CRVAL1 ',wcsheader['CRVAL1']  ,' '),end=True)
            hducube.header.append(('CRVAL2 ',wcsheader['CRVAL2']  ,' '),end=True)
            try:
                hducube.header.append(('CSYER1 ',wcsheader['CSYER1']  ,' [deg] Systematic error in coordinate'),end=True)
                hducube.header.append(('CSYER2 ',wcsheader['CSYER2']  ,' [deg] Systematic error in coordinate'),end=True)
                hducube.header.append(('CUNIT1 ',wcsheader['CUNIT1']  ,' Units of coordinate increment and value'),end=True)
                hducube.header.append(('CUNIT2 ',wcsheader['CUNIT2']  ,' Units of coordinate increment and value'),end=True)
            except:
                pass

            hducube.header.append(('CTYPE3 ','COMPONENT    '      ,' '),end=True)
            hducube.header.append(('CUNIT3 ',''                   ,' '),end=True)
            hducube.header.append(('CD3_3  ',                  1. ,' '),end=True)
            hducube.header.append(('CRPIX3 ',                  1. ,' '),end=True)
            hducube.header.append(('CRVAL3 ',                  1. ,' '),end=True)
            hducube.header.append(('CD1_3  ',                  0. ,' '),end=True)
            hducube.header.append(('CD2_3  ',                  0. ,' '),end=True)
            hducube.header.append(('CD3_1  ',                  0. ,' '),end=True)
            hducube.header.append(('CD3_2  ',                  0. ,' '),end=True)

        hdus = [hducube]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        hdulist = pyfits.HDUList(hdus)             # turn header into to hdulist
        hdulist.writeto(cubename,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)

        # - - - - - - - - - - - - - - - - - - Saving Sum Cube Image - - - - - - - - - - - - - - - - - -
        if savecubesumimg:
            imgname = galfitmodel.replace('.fits','_cubesum.fits')
            if verbose: print ' - Saving model cube to \n   '+imgname
            cubesum = np.sum(cube,axis=0)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # KBS170915 Manually scaling to max value of model for testing:
            scaletomaxmodelpix = False
            if scaletomaxmodelpix:
                scalfactor = np.max(modelarr)/np.max(cubesum)
                print ' -------> Scaled cubesum output by a factor ',scalfactor,' [ max(model)/max(cubesum) ]'
                print '          max(model)   = ',np.max(modelarr)
                print '          max(cubesum) = ',np.max(cubesum)
                cubesum    = cubesum * scalfactor
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            hduimg  = pyfits.PrimaryHDU(cubesum)

            if includewcs:
                if verbose: print ' - Including WCS information from GALFIT reference image extension   '
                wcsheader = pyfits.open(galfitmodel)[1].header
                # writing hdrkeys:    '---KEY--',                       '----------------MAX LENGTH COMMENT-------------'
                hduimg.header.append(('BUNIT  '                      ,'(Ftot/texp)'),end=True)
                hduimg.header.append(('CRPIX1 ',wcsheader['CRPIX1']  ,' Pixel coordinate of reference point'),end=True)
                hduimg.header.append(('CRPIX2 ',wcsheader['CRPIX2']  ,' Pixel coordinate of reference point'),end=True)
                hduimg.header.append(('CD1_1  ',wcsheader['CD1_1 ']  ,' Coordinate transformation matrix element'),end=True)
                hduimg.header.append(('CD1_2  ',wcsheader['CD1_2 ']  ,' Coordinate transformation matrix element'),end=True)
                hduimg.header.append(('CD2_1  ',wcsheader['CD2_1 ']  ,' Coordinate transformation matrix element'),end=True)
                hduimg.header.append(('CD2_2  ',wcsheader['CD2_2 ']  ,' Coordinate transformation matrix element'),end=True)
                hduimg.header.append(('CTYPE1 ',wcsheader['CTYPE1']  ,' Right ascension, gnomonic projection'),end=True)
                hduimg.header.append(('CTYPE2 ',wcsheader['CTYPE2']  ,' Declination, gnomonic projection'),end=True)
                hduimg.header.append(('CRVAL1 ',wcsheader['CRVAL1']  ,' '),end=True)
                hduimg.header.append(('CRVAL2 ',wcsheader['CRVAL2']  ,' '),end=True)
                try:
                    hduimg.header.append(('CSYER1 ',wcsheader['CSYER1']  ,' [deg] Systematic error in coordinate'),end=True)
                    hduimg.header.append(('CSYER2 ',wcsheader['CSYER2']  ,' [deg] Systematic error in coordinate'),end=True)
                    hduimg.header.append(('CUNIT1 ',wcsheader['CUNIT1']  ,' Units of coordinate increment and value'),end=True)
                    hduimg.header.append(('CUNIT2 ',wcsheader['CUNIT2']  ,' Units of coordinate increment and value'),end=True)
                except:
                    pass

                hdus = [hduimg]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            hdulist = pyfits.HDUList(hdus)            # turn header into to hdulist
            hdulist.writeto(imgname,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print '\n'
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def galfit_getheadervalue(compnumber,key,headerinfo):
    """
    Return the paramters of a GALFIT model header

    --- INPUT ---
    compnumber      A string containing the component number to extract info for (number after "COMP_" in header)
    key             The key to extract (keyword after "COMPNUMBER_" in header)
    headerinfo      Header to extract info from.

    """
    hdrinfo = headerinfo[compnumber+'_'+key]

    if '+/-' in hdrinfo:
        value   = float(hdrinfo.split('+/-')[0])
        error   = float(hdrinfo.split('+/-')[1])
    else:
        value   = float(hdrinfo[1:-1])
        error   = None

    if (key == 'XC') or (key == 'YC'):
        xrange, yrange = headerinfo['FITSECT'][1:-1].split(',')
        xrange = np.asarray(xrange.split(':')).astype(float)
        yrange = np.asarray(yrange.split(':')).astype(float)
        if key == 'XC':
            value = value - xrange[0] + 1.0
        if key == 'YC':
            value = value - yrange[0] + 1.0

    return value, error
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def galfit_getcentralcoordinate(modelfile,coordorigin=1,verbose=True):
    """
    Return the central coordinates of a GALFIT model extracted using the reference image WCS and the FITSECT keyword

    --- INPUT ---
    modelfile       Path and name to GALFIT model fits file to retrieve central coordinates for
    coordorigin     Origin of coordinates in reference image to use when converting pixels to degrees (skycoord)
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    fileG   = '/Volumes/DATABCKUP2/TDOSEextractions/models_cutouts/model8685multicomponent/model_acs_814w_candels-cdfs-02_cut_v1.0_id8685_cutout7p0x7p0arcsec.fits' # Gauss components
    fileS   = '/Volumes/DATABCKUP2/TDOSEextractions/models_cutouts/model8685multicomponent/model_acs_814w_candels-cdfs-02_cut_v1.0_id9262_cutout2p0x2p0arcsec.fits' # Sersic components

    xpix, ypix, ra_model, dec_model = tu.galfit_getcentralcoordinate(fileG,coordorigin=1)

    """
    if verbose: print ' - Will extract central coordinates from '+modelfile
    refimg_hdr     = pyfits.open(modelfile)[1].header
    model_hdr      = pyfits.open(modelfile)[2].header
    imgwcs         = wcs.WCS(tu.strip_header(refimg_hdr.copy()))

    fit_region     = model_hdr['FITSECT']
    cutrange_low_x = int(float(fit_region.split(':')[0].split('[')[-1]))
    cutrange_low_y = int(float(fit_region.split(',')[-1].split(':')[0]))
    xsize          = model_hdr['NAXIS1']
    ysize          = model_hdr['NAXIS2']

    xpix           = cutrange_low_x + int(xsize/2.)
    ypix           = cutrange_low_y + int(ysize/2.)

    if verbose: print ' - Converting pixel position to coordinates using a pixel origin='+str(coordorigin)
    skycoord    = wcs.utils.pixel_to_skycoord(xpix,ypix,imgwcs,origin=coordorigin)

    ra_model    = skycoord.ra.value
    dec_model   = skycoord.dec.value

    return xpix,ypix,ra_model,dec_model
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
def gen_overview_plot(objids,setupfile,skipobj=False,outputdir='spec1D_directory',verbose=True):
    """
    Generating summary/overview plot of modeled objects (cutouts) and the corresponding 1D spectra

    --- INPUT ---
    objids         IDs of objects to generate plots for
    setupfile      The setup file used when running TDOSE (defines the location of the products to plot)
    skipobj        Skip objects where an overview plot already exists at the output location
    outputdir      The directory to save the overview plots to. If outputdir = 'spec1D_directory'
                   the plots will be saved to the 'spec1D_directory' specified in the TDOSE setupfile
                   provided.
    verbose        Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    objids    = ['8685']#,'0000008685-0000008685','10195','0010207068-0000010195']
    setupfile = '/Volumes/DATABCKUP2/TDOSEextractions/tdose_spectra/tdose_setup_candels-cdfs-02_logged.txt'
    tu.gen_overview_plot(objids,setupfile)

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Loading setup file to get info on directories to search for TDOSE products'
    setupdic  =  tu.load_setup(setupfile,verbose=verbose)

    if outputdir == 'spec1D_directory':
        outdir = setupdic['spec1D_directory']
    else:
        outdir = outputdir

    if verbose: print ' - Get main source id for cases where multiple sources were combined using parent ID (ids contain "-")'
    baseids = []

    if objids == 'all':

        if verbose: print ' - Objid = "all" so grabbing IDs from file names in spec1D_directory from setupfile'
        specdir  = setupdic['spec1D_directory']

        spectra = glob.glob(specdir+setupdic['spec1D_name']+'_'+setupdic['psf_type']+'*.fits')
        objids = []
        for specfile in spectra:
            objid_fromfile = specfile.split(specdir+setupdic['spec1D_name']+'_'+setupdic['source_model']+'_')[-1].split('.fit')[0]
            objids.append(objid_fromfile)
        objids = np.unique(np.sort(np.asarray(objids)))

    for objid in objids:
        baseids.append(objid.split('-')[-1])

    Nobj = len(objids)
    if verbose: print ' - Will generate summary plot for the '+str(Nobj)+' objects provided IDs for'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Looping over objects and generating plots'
    for ii, objid in enumerate(objids):
        if '-' in objid:
            objidstr    = objid
        else:
            objidstr    = str("%.10d" % int(objid))
        baseid      = baseids[ii]
        plotname    = outdir+setupdic['spec1D_name']+'_'+setupdic['source_model']+'_'+objidstr+'_source_overview.pdf'

        if skipobj & os.path.isfile(plotname):
            skipthisobj = True
        else:
            skipthisobj = False

        if verbose:
            infostr = ' - Generate plot for '+objidstr+' ('+str(ii+1)+'/'+str(Nobj)+') '
            if skipthisobj:
                infostr = infostr+'... plot exists --> skipobj'
            else:
                infostr = infostr+'                           '
            sys.stdout.write("%s\r" % infostr)
            sys.stdout.flush()

        if skipthisobj: continue
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # look for data to be plotted for object
        cutstr, cutoutsize, cut_img, cut_cube, cut_variance, cut_sourcecat = tu.get_datinfo(int(baseid),setupdic)
        sourcemodel  = setupdic['source_model']

        imagebase    = setupdic['ref_image'].split('/')[-1]
        refimg       = setupdic['cutout_directory']+imagebase.replace('.fits',cutstr+'.fits')
        modelimg     = setupdic['models_directory']+imagebase.replace('.fits',cutstr+'_'+setupdic['model_image_ext']+
                                                                      '_'+sourcemodel+'.fits')
        residualimg  = modelimg.replace('.fits','_residual.fits')

        cubebase     = setupdic['data_cube'].split('/')[-1]
        datacube     = setupdic['cutout_directory']+cubebase.replace('.fits',cutstr+'.fits')
        modelcube    = setupdic['models_directory']+cubebase.replace('.fits',cutstr+'_'+setupdic['model_cube_ext']+
                                                                     '_'+sourcemodel+'.fits')
        residualcube = setupdic['models_directory']+cubebase.replace('.fits',cutstr+'_'+setupdic['residual_cube_ext']+
                                                                     '_'+sourcemodel+'.fits')

        spec1D       = setupdic['spec1D_directory']+setupdic['spec1D_name']+'_'+sourcemodel+'_'+objidstr+'.fits'

        dwave        = 150
        wavefix      = 5500

        if os.path.isfile(spec1D):
            specdat = pyfits.open(spec1D)[1].data
            maxs2n  = np.max(specdat['s2n'][ np.isfinite(specdat['s2n']) ])
            maxent  = np.where(specdat['s2n'] == maxs2n)[0][0]
            maxwave = specdat['wave'][maxent]
            maxflux = specdat['flux'][maxent]
            diff    = np.abs(specdat['wave']-wavefix)
            entfix  = np.where(diff == np.min(diff))[0][0]
        else:
            maxent  = 1
            maxwave = 6000.0
            entfix  = 100

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Ncol         = 6
        Nrow         = 7
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        fig = plt.figure(figsize=(Ncol, Nrow*1.2))
        fig.subplots_adjust(wspace=0.75, hspace=0.6,left=0.085, right=0.99, bottom=0.05, top=0.96)
        Fsize  = 6
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()
        #plt.title(plotname.split('/')[-1].replace('_','\_'),fontsize=Fsize)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        #---------------------------- REF IMG ----------------------------
        rowval  = 0
        rowspan = 2
        colval  = 0
        colspan = 2

        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan, rowspan=rowspan)

        tu.gen_overview_plot_image(ax,refimg,fontsize=Fsize,lthick=lthick,alpha=0.5,title='Reerence Image')

        #---------------------------- REF IMG MOD  ----------------------------
        rowval  = 0
        rowspan = 2
        colval  = 2
        colspan = 2
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan, rowspan=rowspan)

        tu.gen_overview_plot_image(ax,modelimg,fontsize=Fsize,lthick=lthick,alpha=0.5,title='Reerence Image Model')

        #---------------------------- REF IMG RES  ----------------------------
        rowval  = 0
        rowspan = 2
        colval  = 4
        colspan = 2
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan, rowspan=rowspan)

        tu.gen_overview_plot_image(ax,residualimg,fontsize=Fsize,lthick=lthick,alpha=0.5,title='Imag - Model Residual')

        #---------------------------- REF IMG HIST ----------------------------
        rowval  = 2
        colval  = 0
        colspan = 6
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_hist(ax,refimg,modelimg,dataext=0,modelext=0,layer=None,
                                  histbins=[-0.01,0.01,0.0001],lthick=lthick,alpha=0.5,title='Reference image',fontsize=Fsize)

        #---------------------------- REF IMG HIST 2 ----------------------------
        # rowval  = 2
        # colval  = 3
        # colspan = 3
        # ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)
        #
        # tu.gen_overview_plot_hist(ax,'dummy','dummy',dataext=0,modelext=0,layer=None,
        #                           histbins=[-0.01,0.01,0.0001],lthick=lthick,alpha=0.5,title='Hist Title?',fontsize=Fsize)


        #---------------------------- FMAX CUBE ----------------------------
        rowval  = 3
        colval  = 0
        colspan = 1
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_image(ax,datacube,imgext=setupdic['cube_extension'],cubelayer=maxent,fontsize=Fsize,lthick=lthick,alpha=0.5,
                                   title='Max S/N layer')

        #---------------------------- FMAX CUBE MOD ----------------------------
        rowval  = 3
        colval  = 1
        colspan = 1
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_image(ax,modelcube,imgext=setupdic['cube_extension'],cubelayer=maxent,fontsize=Fsize,lthick=lthick,alpha=0.5,
                                   title='model')

        #---------------------------- FMAX CUBE RES ----------------------------
        rowval  = 3
        colval  = 2
        colspan = 1
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_image(ax,residualcube,imgext=setupdic['cube_extension'],cubelayer=maxent,fontsize=Fsize,lthick=lthick,alpha=0.5,
                                   title='residual')


        #---------------------------- FFIX CUBE ----------------------------
        rowval  = 3
        colval  = 3
        colspan = 1
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_image(ax,datacube,imgext=setupdic['cube_extension'],cubelayer=entfix,fontsize=Fsize,lthick=lthick,alpha=0.5,
                                   title='Fixed $\\lambda$ layer')

        #---------------------------- FFIX CUBE MOD ----------------------------
        rowval  = 3
        colval  = 4
        colspan = 1
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_image(ax,modelcube,imgext=setupdic['cube_extension'],cubelayer=entfix,fontsize=Fsize,lthick=lthick,alpha=0.5,
                                   title='model')

        #---------------------------- FFIX CUBE RES ----------------------------
        rowval  = 3
        colval  = 5
        colspan = 1
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_image(ax,residualcube,imgext=setupdic['cube_extension'],cubelayer=entfix,fontsize=Fsize,lthick=lthick,alpha=0.5,
                                   title='residual')


        #---------------------------- FMAX CUBE HIST ----------------------------
        rowval  = 4
        colval  = 0
        colspan = 3
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_hist(ax,datacube,modelcube,dataext=setupdic['cube_extension'],modelext=setupdic['cube_extension'],
                                  layer=maxent,histbins=[-20,20,0.3],lthick=lthick,alpha=0.5,title='Max S/N cube layer',fontsize=Fsize)

        #---------------------------- FFIX CUBE HIST ----------------------------
        rowval  = 4
        colval  = 3
        colspan = 3
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_hist(ax,datacube,modelcube,dataext=setupdic['cube_extension'],modelext=setupdic['cube_extension'],
                                  layer=entfix,histbins=[-20,20,0.3],lthick=lthick,alpha=0.5,title='Fixed cube layer ('+str(wavefix)+'\AA)',
                                  fontsize=Fsize)

        #---------------------------- FMAX SPEC ZOOM ----------------------------
        rowval  = 5
        colval  = 0
        colspan = 3
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_spec(ax,spec1D,title='Spectrum around max S/N (+/-'+str(dwave)+'A)',
                                  xrange=[maxwave-dwave,maxwave+dwave],markwave=maxwave,drawbox=False,fontsize=6,
                                  plotSNcurve=False,shownoise=True,lthick=lthick,fillalpha=0.30)

        #---------------------------- FFIX SPEC ----------------------------
        rowval  = 5
        colval  = 3
        colspan = 3
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_spec(ax,spec1D,title='Full spectrum (Max S/N region shaded red)',xrange=[4800,9300],markwave=wavefix,
                                  drawbox=[maxwave-dwave,maxwave+dwave],fontsize=6,
                                  plotSNcurve=False,shownoise=True,lthick=lthick,fillalpha=0.30)

        #---------------------------- FMAX S/N SPEC ZOOM ----------------------------
        rowval  = 6
        colval  = 0
        colspan = 3
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_spec(ax,spec1D,title='',
                                  xrange=[maxwave-dwave,maxwave+dwave],markwave=maxwave,drawbox=False,fontsize=6,
                                  plotSNcurve=True,shownoise=True,lthick=lthick,fillalpha=0.30)

        #---------------------------- FFIX S/N SPEC ----------------------------
        rowval  = 6
        colval  = 3
        colspan = 3
        ax = plt.subplot2grid((Nrow,Ncol), (rowval, colval), colspan=colspan)

        tu.gen_overview_plot_spec(ax,spec1D,title='',xrange=[4800,9300],markwave=wavefix,
                                  drawbox=[maxwave-dwave,maxwave+dwave],fontsize=6,
                                  plotSNcurve=True,shownoise=True,lthick=lthick,fillalpha=0.30)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #if verbose: print ' - Saving plot to',plotname
        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

    if verbose: print '\n ... done '

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_overview_plot_image(ax,imagefile,imgext=0,cubelayer=1,title='Img Title?',fontsize=6,lthick=2,alpha=0.5,
                            cmap='coolwarm'):
    """
    Plotting commands for image (cube layer) overview plotting

    --- INPUT ---

    cubelayer     If the content of the file is a cube, provide the cube layer to plot. If
                    cubelayer = 'fmax' the layer with most flux is plotted

    """

    ax.set_title(title,fontsize=fontsize)
    if os.path.isfile(imagefile):
        imgdata = pyfits.open(imagefile)[imgext].data

        if len(imgdata.shape) == 3: # it is a cube
            imgdata = imgdata[cubelayer,:,:]

        ax.imshow(imgdata, interpolation='None',cmap=cmap,aspect='equal', origin='lower')

        ax.set_xlabel('x-pixel')
        ax.set_ylabel('y-pixel ')
        ax.set_xticks([])
        ax.set_yticks([])

    else:
        textstr = 'No image\nfound'
        ax.text(1.0,22,textstr,horizontalalignment='center',verticalalignment='center',fontsize=fontsize)

        ax.set_ylim([28,16])
        ax.plot([0.0,2.0],[28,16],'r--',lw=lthick)
        ax.plot([2.0,0.0],[28,16],'r--',lw=lthick)

        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        ax.set_xticks([])
        ax.set_yticks([])

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_overview_plot_hist(ax,data,model,dataext=0,modelext=0,layer=None,normalize=False,
                           histbins=[-10,-10,0.1],lthick=1,alpha=0.5,title='Hist Title?',fontsize=6):
    """
    Plotting commands for overview plot
    """

    ax.set_title(title,fontsize=fontsize, loc='left')
    if os.path.isfile(data) & os.path.isfile(model):
        if layer is None:
            datavec   = pyfits.open(data)[dataext].data[:,:].ravel()
            modelvec  = pyfits.open(model)[modelext].data[:,:].ravel()
        else:
            datavec   = pyfits.open(data)[dataext].data[layer,:,:].ravel()
            modelvec  = pyfits.open(model)[modelext].data[layer,:,:].ravel()

        residualvec  = datavec-modelvec

        datavec      = datavec[np.isfinite(datavec)]
        modelvec     = modelvec[np.isfinite(modelvec)]
        residualvec  = residualvec[np.isfinite(residualvec)]

        binvals      = np.arange(histbins[0],histbins[1],histbins[2])

        ax.hist(datavec,bins=binvals,color='b',lw=lthick,label='data',histtype="step",normed=normalize)
        ax.hist(modelvec,bins=binvals,color='r',lw=lthick,label='model',histtype="step",normed=normalize)
        ax.hist(residualvec,bins=binvals,color='g',lw=lthick,label='residual',histtype="step",normed=normalize)
        ylabel = '\# pixels'

        ax.set_xlabel('', fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)

        ax.set_yscale('log')

        leg = plt.legend(fancybox=True, loc='upper left',prop={'size':fontsize},ncol=1,numpoints=1)#,
                         #bbox_to_anchor=(1.25, 1.03))  # add the legend
        leg.get_frame().set_alpha(0.7)

    else:
        textstr = 'Not enough data to geenerate full plot'
        ax.text(1.0,22,textstr,horizontalalignment='center',verticalalignment='center',fontsize=fontsize)

        ax.set_ylim([28,16])
        ax.plot([0.0,2.0],[28,16],'r--',lw=lthick)
        ax.plot([2.0,0.0],[28,16],'r--',lw=lthick)

        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        ax.set_xticks([])
        ax.set_yticks([])

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_overview_plot_spec(ax,spec1Dfile,title='Spec Title?',xrange=[4800,9300],markwave=5500,drawbox=False,fontsize=6,
                           plotSNcurve=False,shownoise=True,lthick=2,fillalpha=0.30):
    """
    Plotting commands for overview plot
    """
    ax.set_title(title,fontsize=fontsize, loc='left')
    if os.path.isfile(spec1Dfile):
        specdat = pyfits.open(spec1Dfile)[1].data
        goodent = np.where((specdat['wave'] > xrange[0]) & (specdat['wave'] < xrange[1]))[0]

        wave    = specdat['wave'][goodent]
        flux    = specdat['flux'][goodent]
        fluxerr = specdat['fluxerror'][goodent]

        if markwave:
            diff    = np.abs(np.asarray(wave)-markwave)
            markent = np.where(diff == np.min(diff))[0]
            if len(markent) == 0:
                print ' WARNING the "markwave" is outside provided plotting xrange '
                markwave = False

        if plotSNcurve:
            try:
                s2ndat = specdat['s2n'][goodent]
            except:
                s2ndat = specdat['flux'][goodent]/specdat['fluxerror'][goodent]
            ax.plot(wave,s2ndat,color='b',lw=lthick)
            ylabel = 'S/N'

            if markwave:
                ax.plot([wave[markent],wave[markent]],ax.get_ylim(),color='r',linestyle='--',lw=lthick)
        else:
            if shownoise:
                plt.fill_between(wave,flux-fluxerr,flux+fluxerr,alpha=fillalpha,color='b')
            ax.plot(wave,flux,color='b',lw=lthick)
            ylabel = 'Flux [1e-20cgs]'

            if markwave:
                ax.plot([wave[markent],wave[markent]],ax.get_ylim(),color='r',linestyle='--',lw=lthick)

        ax.set_xlabel('Wavelength [\AA]', fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)

        ax.plot(xrange,[0,0],'--k',lw=lthick)

        if drawbox:
            yrangeplot = ax.get_ylim()
            plt.fill_between(drawbox,[yrangeplot[0],yrangeplot[0]],[yrangeplot[1],yrangeplot[1]],
                     color='red',alpha=fillalpha)


    else:
        textstr = 'No spectrum found'
        ax.text(1.0,22,textstr,horizontalalignment='center',verticalalignment='center',fontsize=fontsize)

        ax.set_ylim([28,16])
        ax.plot([0.0,2.0],[28,16],'r--',lw=lthick)
        ax.plot([2.0,0.0],[28,16],'r--',lw=lthick)

        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        ax.set_xticks([])
        ax.set_yticks([])

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def test_analyticVSnumerical(setupfile,outputdir,plotcubelayer,xrange=[6000,6500],yrange=[0.95,1.05],vmin=-0.005,vmax=0.005,verbose=True):
    """
    Generating plots comparing an analytic run modeling a objects with single Gaussains (source_model = gauss) to
    a numerical convolution run using the model produced in the 'gauss' run as input (source_model = modelimg).

    Useful, to ensure that the TDOSE outputs from idealized cases are identical (single objects in FoV)
    which they should be; at least within numerical uncertainties and differences in the convolutions applied.

    If an aperture run also exists, the 1D spectrum of these outpus will be overplotted as well

    --- INPUT ---
    setupfile     Setupfile used for the 'source_model = gauss'. Should be the exact same same setup used for the
                  'source_model = modelimg' run apart from the change of the 'source_model' keyword, and that the
                  *tdose_modelimage_gauss.fits was used as model (renaming it accoringly and positioning it in the
                  the 'modelimg_directory').
    plotcubelayer The layer in the cube to plot for the PSF cubes and the model cubes
    verbose       Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_utilities as tu
    setupfile = '/Users/kschmidt/work/TDOSE/tdose_setup_candels-cdfs-02.txt'
    plotdir   = '/Volumes/DATABCKUP2/TDOSEextractions/comparisonplots/'
    tu.test_analyticVSnumerical(setupfile,plotdir,1200,xrange=[6000,6500])
    tu.test_analyticVSnumerical(setupfile,plotdir,1200,xrange=[4700,9300])

    """
    setupdic  = tu.load_setup(setupfile=setupfile)
    specdir   = setupdic['spec1D_directory']
    moddir    = setupdic['models_directory']
    modimgdir = setupdic['modelimg_directory']
    objids    = setupdic['sources_to_extract']
    mie       = setupdic['model_image_ext']
    mce       = setupdic['model_cube_ext']
    smce      = setupdic['source_model_cube_ext']

    for objid in objids:
        base_cube = setupdic['data_cube'].split('/')[-1].split('.fit')[0]
        base_ref  = setupdic['ref_image'].split('/')[-1].split('.fit')[0]
        idstr     = str("%.10d" % objid)
        if setupdic['model_cutouts']:
            cutstr, cutoutsize, cut_img, cut_cube, cut_variance, cut_sourcecat = tu.get_datinfo(objid,setupdic)
            base_cube = base_cube+cutstr
            base_ref  = base_ref+cutstr

        aperturespec = specdir+setupdic['spec1D_name']+'_aperture_'+idstr+'.fits'
        if os.path.isfile(aperturespec):
            aperture  = True
            ta_spec   = pyfits.open(aperturespec)[1].data
        else:
            aperture  = False

        tg_spec   = pyfits.open(specdir+setupdic['spec1D_name']+'_gauss_'+idstr+'.fits')[1].data
        tg_smc    = pyfits.open(moddir+base_cube+'_'+smce+'_gauss.fits')['DATA_DCBGC'].data
        tg_model  = pyfits.open(moddir+base_cube+'_'+mce+'_gauss.fits')['DATA_DCBGC'].data
        tg_psf    = pyfits.open(moddir+base_cube+'_tdose_psfcube_gauss.fits')['DATA_DCBGC'].data
        tg_scales = pyfits.open(moddir+base_cube+'_'+mce+'_gauss.fits')['WAVESCL'].data
        tg_acsmod = pyfits.open(moddir+base_ref+'_'+mie+'_gauss.fits')[0].data
        tg_cubWCS = pyfits.open(moddir+base_ref+'_'+mie+'_cubeWCS_gauss.fits')[0].data

        sc_spec   = pyfits.open(specdir+setupdic['spec1D_name']+'_modelimg_'+idstr+'.fits')[1].data
        sc_smc    = pyfits.open(moddir+base_cube+'_'+smce+'_modelimg.fits')['DATA_DCBGC'].data
        sc_model  = pyfits.open(moddir+base_cube+'_'+mce+'_modelimg.fits')['DATA_DCBGC'].data
        sc_psf    = pyfits.open(moddir+base_cube+'_tdose_psfcube_modelimg.fits')['DATA_DCBGC'].data
        sc_scales = pyfits.open(moddir+base_cube+'_'+mce+'_modelimg.fits')['WAVESCL'].data
        sc_acsmod = pyfits.open(modimgdir+'model_'+base_ref+'.fits')[0].data
        sc_cubWCS = pyfits.open(moddir+base_ref+'_'+mie+'_cubeWCS_modelimg.fits')[0].data

        rangestr = '_wavelength'+str(xrange[0]).replace('.','p')+'to'+str(xrange[1]).replace('.','p')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'spec1D'+rangestr+'_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(10, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.06, right=0.81, bottom=0.15, top=0.95)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        plt.plot(tg_spec['wave'],tg_spec['flux'],'-k',label='TDOSE "gauss"',lw=lthick)
        plt.plot(sc_spec['wave'],sc_spec['flux'],'--r',label='TDOSE "modelimg"',lw=lthick)
        if aperture:
            plt.plot(ta_spec['wave'],ta_spec['flux'],'-b',label='TDOSE "aperture"',lw=lthick)

        plt.plot([tg_spec['wave'][plotcubelayer],tg_spec['wave'][plotcubelayer]],plt.gca().get_ylim(),'--k',alpha=0.5,
                 label='Layer '+str(plotcubelayer))

        plt.xlabel('Wavelength [\AA]', fontsize=Fsize)
        plt.ylabel('1D Flux', fontsize=Fsize)

        plt.xlim(xrange)
        #plt.ylim(yrange))

        leg = plt.legend(fancybox=True, loc='upper right',prop={'size':Fsize},ncol=1,numpoints=1,
                         bbox_to_anchor=(1.25, 1.03))  # add the legend
        leg.get_frame().set_alpha(0.7)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'s2n1D'+rangestr+'_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(10, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.06, right=0.81, bottom=0.15, top=0.95)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        plt.plot(tg_spec['wave'],tg_spec['s2n'],'-k',label='TDOSE "gauss"',lw=lthick)
        plt.plot(sc_spec['wave'],sc_spec['s2n'],'--r',label='TDOSE "modelimg"',lw=lthick)
        if aperture:
            plt.plot(ta_spec['wave'],ta_spec['s2n'],'-b',label='TDOSE "aperture"',lw=lthick)

        plt.plot([tg_spec['wave'][plotcubelayer],tg_spec['wave'][plotcubelayer]],plt.gca().get_ylim(),'--k',alpha=0.5,
                 label='Layer '+str(plotcubelayer))

        plt.xlabel('Wavelength [\AA]', fontsize=Fsize)
        plt.ylabel('S/N', fontsize=Fsize)

        plt.xlim(xrange)
        #plt.ylim(yrange)

        leg = plt.legend(fancybox=True, loc='upper right',prop={'size':Fsize},ncol=1,numpoints=1,
                         bbox_to_anchor=(1.25, 1.03))  # add the legend
        leg.get_frame().set_alpha(0.7)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'spec_gaussOVERmodelimg'+rangestr+'_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(10, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.06, right=0.81, bottom=0.15, top=0.95)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        ratio = tg_spec['flux']/sc_spec['flux']
        plt.plot(tg_spec['wave'],ratio,'-k',lw=lthick)
        plt.plot(xrange,[1.0,1.0],'--k',alpha=0.5,lw=lthick)

        plt.plot([tg_spec['wave'][plotcubelayer],tg_spec['wave'][plotcubelayer]],yrange,'--k',alpha=0.5,
                 label='Layer '+str(plotcubelayer))

        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   mean(gauss/modelimg)   = ',np.mean(ratio)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   median(gauss/modelimg) = ',np.median(ratio)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   std(gauss/modelimg)    = ',np.std(ratio)

        plt.xlabel('Wavelength [\AA]', fontsize=Fsize)
        plt.ylabel('1D Flux: "gauss" / "modelimg"', fontsize=Fsize)

        plt.xlim(xrange)
        plt.ylim(yrange)

        leg = plt.legend(fancybox=True, loc='upper right',prop={'size':Fsize},ncol=1,numpoints=1,
                         bbox_to_anchor=(1.25, 1.03))  # add the legend
        leg.get_frame().set_alpha(0.7)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'wavescl_gaussOVERmodelimg'+rangestr+'_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(10, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.06, right=0.81, bottom=0.15, top=0.95)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        ratio = tg_scales/sc_scales
        plt.plot(tg_spec['wave'],ratio[0],'-k',lw=lthick)
        plt.plot(xrange,[1.0,1.0],'--k',alpha=0.5,lw=lthick)

        plt.plot([tg_spec['wave'][plotcubelayer],tg_spec['wave'][plotcubelayer]],yrange,'--k',alpha=0.5,
                 label='Layer '+str(plotcubelayer))

        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   mean(gauss/modelimg)   = ',np.mean(ratio)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   median(gauss/modelimg) = ',np.median(ratio)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   std(gauss/modelimg)    = ',np.std(ratio)

        plt.xlabel('Wavelength [\AA]', fontsize=Fsize)
        plt.ylabel('Wavescale: "gauss" / "modelimg"', fontsize=Fsize)

        plt.xlim(xrange)
        plt.ylim(yrange)

        leg = plt.legend(fancybox=True, loc='upper right',prop={'size':Fsize},ncol=1,numpoints=1,
                         bbox_to_anchor=(1.25, 1.03))  # add the legend
        leg.get_frame().set_alpha(0.7)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'sumSMC_gaussOVERmodelimg'+rangestr+'_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(10, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.06, right=0.81, bottom=0.15, top=0.95)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        ratio = np.sum(np.sum(tg_smc[0],axis=1),axis=1) / np.sum(np.sum(sc_smc[0],axis=1),axis=1)
        plt.plot(tg_spec['wave'],ratio,'-k',lw=lthick)
        plt.plot(xrange,[1.0,1.0],'--k',alpha=0.5,lw=lthick)

        plt.plot([tg_spec['wave'][plotcubelayer],tg_spec['wave'][plotcubelayer]],yrange,'--k',alpha=0.5,
                 label='Layer '+str(plotcubelayer))

        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   mean(gauss/modelimg)   = ',np.mean(ratio)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   median(gauss/modelimg) = ',np.median(ratio)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   std(gauss/modelimg)    = ',np.std(ratio)

        plt.xlabel('Wavelength [\AA]', fontsize=Fsize)
        plt.ylabel('sum(SMC): "gauss" / "modelimg"', fontsize=Fsize)

        plt.xlim(xrange)
        plt.ylim(yrange)

        leg = plt.legend(fancybox=True, loc='upper right',prop={'size':Fsize},ncol=1,numpoints=1,
                         bbox_to_anchor=(1.25, 1.03))  # add the legend
        leg.get_frame().set_alpha(0.7)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'acsmodel_gaussMINUSmodelimg_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(3, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.15, right=0.9, bottom=0.1, top=0.9)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        plt.imshow(tg_acsmod-sc_acsmod,interpolation=None,origin='lower',vmin=vmin,vmax=vmax)
        plt.colorbar()

        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   sum(gauss-modelimg)     = ',np.sum(tg_acsmod-sc_acsmod)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   median(gauss-modelimg)  = ',np.median(tg_acsmod-sc_acsmod)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'acsmodelWCS_gaussMINUSmodelimg_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(3, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.15, right=0.9, bottom=0.1, top=0.9)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        plt.imshow(tg_cubWCS-sc_cubWCS,interpolation=None,origin='lower',vmin=vmin,vmax=vmax)
        plt.colorbar()

        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   sum(gauss-modelimg)   = ',np.sum(tg_cubWCS-sc_cubWCS)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   median(gauss-modelimg)= ',np.median(tg_cubWCS-sc_cubWCS)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'modelcubeSum1D_gaussOVERmodelimg_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(10, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.06, right=0.81, bottom=0.15, top=0.95)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        tg_modelsum = np.sum(np.sum(tg_model,axis=1),axis=1)
        sc_modelsum = np.sum(np.sum(sc_model,axis=1),axis=1)
        ratio       = tg_modelsum/sc_modelsum

        plt.plot(tg_spec['wave'],ratio,'-k',lw=lthick)
        plt.plot(xrange,[1.0,1.0],'--k',alpha=0.5,lw=lthick)

        plt.plot([tg_spec['wave'][plotcubelayer],tg_spec['wave'][plotcubelayer]],yrange,'--k',alpha=0.5,
                 label='Layer '+str(plotcubelayer))


        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   mean(gauss/modelimg)   = ',np.mean(ratio)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   median(gauss/modelimg) = ',np.median(ratio)
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   std(gauss/modelimg)    = ',np.std(ratio)

        plt.xlabel('Wavelength [\AA]', fontsize=Fsize)
        plt.ylabel('sum(modelcube): "gauss" / "modelimg"', fontsize=Fsize)

        plt.xlim(xrange)
        plt.ylim(yrange)

        leg = plt.legend(fancybox=True, loc='upper right',prop={'size':Fsize},ncol=1,numpoints=1,
                         bbox_to_anchor=(1.25, 1.03))  # add the legend
        leg.get_frame().set_alpha(0.7)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'modelcubeSum1DOVERWavescl_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(10, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.06, right=0.81, bottom=0.15, top=0.95)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        plt.plot(tg_spec['wave'],tg_modelsum/tg_scales[0],'-k',label='TDOSE "gauss"',lw=lthick)
        plt.plot(sc_spec['wave'],sc_modelsum/sc_scales[0],'--r',label='TDOSE "modelimg"',lw=lthick)

        plt.plot(xrange,[1.0,1.0],'--k',alpha=0.5,lw=lthick)

        plt.plot([tg_spec['wave'][plotcubelayer],tg_spec['wave'][plotcubelayer]],yrange,'--k',alpha=0.5,
                 label='Layer '+str(plotcubelayer))

        plt.xlabel('Wavelength [\AA]', fontsize=Fsize)
        plt.ylabel('sum(modelcube) / wavescales', fontsize=Fsize)

        plt.xlim(xrange)
        plt.ylim(yrange)

        leg = plt.legend(fancybox=True, loc='upper right',prop={'size':Fsize},ncol=1,numpoints=1,
                         bbox_to_anchor=(1.25, 1.03))  # add the legend
        leg.get_frame().set_alpha(0.7)

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'modelcube_gaussMINUSmodelimg_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(3, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.15, right=0.9, bottom=0.1, top=0.9)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        plt.imshow(tg_model[plotcubelayer,:,:]-sc_model[plotcubelayer,:,:],interpolation=None,origin='lower',vmin=vmin,vmax=vmax)
        plt.colorbar()

        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   sum(gauss-modelimg)     = ',np.sum(tg_model[plotcubelayer,:,:]-sc_model[plotcubelayer,:,:])
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   median(gauss-modelimg)  = ',np.median(tg_model[plotcubelayer,:,:]-sc_model[plotcubelayer,:,:])

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plotname = outputdir+'psfcube_gaussMINUSmodelimg_'+idstr+'.pdf'
        if verbose: print ' - Generating '+plotname
        fig = plt.figure(figsize=(3, 3))
        fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.15, right=0.9, bottom=0.1, top=0.9)
        Fsize  = 10
        lthick = 1
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=Fsize)
        plt.rc('xtick', labelsize=Fsize)
        plt.rc('ytick', labelsize=Fsize)
        plt.clf()
        plt.ioff()

        plt.imshow(tg_psf[plotcubelayer,:,:]-sc_psf[plotcubelayer,:,:],interpolation=None,origin='lower',vmin=vmin,vmax=vmax)
        plt.colorbar()

        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   sum(gauss-modelimg)       = ',np.sum(tg_model[plotcubelayer,:,:]-sc_model[plotcubelayer,:,:])
        print '   '+plotname.split('/')[-1].split('.pdf')[0]+':   median(gauss-modelimg)    = ',np.median(tg_model[plotcubelayer,:,:]-sc_model[plotcubelayer,:,:])

        plt.savefig(plotname)
        plt.clf()
        plt.close('all')
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def test_sersicprofiles(Ieff=1.0,reff=25.0,sersicindex=4.0,axisratio=0.5,size=1000,angle=-57,
                        outputdir='/Volumes/DATABCKUP2/TDOSEextractions/models_cutouts/',verbose=True):
    """
    Function to generate and investigate sersic profiles.

    --- EXAMPLE OF USE ---
    tu.test_sersicprofiles(Ieff=1.0,reff=25.0,sersicindex=4.0,axisratio=0.5)

    """
    theta = angle/180.0*np.pi

    if verbose: print ' - Input: \n   Ieff      =',Ieff,'\n   reff      =',reff,\
        '\n   n         =',sersicindex,'\n   axisratio =',axisratio,'\n   angle     =',angle,'\n   theta     =',theta

    x,y = np.meshgrid(np.arange(size), np.arange(size))

    mod = Sersic2D(amplitude = Ieff, r_eff = reff, n=sersicindex, x_0=size/2.0, y_0=size/2.0, ellip=1-axisratio, theta=theta)
    img = mod(x, y)
    hducube  = pyfits.PrimaryHDU(img)
    hdus = [hducube]
    hdulist = pyfits.HDUList(hdus)
    hdulist.writeto(outputdir+'model_sersic.fits',clobber=True)

    mod = Sersic2D(amplitude = Ieff, r_eff = reff, n=sersicindex, x_0=size/2.0, y_0=size/2.0, ellip=0, theta=theta)
    img = mod(x, y)
    hducube  = pyfits.PrimaryHDU(img)
    hdus = [hducube]
    hdulist = pyfits.HDUList(hdus)
    hdulist.writeto(outputdir+'model_sersic_spherical.fits',clobber=True)

    # integrate.nquad(mod, [[-np.inf, np.inf],[-np.inf, np.inf]])
    # integrate.nquad(mod, [[size/2.0-reff, size/2.0+reff],[size/2.0-reff, size/2.0+reff]])

    plt.figure()
    plt.subplot(111, xscale='log', yscale='log')
    s1 = Sersic1D(amplitude=Ieff, r_eff=reff,n=sersicindex)
    r=np.arange(0, size, 0.01)

    plt.plot(r, s1(r), color='red')
    plt.plot([reff,reff],[0,s1(reff)],color='red')
    plt.plot([0,reff],[s1(reff),s1(reff)],color='red')

    plt.axis([1e-3,1e3, 1e-3, 1e3])
    plt.xlabel('log Radius')
    plt.ylabel('log Surface Brightness')
    plt.savefig(outputdir+'model_sersic_1D.pdf')

    ent_eff       = np.where(s1(r) == s1(reff))[0][0]
    Ftot_1D       = np.trapz(s1(r),x=r,dx=0.01)
    Ftot_eff_1D   = np.trapz(s1(r[:ent_eff]),x=r[:ent_eff],dx=0.01)
    Ftot_outer_1D = np.trapz(s1(r[ent_eff:]),x=r[ent_eff:],dx=0.01)

    if verbose: print '\n - np.trapz of 1D profile for r=[0;'+str(size)+']:     Ftot_reff=',Ftot_1D
    if verbose: print ' - np.trapz of 1D profile for r=[0;r_eff]:    Ftot_reff=',Ftot_eff_1D
    if verbose: print ' - np.trapz of 1D profile for r=[r_eff;'+str(size)+']: Ftot_outer=',Ftot_outer_1D

    Ftot_2D       = integrate.quad(lambda x: x*s1(x),0 ,np.inf)[0] * 2 * np.pi
    Ftot_eff_2D   = integrate.quad(lambda x: x*s1(x),0 ,reff)[0] * 2 * np.pi
    Ftot_outer_2D = integrate.quad(lambda x: x*s1(x),reff,np.inf)[0] * 2 * np.pi

    if verbose: print '\n - scipy.integrate.quad of 2D profile for r=[0;np.inf]:       Ftot_2D       =',Ftot_2D
    if verbose: print ' - scipy.integrate.quad of 2D profile for r=[0;r_eff]:        Ftot_reff_2D  =',Ftot_eff_2D
    if verbose: print ' - scipy.integrate.quad of 2D profile for r=[r_eff;np.inf]:   Ftot_2D_outer =',Ftot_outer_2D

    Ftot_full = integrate.quad(lambda x: x*s1(x),reff,size/2.)[0] * 2 * np.pi
    if verbose: print ' - scipy.integrate.quad of 2D profile for r=[0;'+str(int(size/2.))+\
                      ']:          Ftot_2D_r'+str(int(size/2.))+'  =',Ftot_full

    Ftot_calc   = tu.get_2DsersicIeff(Ieff,reff,sersicindex,1.0,returnFtot=True)
    if verbose: print '\n - tu.get_2DsersicIeff calculation of Ftot:                   Ftot_calc     =',Ftot_calc

    Ieff_calc   = tu.get_2DsersicIeff(Ftot_2D,reff,sersicindex,1.0)
    if verbose: print ' - tu.get_2DsersicIeff calculation of Ieff from Ftot_2D:      I_eff         =',Ieff_calc

    Ieff_calc   = tu.get_2DsersicIeff(Ftot_eff_2D,reff,sersicindex,1.0)
    if verbose: print ' - tu.get_2DsersicIeff calculation of Ieff from Ftot_reff_2D: I_eff         =',Ieff_calc

    # Summing in circular mask to mimic DS9 "integration"
    r    = reff
    y,x  = np.ogrid[-size/2.0:size/2.0, -size/2.0:size/2.0]
    mask = x*x + y*y <= r*r

    sum_mask_reff = np.sum(img[mask])
    if verbose: print '\n - Sum of pixels within r_eff of sersic img array (spherical):    Fsum_reff =',sum_mask_reff

    r    = size/2.0
    y,x  = np.ogrid[-size/2.0:size/2.0, -size/2.0:size/2.0]
    mask = x*x + y*y <= r*r

    sum_mask = np.sum(img[mask])
    if verbose: print ' - Sum of pixels within r_eff of sersic img array (spherical):    Fsum_r'+str(int(size/2.))+' =',sum_mask


    Ftot_calc   = tu.get_2DsersicIeff(Ieff,reff,sersicindex,axisratio,returnFtot=True)
    if verbose: print '\n - tu.get_2DsersicIeff calculation of Ftot (axisratio='+str(axisratio)+'):               Ftot_calc  =',Ftot_calc

    Ieff_calc   = tu.get_2DsersicIeff(Ftot_2D,reff,sersicindex,axisratio)
    if verbose: print ' - tu.get_2DsersicIeff calculation of Ieff from Ftot_2D (axisratio='+str(axisratio)+'):      I_eff  =',Ieff_calc

    Ieff_calc   = tu.get_2DsersicIeff(Ftot_eff_2D,reff,sersicindex,axisratio)
    if verbose: print ' - tu.get_2DsersicIeff calculation of Ieff from Ftot_reff_2D (axisratio='+str(axisratio)+'): I_eff  =',Ieff_calc

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =