#
# Performing gauss, aperture and modelimg extractions with TDOSE
#
# -------------------------------------------------------------------------------
# Importing
import glob
import tdose
import tdose_utilities as tu

# -------------------------------------------------------------------------------
# Defining working directory and setup name and location
workingdirectory = '../../tdose_examples/'
setupname        = 'Rafelski-MXDF_ZAP_COR_V2'
setupdir         = workingdirectory+'tdose_setupfiles/'

# -------------------------------------------------------------------------------
# Generating TDOSE setup files by duplicating and filling setup template
infofile             = setupdir+'tdose_setupfile_info_'+setupname+'.txt'
namebase             = 'tdose_setupfile'
tu.duplicate_setup_template(setupdir,infofile,namebase=namebase,clobber=True,loopcols='all',infofmt="S250",infohdr=3)

# -------------------------------------------------------------------------------
# Make sure not to perform data cube cutouts. These have already been generated
# which avoids having to store the full 3D data cube on GitHub.
# The parent data cube will be available upon request if the cutout capability
# of TDOSE needs to be tested (or another data cube can be used).
performcutout     = False

# -------------------------------------------------------------------------------
# Setup for aperture extraction using corresponding setup file.
# Hence, tdose will simply drop down apertures of the size specified in the
# setup and extract the spectrum within it.
setup_aperture      = setupdir+'tdose_setupfile_'+setupname+'_aperture.txt'

# -------------------------------------------------------------------------------
# Setup for gauss extraction using corresponding setup file.
# Hence, tdose will model the reference image using a single multi-variate
# Guassian for each object in the source catalog provided in the setup.
setup_gauss         = setupdir+'/tdose_setupfile_'+setupname+'_gauss.txt'

# -------------------------------------------------------------------------------
# Setup for modelimg extraction using corresponding setup file.
# Hence, TDOSE will load reference image model (cube) directly from the
# specified loaction in the setup, and base the extraction and contamination
# handling on this model. In this case the model cubes were generated from
# galfit multi-sersic models.
setup_modelimg      = setupdir+'tdose_setupfile_'+setupname+'_modelimg.txt'

# -------------------------------------------------------------------------------
# Performing the actual TDOSE extractions
for tdose_setup in [setup_modelimg,setup_gauss,setup_aperture]: #
    tdose.perform_extraction(setupfile=tdose_setup,performcutout=performcutout,generatesourcecat=True,
                             verbose=True,verbosefull=True,clobber=True,store1Dspectra=True,plot1Dspectra=True,
                             skipextractedobjects=False,logterminaloutput=False)

# -------------------------------------------------------------------------------
# Stripping source data cube extension from the extracted spectra
# This helps limit the size of a sample of extracted spectra
outputdir = workingdirectory+'tdose_spectra/tdose_spectra_stripped/'
spectra   = glob.glob(workingdirectory+'tdose_spectra/tdose_spectrum_*.fits')
for spectrum in spectra:
    tu.strip_extension_from_fitsfile(spectrum,outputdir,removeextension='SOURCECUBE',overwrite=True,verbose=True)

