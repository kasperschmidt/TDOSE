# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import os
import sys
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
    import tdose_load_setup as tls
    setup = tls.load_setup(setupfile='./tdose_setup_template.txt')

    """
    if verbose: print ' --- tdose_load_setup.load_setup() --- '
    #------------------------------------------------------------------------------------------------------
    if verbose: print ' -  Loading setup for TDOSE in '+setupfile
    setup_arr = np.genfromtxt(setupfile,dtype=None,names=None)
    setup_dic = {}
    for ii in xrange(setup_arr.shape[0]):
        try:
            val = float(setup_arr[ii,1])
        except:
            val = str(setup_arr[ii,1])
        setup_dic[setup_arr[ii,0]] = val
    if verbose: print ' -  Returning dictionary containing setup parameters'
    return setup_dic
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def generate_setup_template(outputfile='./tdose_setup_template.txt',clobber=False,verbose=True):
    """
    Generate setup text file template

    --- INPUT ---
    outputfile    The name of the output which will contain the TDOSE setup template

    --- EXAMPLE OF USE ---
    import tdose_load_setup as tls
    tls.generate_setup_template(outputfile='./tdose_setup_template.txt',clobber=True)

    """
    if verbose: print ' --- tdose_load_setup.generate_setup_template() --- '
    #------------------------------------------------------------------------------------------------------
    if os.path.isfile(outputfile) & (clobber == False):
        sys.exit(' ---> Outputfile already exists and clobber=False ')
    else:
        if verbose: print ' - Will store setup template in '+outputfile
        if os.path.isfile(outputfile) & (clobber == True):
            if verbose: print ' - Output already exists but clobber=True so overwriting it '

        setuptemplate = """
# Template for TDOSE (http://github.com/kasperschmidt/TDOSE) setup file
# Generated with tdose_load_setup.generate_setup_template()
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - DATA SETUP  - - - - - - - - - - - - - - - - - - - - - - - - - - -
data_cube         ./filename.fits   # Path and name of data cube to extract spectra from

ref_image         ./filename.fits   # Path and name of fits file containing image to use as reference when creating source model
img_extension     0                 # Name or number of fits extension containing reference image

object_catalog    ./filename.fits   # Path and name of object catalog containing objects to extract spectra for
objcat_xposcol    x_image           # Column containing x pixel position in object_catalog
objcat_yposcol    y_image           # Column containing y pixel position in object_catalog

# - - - - - - - - - - - - - - - - - - - - - - - - OBJECT MODEL SETUP  - - - - - - - - - - - - - - - - - - - - - - - - -
obj_model         gauss             # Indicate what model to use for objects; gauss, mog, galfit
model_name        ./default         # Path and base name (i.e., without extension) of output to generate

# - - - - - - - - - - - - - - - - - - - - - - - - - PSF MODEL SETUP - - - - - - - - - - - - - - - - - - - - - - - - - -
psf_type          data              # Select PSF type to use. If data 'psf_model' is used. If 'analytic' a
                                    # PSF model is build using the given setup (not enabled as of 151209)
psf_model         ./filename.fits   # Path and name of PSF model
psf_name          ./default         # Path and base name (i.e., without extension) of output to generate

# - - - - - - - - - - - - - - - - - - - - - - - SPECTRAL EXTRACTION SETUP - - - - - - - - - - - - - - - - - - - - - - -
spec_name         ./default         # Path and base name (i.e., without extension) of output to generate

# - - - - - - - - - - - - - - - - - - - - - - - - - MODIFY CUBE SETUP - - - - - - - - - - - - - - - - - - - - - - - - -
obj_remove        ./filename.fits   # Path and name of catalog of objects to remove from "data_cube"
mod_cat_name      ./default         # Path and (base)name of output cube to generate

"""
        fout = open(outputfile,'w')
        fout.write(setuptemplate)
        fout.close()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =