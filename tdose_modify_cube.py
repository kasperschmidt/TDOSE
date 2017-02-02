# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import sys
import pyfits
import tdose_utilities as tu
import pdb
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def remove_object(datacube,sourcemodelcube,objects=[1],remove=True,dataext=0,sourcemodelext=0,
                  savecube=False,clobber=False,verbose=True):
    """
    Use source model cube to remove object(s) from data cube

    --- INPUT ---
    datacube
    sourcemodelcube
    objects            The objects to remove from data cube. Provide number of source sub-cube in
                       "sourcemodelcube".
    remove             If true objects in "objects" will be removed from "datacube". If remove=False
                       everything but the objects listed in "objects" will be removed.
    savecube           If a string is provided the modified cube will be stored in a new fits file
                       appending the provided string to the data cube file name.
    clobber            If true any existing fits file will be overwritten if modified cube is saved
    verbose            Toggle verbosity

    --- EXAMPLE OF USE ---


    """
    if verbose: print ' - Loading data and source model cubes '
    dataarr         = pyfits.open(datacube)[dataext].data
    dataarr_hdr     = pyfits.open(datacube)[dataext].header

    sourcemodel     = pyfits.open(sourcemodelcube)[sourcemodelext].data
    sourcemodel_hdr = pyfits.open(sourcemodelcube)[sourcemodelext].header
    Nmodels         = sourcemodel.shape[0]
    models          = np.arange(Nmodels)

    if verbose: print ' - Check that all objects indicated are present in source model cube '
    objects  = np.asarray(objects)

    maxobj = np.max(np.abs(objects))
    if maxobj > Nmodels:
        sys.exit(' ---> Object model "'+str(maxobj)+'" is not included in source model cube ')
    else:
        if verbose: print '   All object models appear to be included in the '+str(Nmodels)+' source models found in cube'

    if verbose: print ' - Determining objects (source models) to remove from data cube '
    if remove:
        obj_remove = objects
        obj_keep   = np.setdiff1d(models,obj_remove)
    else:
        obj_keep   = objects
        obj_remove = np.setdiff1d(models,obj_keep)

    remove_cube = np.sum(sourcemodel[obj_remove,:,:,:],axis=0)

    modified_cube = dataarr - remove_cube

    if savecube:
        outname = datacube.replace('.fits','_'+str(savecube)+'.fits')
        if verbose: print ' - Saving modified cube to \n   '+outname
        if verbose: print '   (Using datacube header with modification keywords appended) '
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        hducube = pyfits.PrimaryHDU(modified_cube)       # default HDU with default minimal header
        hducube.header = dataarr_hdr

        # writing hdrkeys:     '---KEY--',                       '----------------MAX LENGTH COMMENT-------------'
        hducube.header.append(('MODIFIED',                'True','Cube is modified with tdose_modify_cube.py?'),end=True)
        hducube.header.append(('MODTIME ',   tu.get_now_string(),'Date and time of cube modification'),end=True)
        hducube.header.append(('NREMOVE ',       len(obj_remove),'Number of sources removed from cube'),end=True)
        hducube.header.append(('NKEEP   ',         len(obj_keep),'Number of sources kept in cube'),end=True)
        hducube.header.append(('COMMENT ','Source models removed:'+','.join([str(oo) for oo in obj_keep])),end=True)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        hdulist = pyfits.HDUList([hducube])       # turn header into to hdulist
        hdulist.writeto(outname,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)

    return modified_cube

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =