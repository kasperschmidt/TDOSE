# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import sys
import pyfits
import scipy.ndimage
import scipy.optimize as opt
import tdose_utilities as tu
import tdose_extract_spectra as tes
import matplotlib as mpl
import matplotlib.pylab as plt
import tdose_model_FoV as tmf
import pdb
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_spectra(datacube,sourcemodelcube,wavelengths,speclist,specids='None',outputdir='./',noisecube=False,
                    sourcemodel_hdr='None',verbose=True):
    """
    Wrapper for tes.extract_spectrum() to extract mutliple spectra

    --- INPUT ----
    datacube          Datacube to extract spectra from
    sourcemodelcube   Cube containing the source models for each object used as "extraction cube"
                      Dimensions should be [Nsources,datacube.shape]
    wavelengths       Wavelength vector to use for extracted 1D spectrum.
    speclist          List of spectra to extract. Indexes corresponding to the source models in the
                      sourcemodlecube
    specids           List of IDs to use in naming of output for source models referred to in "speclist"
    noisecube         Cube with uncertainties (sqrt(variance)) of data cube to be used in extraction
    souremodel_hdr    If not 'None' provide a basic fits header for the source model cubes extracted
                      and they will be appended to the individual output fits file containing the extracted
                      spectra.
    verbose           Toggle verbosity

    --- EXAMPLE OF USE ---

    """
    if verbose: print ' - Check that source models indicated are present in source model cube '
    specnames  = []
    Nmodels    = sourcemodelcube.shape[0]
    maxobj     = np.max(speclist)
    if maxobj >= Nmodels:
        sys.exit(' ---> Object model "'+str(maxobj)+'" is not included in source model cube (models start at 0)')
    else:
        if verbose: print '   All object models appear to be included in the '+str(Nmodels)+' source models found in cube'

    if datacube.shape != sourcemodelcube[0].shape:
        sys.exit(' ---> Shape of datacube ('+str(datacube.shape)+') and shape of source models ('+
                 sourcemodelcube[0].shape+') do not match.')

    sourcemodel_sum = np.sum(sourcemodelcube,axis=0)
    for spec in speclist:
        if specids == 'None':
            specid = spec
        else:
            specid = specids[spec]

        specname = outputdir+'tdose_spectrum_'+str("%.12d" % specid)+'.fits'
        specnames.append(specname)

        sourcemodel     = sourcemodelcube[spec,:,:,:]
        sourceweights   = sourcemodel/sourcemodel_sum  # fractional flux of model for given source in each pixel
        sourcemodel_hdr.append(('OBJMODEL',spec      ,'Source model number in parent source model cube'),end=True)
        sourcemodel_hdr.append(('OBJID   ',specid    ,'ID of source'),end=True)

        if verbose:
            infostr = ' - Extracting spectrum '+str("%6.f" % (spec+1))+' / '+str("%6.f" % len(speclist))
            sys.stdout.write("%s\r" % infostr)
            sys.stdout.flush()

        sourceoutput = tes.extract_spectrum(datacube,sourceweights,wavelengths,specname=specname,
                                            noisecube=noisecube,spec1Dmethod='sum',sourcecube_hdr=sourcemodel_hdr,
                                            verbose=verbose)

    if verbose: print '\n - Done extracting spectra. Returning list of fits files containing spectra'
    return specnames
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_spectrum(datacube,sourceweights,wavelengths,specname='tdose_extract_spectra_extractedspec.fits',
                     noisecube=False,spec1Dmethod='sum',sourcecube_hdr='None',verbose=True):
    """
    Extracting a spectrum from a data cube given a source model (cube) to be used as 'extraction cube'

    --- INPUT ---
    datacube          Datacube to extract spectra from
    sourceweights     Weights from source model to use as "extraction cube". The weights should contain the
                      fractional flux belonging to the source in each pixel
    wavelengths       Wavelength vector to use for extracted 1D spectrum.
    noisecube         Cube with uncertainties (sqrt(variance)) of data cube to be used in extraction
    spec1Dmethod      Method used to extract 1D spectrum from source cube with
    sourcecube_hdr    If not 'None' provide a fits header for the source cube and it ill be appended to the
                      output fits file.
    verbose           Toggle verbosity

    --- EXAMPLE OF USE ---


    """
    if verbose: print ' - Checking shape of data and source model cubes'

    if datacube.shape != sourceweights.shape:
        sys.exit(' ---> Shape of datacube ('+str(datacube.shape)+') and source weights ('+
                 sourceweights.shape+') do not match.')
    else:
        if verbose: print '   dimensions match; proceeding with extraction '
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Applying weights to "datacube" to obtain source cube '
    sourcecube     = datacube*sourceweights

    if noisecube:
        if verbose: print ' - Using "noisecube" for error propagation '
        datanoise = noisecube
    else:
        if verbose: print ' - No "noisecube" provided. Setting all errors to 1'
        datanoise = np.ones(datacube.shape)

    if verbose: print ' - Assuming uncertainty on source weights equals the datanoise when propgating errors'
    sourceweights_err = datanoise
    sourcecube_err    = sourcecube * np.sqrt( (datanoise/datacube)**2 + (sourceweights_err/sourceweights)**2 )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Generating 1D spectrum from source cube via:'
    spec_wave   = wavelengths
    maskinvalid = np.ma.masked_invalid(sourcecube * sourcecube_err).mask
    if spec1Dmethod == 'sum':
        if verbose: print '   Simple summation of fluxes in sourcecube.'
        spec_flux = np.sum(np.sum(np.ma.array(sourcecube,mask=maskinvalid),axis=1),axis=1).filled()
        if verbose: print '   Errors are propagated as sum of squares.'
        spec_err  = np.sqrt( np.sum( np.sum(np.ma.array(sourcecube_err,mask=maskinvalid)**2,axis=1),axis=1) ).filled()
    else:
        sys.exit(' ---> The chosen spec1Dmethod ('+str(spec1Dmethod)+') and source model (')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Saving extracted 1D spectrum and source cube to \n   '+specname
    mainHDU = pyfits.PrimaryHDU()       # primary HDU
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    c1 = pyfits.Column(name='wave', format='D', unit='ANGSTROMS', array=spec_wave)
    c2 = pyfits.Column(name='flux', format='D', unit='', array=spec_flux)
    c3 = pyfits.Column(name='fluxerror', format='D', unit='', array=spec_err)

    coldefs = pyfits.ColDefs([c1,c2,c3])
    th = pyfits.new_table(coldefs) # creating default header

    # writing hdrkeys:'---KEY--',                             '----------------MAX LENGTH COMMENT-------------'
    th.header.append(('EXTNAME ','SPEC1D'                     ,'cube containing source'),end=True)
    th.header.append(('SPECMETH' , spec1Dmethod               ,'Method used for spectral extraction'),end=True)
    head    = th.header

    tbHDU  = pyfits.new_table(coldefs, header=head)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if sourcecube_hdr != 'None':
        sourceHDU        = pyfits.ImageHDU(sourcecube)       # default HDU with default minimal header
        for hdrkey in sourcecube_hdr.keys():
            if not hdrkey in sourceHDU.header.keys():
                sourceHDU.header.append((hdrkey,sourcecube_hdr[hdrkey],sourcecube_hdr.comments[hdrkey]),end=True)

        sourceHDU.header.append(('EXTNAME ','SOURCECUBE'            ,'cube containing source'),end=True)

        hdulist = pyfits.HDUList([mainHDU,tbHDU,sourceHDU])
    else:
        hdulist = pyfits.HDUList([mainHDU,tbHDU])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    hdulist.writeto(specname, clobber=True)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return sourcecube, sourcecube_err, spec_wave, spec_flux, spec_err
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =