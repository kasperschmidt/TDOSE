# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import sys
import pyfits
import collections
import tdose_utilities as tu
import tdose_extract_spectra as tes
import tdose_build_mock_cube as tbmc
import pdb
import matplotlib.pyplot as plt
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_spectra(model_cube_file,source_association_dictionary=None,outputdir='./',clobber=False,
                    noise_cube_file=None,noise_cube_ext='ERROR',source_model_cube_file=None,verbose=True):
    """
    Assemble the spectra determined by the wavelength layer scaling of the normalized models
    when generating the source model cube

    --- INPUT ---
    model_cube_file
    source_association_dictionary
    outputdir
    clobber
    noise_cube_file
    noise_cube_ext
    source_model_cube_file
    verbose

    --- EXAMPLE OF USE ---


    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Loading data needed for spectral assembly'
    model_cube        = pyfits.open(model_cube_file)[0].data
    model_cube_hdr    = pyfits.open(model_cube_file)[0].header
    layer_scale_arr   = pyfits.open(model_cube_file)[1].data
    if noise_cube_file is not None:
        noise_cube        = pyfits.open(noise_cube_file)[noise_cube_ext].data
        source_model_cube = pyfits.open(source_model_cube_file)[0].data
    else:
        noise_cube        = None
        source_model_cube = None
    Nsources = layer_scale_arr.shape[0]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if source_association_dictionary is None:
        if verbose: print ' - Building default source association dictionary ' \
                          '(determining what sources are combined into objects), i.e., one source per object '
        sourcIDs_dic = collections.OrderedDict()
        for oo in xrange(Nsources):
            sourcIDs_dic[str(oo)] = [oo]
    else:
        sourcIDs_dic = source_association_dictionary
    Nobj = len(sourcIDs_dic.keys())
    if verbose: print ' - Found '+str(Nobj)+' objects to generate spectra for in source_association_dictionary '

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Assembling wavelength vector for spectra '
    wavelengths     = np.arange(model_cube.shape[0]) * model_cube_hdr['CD3_3'] + model_cube_hdr['CRVAL3'] - \
                      (model_cube_hdr['CRPIX3']-1.0)*model_cube_hdr['CD3_3']

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    specfiles = []
    for oo, key in enumerate(sourcIDs_dic.keys()):
        obj_cube_hdr = model_cube_hdr.copy()
        specid       = str("%.12d" % int(key))
        specname     = outputdir+'tdose_spectrum_'+specid+'.fits'
        specfiles.append(specname)
        sourceIDs    = sourcIDs_dic[key]

        obj_cube_hdr.append(('OBJID   ',specid         ,'ID of object'),end=True)
        obj_cube_hdr.append(('SRCIDS  ',str(sourceIDs) ,'IDs of sources combined in object'),end=True)

        if verbose:
            infostr = ' - Extracting spectrum '+str("%6.f" % (oo+1))+' / '+str("%6.f" % Nobj)
            sys.stdout.write("%s\r" % infostr)
            sys.stdout.flush()

        sourceoutput = tes.extract_spectrum(sourceIDs,layer_scale_arr,wavelengths,noise_cube=noise_cube,
                                            source_model_cube=source_model_cube,
                                            specname=specname,obj_cube_hdr=obj_cube_hdr,clobber=clobber,verbose=True)

    if verbose: print '\n - Done extracting spectra. Returning list of fits files generated'
    return specfiles
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_spectrum(sourceIDs,layer_scale_arr,wavelengths,noise_cube=None,source_model_cube=None,
                     specname='tdose_extract_spectra_extractedspec.fits',obj_cube_hdr=None,
                     clobber=False,verbose=True):
    """
    Extracting a spectrum based on the layer scale image from the model cube priovide a list of sources to combine.
    Noise is estimated from the noise cube (of the data)


    --- INPUT ---
    sourceIDs         The source IDs to combine into spectrum
    layer_scale_arr   Layer scale array (or image) produced when generating the model cube
                      fractional flux belonging to the source in each pixel
    wavelengths       Wavelength vector to use for extracted 1D spectrum.
    noise_cube        Cube with uncertainties (sqrt(variance)) of data cube to be used for estimating 1D uncertainties
                      To estimate S/N and 1D noise, providing a source model cube is required
    source_model_cube Source model cube containing the model cube for each individual source seperately
    specname          Name of file to save spectrum to
    obj_cube_hdr      Provide a template header to save the object cube (from combining the individual source cube)
                      as an extension to the extracted spectrum
    clobber           To overwrite existing files set clobber=True
    verbose           Toggle verbosity

    --- EXAMPLE OF USE ---


    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Checking shape of wavelengths and layer_scale_arr'
    if wavelengths.shape[0] != layer_scale_arr.shape[1]:
        sys.exit(' ---> Shape of wavelength vector ('+str(wavelengths.shape)+
                 ') and wavelength dimension of layer scale array ('+
                 layer_scale_arr.shape[1].shape+') do not match.')
    else:
        if verbose: print '   dimensions match; proceeding...'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Checking all sources have spectra in layer_scale_arr'
    maxsource = np.max(sourceIDs)
    if maxsource >= layer_scale_arr.shape[0]:
        sys.exit(' ---> Sources in list '+str(str(sourceIDs))+
                 ' not available among '+layer_scale_arr.shape[0].shape+' sources in layer_scale_arr.')
    else:
        if verbose: print '   All sources exist in layer_scale_arr; proceeding...'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Assembling object spectrum from source scaling'
    source_ent = np.asarray(sourceIDs).astype(int)
    if len(source_ent) < 1:
        spec_1D   = layer_scale_arr[source_ent,:]
    else:
        spec_1D   = np.sum( layer_scale_arr[source_ent,:],axis=0)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if noise_cube is not None:
        if verbose: print ' - Estimate S/N at each wavelength for 1D spectrum'
        total_cube    = np.sum(source_model_cube,axis=0)
        object_cube   = np.sum(source_model_cube[source_ent,:,:],axis=0)

        if verbose: print '   Estimating fraction flux belonging to object in each cube pixel'
        fluxfrac_cube = object_cube/total_cube

        fluxfrac_min  = 0.1
        if verbose: print '   Defining pixel mask (ignoring NaN pixels and pixels with <'+str(fluxfrac_min)+\
                          ' of total pixel flux in model cube) '
        pix_mask      = (fluxfrac_cube < fluxfrac_min)
        invalid_mask1 = np.ma.masked_invalid(fluxfrac_cube).mask
        invalid_mask2 = np.ma.masked_invalid(noise_cube).mask

        # combining mask making sure all individual mask pixels have True for it to be true in combined mask
        comb_mask     = (pix_mask | invalid_mask1 | invalid_mask2)

        if verbose: print '   Calculating noise propogated as d_spec_k = sqrt( SUMij (d_pixij / fluxfrac)**2 )'

        squared_ratio = ( np.ma.array(noise_cube,mask=comb_mask) / np.ma.array(fluxfrac_cube,mask=comb_mask) )**2
        noise_masked  = np.sqrt( np.sum( np.sum( squared_ratio, axis=1), axis=1) )
        noise_1D      = noise_masked.filled()

        if verbose: print '   Generating S/N vecotr'
        SN_1D         = spec_1D / noise_1D
    else:
        if verbose: print ' - No "noise_cube" provided. Setting all errors and S/N values to NaN'
        SN_1D    = np.zeros(spec_1D.shape)*np.NaN
        noise_1D = np.zeros(spec_1D.shape)*np.NaN

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Saving extracted 1D spectrum and source cube to \n   '+specname
    mainHDU = pyfits.PrimaryHDU()       # primary HDU
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    c1 = pyfits.Column(name='wave',      format='D', unit='ANGSTROMS', array=wavelengths)
    c2 = pyfits.Column(name='flux',      format='D', unit='', array=spec_1D)
    c3 = pyfits.Column(name='fluxerror', format='D', unit='', array=noise_1D)
    c4 = pyfits.Column(name='s2n',       format='D', unit='', array=SN_1D)

    coldefs = pyfits.ColDefs([c1,c2,c3,c4])
    th = pyfits.new_table(coldefs) # creating default header

    # writing hdrkeys:'---KEY--',                             '----------------MAX LENGTH COMMENT-------------'
    th.header.append(('EXTNAME ','SPEC1D'                     ,'cube containing source'),end=True)
    head    = th.header

    tbHDU  = pyfits.new_table(coldefs, header=head)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if obj_cube_hdr is not None:
        objHDU        = pyfits.ImageHDU(object_cube)
        for hdrkey in obj_cube_hdr.keys():
            if not hdrkey in objHDU.header.keys():
                objHDU.header.append((hdrkey,obj_cube_hdr[hdrkey],obj_cube_hdr.comments[hdrkey]),end=True)

        objHDU.header.append(('EXTNAME ','SOURCECUBE'            ,'cube containing source'),end=True)

        hdulist = pyfits.HDUList([mainHDU,tbHDU,objHDU])
    else:
        hdulist = pyfits.HDUList([mainHDU,tbHDU])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    hdulist.writeto(specname, clobber=clobber)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return wavelengths, spec_1D, noise_1D, object_cube

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_spectra_viasourcemodelcube(datacube,sourcemodelcube,wavelengths,speclist,specids='None',outputdir='./',
                                       noisecube=False,sourcemodel_hdr='None',verbose=True):
    """
    Wrapper for tes.extract_spectrum_viasourcemodelcube() to extract mutliple spectra

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
    for ss, spec in enumerate(speclist):
        if specids == 'None':
            specid = spec
        else:
            specid = specids[ss]

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

        sourceoutput = tes.extract_spectrum_viasourcemodelcube(datacube,sourceweights,wavelengths,specname=specname,
                                                               noisecube=noisecube,spec1Dmethod='sum',
                                                               sourcecube_hdr=sourcemodel_hdr,verbose=verbose)

    if verbose: print '\n - Done extracting spectra. Returning list of fits files containing spectra'
    return specnames
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def extract_spectrum_viasourcemodelcube(datacube,sourceweights,wavelengths,
                                        specname='tdose_extract_spectra_extractedspec.fits',
                                        noisecube=None,spec1Dmethod='sum',sourcecube_hdr='None',verbose=True):
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

    if noisecube is not None:
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
    elif spec1Dmethod == 'sum_SNweight':
        pdb.set_trace()
    else:
        sys.exit(' ---> The chosen spec1Dmethod ('+str(spec1Dmethod)+') is invalid')

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
def plot_1Dspecs(filelist,plotname='./tdose_1Dspectra.pdf',colors=None,labels=None,plotSNcurve=False,yrange=None,
                 simsources=None,simsourcefile='/Users/kschmidt/work/TDOSE/mock_cube_sourcecat161213_all.fits',
                 sim_cube_dim=None,showspecs=False,verbose=True):
    """
    Simple plots of multiple 1D spectra

    """
    if verbose: print ' - Plotting spectra to '+plotname
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
    #plt.title(plotname.split('TDOSE 1D spectra'),fontsize=Fsize)

    for ff, specfile in enumerate(filelist):
        specdat = pyfits.open(specfile)[1].data

        if colors is None:
            spec_color = None
        else:
            spec_color = colors[ff]

        if labels is None:
            spec_label = specfile
        else:
            spec_label = labels[ff]

        if plotSNcurve:
            plt.plot(specdat['wave'],specdat['s2n'],color=spec_color,lw=lthick*2, label=spec_label)
            ylabel = 'S/N'
        else:
            fillalpha = 0.30
            #if spec_color == 'green': pdb.set_trace()
            plt.fill_between(specdat['wave'],specdat['flux']-specdat['fluxerror'],specdat['flux']+specdat['fluxerror'],
                             alpha=fillalpha,color=spec_color)
            plt.plot(specdat['wave'],specdat['flux'],color=spec_color,lw=lthick*2, label=spec_label)
            ylabel = 'flux'

    if simsources is not None:
        sim_total = np.zeros(len(specdat['wave']))
        for sourcenumber in simsources:
            sourcedat = pyfits.open(simsourcefile)[1].data
            xpos       = sourcedat['xpos'][sourcenumber]
            ypos       = sourcedat['ypos'][sourcenumber]
            fluxscale  = sourcedat['fluxscale'][sourcenumber]
            sourcetype = sourcedat['sourcetype'][sourcenumber]
            spectype   = sourcedat['spectype'][sourcenumber]
            sourcecube = tbmc.gen_source_cube([ypos,xpos],fluxscale,sourcetype,spectype,cube_dim=sim_cube_dim,
                                              verbose=verbose,showsourceimgs=False)

            simspec    = np.sum( np.sum(sourcecube, axis=1), axis=1)
            sim_total  = sim_total + simspec

            plt.plot(specdat['wave'],simspec,'--',color='black',lw=lthick)

        plt.plot(specdat['wave'],sim_total,'--',color='black',lw=lthick*2,
                 label='Sim. spectrum: \nsimsource='+str(simsources))


    plt.xlabel('Wavelength [\AA]', fontsize=Fsize)
    plt.ylabel(ylabel, fontsize=Fsize)

    if yrange is not None:
        plt.ylim(yrange)

    leg = plt.legend(fancybox=True, loc='upper right',prop={'size':Fsize},ncol=1,numpoints=1,
                     bbox_to_anchor=(1.25, 1.03))  # add the legend
    leg.get_frame().set_alpha(0.7)

    if showspecs:
        if verbose: print '   Showing plot (not saving to file)'
        plt.show()
    else:
        if verbose: print '   Saving plot to',plotname
        plt.savefig(plotname)

    plt.clf()
    plt.close('all')

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =