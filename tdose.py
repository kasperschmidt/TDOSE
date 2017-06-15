# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import pyfits
import pdb
import time
import os
import sys
import numpy as np
import collections
import astropy
from astropy import wcs
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import tdose
import tdose_utilities as tu
import tdose_modify_cube as tmoc
import tdose_model_FoV as tmf
import tdose_model_cube as tmc
import tdose_extract_spectra as tes
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def perform_extraction(setupfile='./tdose_setup_template.txt',
                       performcutout=True,generatesourcecat=True,modelrefimage=True,refimagemodel2cubewcs=True,
                       definePSF=True,modeldatacube=True,createsourcecube=True,store1Dspectra=True,plot1Dspectra=True,
                       plotS2Nspectra=True,save_init_model_output=False,clobber=False,verbose=True,verbosefull=False,
                       logterminaloutput=False):
    """
    Perform extraction of spectra from data cube based on information in TDOSE setup file

    --- INPUT ---
    setupfile              TDOSE setup file. Templetae can be generated with tu.generate_setup_template()
    performcutout          To skip cutting out subcubes and images (i.e., if the cutouts have already been
                           genereated and exist) set performcutout=False
    generatesourcecat      To skip generating the cutout source catalogs from the main source catalog of sources
                           to model (e.g., after editing the source catalog) set generatesourcecat=False
    modelrefimage          To skip modeling the reference image set modelrefimage=False
    refimagemodel2cubewcs  To skip converting the refence image model to the cube WCS system set refimagemodel2cubewcs=False
    definePSF              To skip generating the PSF definePSF=False
    modeldatacube          To skip modeling the data cube set modeldatacube=False
    createsourcecube       To skip creating the source model cube set createsourcecube=False
    store1Dspectra         To skip storing the 1D spectra to binary fits tables set store1Dspectra=False
    plot1Dspectra          Plot the 1D spectra after extracting them
    plotS2Nspectra         Plot signal-to-noise spectra after extracting the 1D spectra
    save_init_model_output If a SExtractor catalog is provide to the keyword gauss_guess in the setup file
                           an initial guess including the SExtractor fits is generated for the Gaussian model.
                           To save a ds9 region, image and paramater list (the two latter is available from the default
                           output of the TDOSE modeling) set save_init_model_output=True
    clobber                If True existing output files will be overwritten
    verbose                Toggle verbosity
    verbosefull            Toggle extended verbosity
    logterminaloutput      The setup file used for the run will be looged (copied to the spec1D_directory) automatically
                           for each TDOSE extraction. To also log the output from the terminal set logterminaloutput=True
                           In this case no TDOSE output will be passed to the terminal.
    --- EXAMPLE OF USE ---
    import tdose

    # full extraction with minimal text output to prompt
    tdose.perform_extraction(setupfile='./tdose_setup_candels-cdfs-02.txt',verbose=True,verbosefull=False)

    # only plotting:
    tdose.perform_extraction(setupfile='./tdose_setup_candels-cdfs-02.txt',performcutout=False,generatesourcecat=False,modelrefimage=False,refimagemodel2cubewcs=False,definePSF=False,modeldatacube=False,createsourcecube=False,store1Dspectra=False,plot1Dspectra=True,clobber=True,verbosefull=False)


    """
    # defining function withing the routing that can be called by the output logger
    def tdosefunction(setupfile,performcutout,generatesourcecat,modelrefimage,refimagemodel2cubewcs,
                       definePSF,modeldatacube,createsourcecube,store1Dspectra,plot1Dspectra,
                       plotS2Nspectra,save_init_model_output,clobber,verbose,verbosefull):
        start_time = time.clock()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print '=================================================================================================='
        if verbose: print ' TDOSE: Loading setup                                       '+\
                          '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        if verbosefull:
            verbose = True

        setupdic        = tu.load_setup(setupfile,verbose=verbose)

        sourcecat_init  = setupdic['source_catalog']
        sourcedat_init  = pyfits.open(sourcecat_init)[1].data
        sourcehdr_init  = pyfits.open(sourcecat_init)[1].header
        sourceids_init  = sourcedat_init[setupdic['sourcecat_IDcol']]

        Nsources        = len(sourceids_init)
        sourcenumber    = np.arange(Nsources)

        if type(setupdic['sources_to_extract']) == np.str_ or (type(setupdic['sources_to_extract']) == str):
            if setupdic['sources_to_extract'].lower() == 'all':
                extractids = sourceids_init.astype(float)
            else:
                extractids = np.genfromtxt(setupdic['sources_to_extract'],dtype=None,comments='#')
                extractids = list(extractids.astype(float))
        else:
            extractids = setupdic['sources_to_extract']
        Nextractions = len(extractids)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print '=================================================================================================='
        if verbose: print ' TDOSE: Logging setup                                       '+\
                          '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'

        setuplog = setupdic['spec1D_directory']+setupfile.split('/')[-1].replace('.txt','_logged.txt')
        if os.path.isfile(setuplog) & (clobber == False):
            if verbose: print ' - WARNING Logged setupfile exists and clobber = False. Not storing setup '

        else:
            if verbose: print ' - Writing setup and command to spec1D_directory to log extraction setup and command that was run'
            setupinfo    = open(setupfile,'r')
            setupcontent = setupinfo.read()
            setupinfo.close()

            cmdthatwasrun = "import tdose; tdose.perform_extraction(setupfile='%s',performcutout=%s,generatesourcecat=%s,modelrefimage=%s," \
                            "refimagemodel2cubewcs=%s,definePSF=%s,modeldatacube=%s,createsourcecube=%s,store1Dspectra=%s," \
                            "plot1Dspectra=%s,plotS2Nspectra=%s,save_init_model_output=%s,clobber=%s,verbose=%s,verbosefull=%s," \
                            "logterminaloutput=%s)" % \
                            (setuplog,performcutout,generatesourcecat,modelrefimage,refimagemodel2cubewcs,definePSF,modeldatacube,
                             createsourcecube,store1Dspectra,plot1Dspectra,plotS2Nspectra,save_init_model_output,clobber,verbose,
                             verbosefull,logterminaloutput)

            loginfo = open(setuplog, 'w')
            loginfo.write("# The setup file appended below was run with the command: \n# "+cmdthatwasrun+
                          " \n# on "+tu.get_now_string()+'\n# ')

            loginfo.write(setupcontent)
            loginfo.close()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if setupdic['model_cutouts']:
            if verbose: print '=================================================================================================='
            if verbose: print ' TDOSE: Generate cutouts around sources to extract          '+\
                              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
            tdose.gen_cutouts(setupdic,extractids,sourceids_init,sourcedat_init,
                              performcutout=performcutout,generatesourcecat=generatesourcecat,clobber=clobber,
                              verbose=verbose,verbosefull=verbosefull,start_time=start_time)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print '=================================================================================================='
        if verbose: print ' TDOSE: Defining and loading data for extractions           '+\
                          '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        Nloops    = 1
        loopnames = [-9999]

        if setupdic['model_cutouts']:
            Nloops    = Nextractions
            loopnames = extractids

        for oo, extid in enumerate(loopnames):
            if verbose:
                infostr = ' - Starting extraction for object '+str("%4.f" % (oo+1))+' / '+\
                          str("%4.f" % Nloops)+' with ID = '+str(extid)+'           '+tu.get_now_string()
                if verbosefull:
                    print infostr
                else:
                    sys.stdout.write("%s\r" % infostr)
                    sys.stdout.flush()

            imgstr, imgsize, refimg, datacube, variancecube, sourcecat = tdose.get_datinfo(extid,setupdic)
            if setupdic['wht_image'] is not None:
                refimg    = refimg[0]

            cube_data     = pyfits.open(datacube)[setupdic['cube_extension']].data
            cube_variance = np.sqrt(pyfits.open(variancecube)[setupdic['variance_extension']].data)
            cube_hdr      = pyfits.open(datacube)[setupdic['cube_extension']].header
            cube_wcs2D    = tu.WCS3DtoWCS2D(wcs.WCS(tu.strip_header(cube_hdr.copy())))
            cube_scales   = wcs.utils.proj_plane_pixel_scales(cube_wcs2D)*3600.0
            cube_waves    = np.arange(cube_hdr['NAXIS3'])*cube_hdr['CD3_3']+cube_hdr['CRVAL3']

            img_data      = pyfits.open(refimg)[setupdic['img_extension']].data
            img_hdr       = pyfits.open(refimg)[setupdic['img_extension']].header
            img_wcs       = wcs.WCS(tu.strip_header(img_hdr.copy()))
            img_scales    = wcs.utils.proj_plane_pixel_scales(img_wcs)*3600.0

            modelimg      = setupdic['models_directory']+'/'+\
                            refimg.split('/')[-1].replace('.fits','_'+setupdic['model_image_ext']+'_'+setupdic['source_model']+'.fits')

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbosefull: print '--------------------------------------------------------------------------------------------------'
            if setupdic['ref_image_model'] is None:
                FoV_modelexists = False
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                if setupdic['source_model'] == 'galfit':
                    if verbosefull: print ' Looking for galfit model of source ... ',
                    model_file    = setupdic['galfit_directory']+'galfit_'+\
                                    setupdic['ref_image'].split('/')[-1].replace('.fits','_output.fits')

                    if setupdic['model_cutouts']:
                        model_file = model_file.replace('.fits',imgstr+'.fits')

                    if os.path.isfile(model_file):
                        if verbosefull: print 'found it, so it will be used'
                        FoV_modelexists = True
                        FoV_modelfile   = model_file
                        FoV_modeldata   = pyfits.open(FoV_modelfile)[setupdic['galfit_model_extension']].data
                    else:
                        if verbosefull: print 'did not find it, so will generate gaussian TDOSE model'
                    sys.exit(' ---> Loading parameters and building model from galfit output is not enabled yet; sorry. '
                             'If you have the model try the source_model = modelimg setup')
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                if setupdic['source_model'] == 'modelimg':
                    if verbosefull: print ' Looking for ref_image model of source in "modelimg_directory"... ',
                    model_file    = setupdic['modelimg_directory']+'model_'+\
                                    setupdic['ref_image'].split('/')[-1]
                    if setupdic['model_cutouts']:
                        model_file = model_file.replace('.fits',imgstr+'.fits')

                    if os.path.isfile(model_file):
                        if verbosefull: print 'found it, so it will be used'
                        FoV_modelexists = True
                        FoV_modelfile   = model_file
                        FoV_modeldata   = pyfits.open(FoV_modelfile)[setupdic['modelimg_extension']].data
                    else:
                        if verbosefull: print 'did not find the model\n    '+model_file+'\n   so will skip object '+extid
                        continue
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                if not FoV_modelexists:
                    if verbosefull: print ' TDOSE: Model reference image                               '+\
                                          '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
                    regionfile    = setupdic['models_directory']+'/'+\
                                    refimg.split('/')[-1].replace('.fits','_'+setupdic['model_param_reg']+'_'+setupdic['source_model']+'.reg')
                    modelparam    = modelimg.replace('.fits','_objparam.fits') # output from reference image modeling

                    names         = []
                    sourceids     = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_IDcol']]
                    for ii, sid in enumerate(sourceids):
                        if setupdic['sourcecat_parentIDcol'] is not None:
                            parentid = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_parentIDcol']][ii]
                            namestr  = str(parentid)+'>>'+str(sid)
                        else:
                            namestr  = str(sid)
                        names.append(namestr)

                    if modelrefimage:
                        tdose.model_refimage(setupdic,refimg,img_hdr,sourcecat,modelimg,modelparam,regionfile,img_wcs,img_data,names,
                                             save_init_model_output=save_init_model_output,clobber=clobber,verbose=verbose,
                                             verbosefull=verbosefull)
                    else:
                        if verbose: print ' >>> Skipping modeling reference image (assume models exist)'
            else:
                if verbose: print ' >>> Skipping modeling reference image (model provided in setup file)'
                sys.exit(' ---> Use of the setup parameter ref_image_model is not enabled yet and must be set to "None"; sorry.')
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbosefull: print '--------------------------------------------------------------------------------------------------'
            if verbosefull: print ' TDOSE: Convert ref. image model to cube WCS                '+\
                                  '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
            cubewcsimg       = setupdic['models_directory']+'/'+\
                               refimg.split('/')[-1].replace('.fits','_'+setupdic['model_image_cube_ext']+'_'+
                                                             setupdic['source_model']+'.fits')
            if not FoV_modelexists:
                paramREF      = tu.build_paramarray(modelparam,verbose=verbosefull)
                paramCUBE     = tu.convert_paramarray(paramREF,img_hdr,cube_hdr,type=setupdic['source_model'].lower(),verbose=verbosefull)
            elif FoV_modelexists:
                modelimgsize = model_file
                # pdb.set_trace()
                # sys.exit(' ---> convert model to WCS (pixel scale) of IFU... not enabled yet')

            if refimagemodel2cubewcs:
                cubehdu       = pyfits.PrimaryHDU(cube_data[0,:,:])
                cubewcshdr    = cube_wcs2D.to_header()
                for key in cubewcshdr:
                    if key == 'PC1_1':
                        cubehdu.header.append(('CD1_1',cubewcshdr[key],cubewcshdr[key]),end=True)
                    elif key == 'PC2_2':
                        cubehdu.header.append(('CD2_2',cubewcshdr[key],cubewcshdr[key]),end=True)
                    else:
                        cubehdu.header.append((key,cubewcshdr[key],cubewcshdr[key]),end=True)

                if not FoV_modelexists:
                        modelimgsize = cube_data.shape[1:]
                else:
                    projected_image, footprint = reproject_interp( (FoV_modeldata, img_wcs), cube_wcs2D, shape_out=cube_data.shape[1:])
                    paramCUBE    = projected_image
                    #paramCUBE    = tu.reshape_array(FoV_modeldata,cube_data.shape[1:],pixcombine='sum')

                tmf.save_modelimage(cubewcsimg,paramCUBE,modelimgsize,modeltype=setupdic['source_model'].lower(),
                                    param_init=False,clobber=clobber,outputhdr=cubehdu.header,verbose=verbosefull)
            else:
                if verbose: print ' >>> Skipping converting reference image model to cube WCS frame (assume models exist)'
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbosefull: print '--------------------------------------------------------------------------------------------------'
            if verbosefull: print ' TDOSE: Defining PSF as FWHM = p0 + p1(lambda-7000A)        '+\
                                  '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
            if definePSF or modeldatacube:
                if setupdic['source_model'] == 'aperture':
                    if verbose: print ' >>> Skipping defining PSF as source_model = "aperture", i.e., convolution of ref. image model'
                    paramPSF = None
                else:
                    paramPSF = tdose.define_psf(setupdic,datacube,cube_data,cube_scales,cube_hdr,cube_waves,
                                                clobber=clobber,verbose=verbose,verbosefull=verbosefull)
            else:
                if verbose: print ' >>> Skipping defining PSF of data cube (assume it is defined)'

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbosefull: print '--------------------------------------------------------------------------------------------------'
            if verbosefull: print ' TDOSE: Modelling data cube                                 '+\
                                  '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
            modcubename = setupdic['models_directory']+'/'+\
                          datacube.split('/')[-1].replace('.fits','_'+setupdic['model_cube_ext']+'_'+setupdic['source_model']+'.fits')
            rescubename = setupdic['models_directory']+'/'+\
                          datacube.split('/')[-1].replace('.fits','_'+setupdic['residual_cube_ext']+'_'+setupdic['source_model']+'.fits')

            if modeldatacube:
                tdose.model_datacube(setupdic,extid,modcubename,rescubename,cube_data,cube_variance,paramCUBE,cube_hdr,paramPSF,
                                     clobber=clobber,verbose=verbose,verbosefull=verbosefull)
            else:
                if verbose: print ' >>> Skipping modeling of data cube (assume it exists)'

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            sourcecubename  = setupdic['models_directory']+'/'+\
                              datacube.split('/')[-1].replace('.fits','_'+setupdic['source_model_cube']+'_'+
                                                              setupdic['source_model']+'.fits')

            if createsourcecube:
                if verbosefull: print '--------------------------------------------------------------------------------------------------'
                if verbosefull: print ' TDOSE: Creating source model cube                          '+\
                                      '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
                model_cube        = pyfits.open(modcubename)[setupdic['cube_extension']].data
                layer_scales      = pyfits.open(modcubename)['WAVESCL'].data

                source_model_cube = tmc.gen_source_model_cube(layer_scales,model_cube.shape,paramCUBE,paramPSF,
                                                              paramtype=setupdic['source_model'],
                                                              psfparamtype=setupdic['psf_type'],save_modelcube=True,
                                                              cubename=sourcecubename,clobber=clobber,outputhdr=cube_hdr,
                                                              verbose=verbosefull)

            else:
                if verbose: print ' >>> Skipping generating source model cube (assume it exists)'

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbose: print '=================================================================================================='
            if verbose: print ' TDOSE: Storing extracted 1D spectra to files               '+\
                              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
            specoutputdir      = setupdic['spec1D_directory']
            model_cube_file    = modcubename
            variance_cube_file = variancecube
            variance_cube_ext  = setupdic['variance_extension']
            smc_file           = sourcecubename
            smc_ext            = setupdic['cube_extension']

            SAD    = collections.OrderedDict()

            if extid == -9999:
                if setupdic['sources_to_extract'] == 'all':
                    for ss, sid in enumerate(extractids):
                        SAD[str("%.10d" % int(sid))] = [ss]
                else:
                    if setupdic['sourcecat_parentIDcol'] is not None:
                        parentids = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_parentIDcol']]

                        for ss, sid in enumerate(extractids):
                            sourceent = np.where(sourceids_init == sid)[0]
                            parent    = parentids[sourceent]
                            groupent  = np.where(parentids == parent)[0]
                            SAD       = {str("%.10d" % int(parent))+'-'+str("%.10d" % int(sid)) : groupent.tolist()}
                    else:
                        for ss, sid in enumerate(extractids):
                            SAD[str("%.10d" % int(sid))] = [ss]

            else:
                if setupdic['sourcecat_parentIDcol'] is not None:
                    parentids = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_parentIDcol']]
                    sourceent = np.where(sourceids == extid)[0]
                    parent    = parentids[sourceent][0]
                    groupent  = np.where(parentids == parent)[0]

                    SAD[str("%.10d" % int(parent))+'-'+str("%.10d" % int(extid))] =  groupent.tolist()
                else:
                    sourceent = np.where(sourceids == extid)[0]
                    SAD[str("%.10d" % int(extid))] =  sourceent.tolist()

            if store1Dspectra:
                specfiles  = tes.extract_spectra(model_cube_file,model_cube_ext=setupdic['cube_extension'],
                                                 layer_scale_ext='WAVESCL',clobber=clobber,
                                                 nameext=setupdic['spec1D_name']+'_'+setupdic['source_model'],
                                                 source_association_dictionary=SAD,outputdir=specoutputdir,
                                                 variance_cube_file=variance_cube_file,variance_cube_ext=variance_cube_ext,
                                                 source_model_cube_file=smc_file,source_cube_ext=smc_ext,
                                                 data_cube_file=datacube,verbose=verbosefull)

            else:
                if verbose: print ' >>> Skipping storing 1D spectra to binary fits tables '

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbose: print '=================================================================================================='
            if verbose: print ' TDOSE: Plotting extracted spectra                          '+\
                              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
            if setupdic['plot_generate']:
                tdose.plot_spectra(setupdic,SAD,specoutputdir,plot1Dspectra=plot1Dspectra,plotS2Nspectra=plotS2Nspectra,
                                   verbose=verbosefull)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbose:
                print '=================================================================================================='
                print ' TDOSE: Modeling and extraction done for object '+str(extid)+\
                      '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
                print ' - To open all generated files in DS9 execute the following command '
                ds9cmd  = ' ds9 '
                Nframes = 0
                if os.path.isfile(refimg):
                    ds9cmd  = ds9cmd+refimg+' '
                    Nframes = Nframes + 1

                    if os.path.isfile(setupdic['source_catalog'].replace('.fits','.reg')):
                        ds9cmd = ds9cmd+' -region '+setupdic['source_catalog'].replace('.fits','.reg')+' '

                if os.path.isfile(modelimg):
                    ds9cmd = ds9cmd+modelimg
                    Nframes = Nframes + 1

                    if os.path.isfile(regionfile):
                        ds9cmd = ds9cmd+' -region '+regionfile+' '

                if os.path.isfile(modelimg.replace('.fits','_residual.fits')):
                    ds9cmd = ds9cmd+modelimg.replace('.fits','_residual.fits')+' '
                    Nframes = Nframes + 1

                if os.path.isfile(cubewcsimg):
                    ds9cmd = ds9cmd+cubewcsimg+' '
                    Nframes = Nframes + 1

                if os.path.isfile(datacube):
                    ds9cmd = ds9cmd+datacube+' '
                    Nframes = Nframes + 1

                if os.path.isfile(modcubename):
                    ds9cmd = ds9cmd+modcubename+' '
                    Nframes = Nframes + 1

                if os.path.isfile(rescubename):
                    ds9cmd = ds9cmd+rescubename+' '
                    Nframes = Nframes + 1

                if os.path.isfile(sourcecubename):
                    ds9cmd = ds9cmd+sourcecubename+' '
                    Nframes = Nframes + 1

                ds9cmd = ds9cmd+'-lock frame wcs -tile grid layout '+str(Nframes)+' 1 &'
                print ds9cmd
                print '=================================================================================================='

        if verbose:
            print """
                                                   .''.
                         .''.             *''*    :_\/_:     .
                        :_\/_:   .    .:.*_\/_*   : /\ :  .'.:.'.
                    .''.: /\ : _\(/_  ':'* /\ *  : '..'.  -=:o:=-
                   :_\/_:'.:::. /)\*''*  .|.* '.\'/.'_\(/_'.':'.'
                   : /\ : :::::  '*_\/_* | |  -= o =- /)\    '  *
                    '..'  ':::'   * /\ * |'|  .'/.\'.  '._____
                        *        __*..* |  |     :      |.   |' .---|
                         _*   .-'   '-. |  |     .--'|  ||   | _|   |
                      .-'|  _.|  |    ||   '-__  |   |  |    ||     |
                      |' | |.    |    ||       | |   |  |    ||     |
                   ___|  '-'     '    ""       '-'   '-.'    '`     |____

                     _________   _____       ______     _____     ______
                    |__    __|| |  __ \\\\    /  __  \\\\  / ____\\\\  |  ___||
                       |  ||    | || \ \\\\   | || | ||  \ \\\\__    | ||__
                       |  ||    | || | ||   | || | ||   \__  \\\\  |  __||
                       |  ||    | ||_/ //   | ||_| ||   ___\  \\\\ | ||___
                       |__||    |_____//    \_____//   |______// |_____||

                       _____      ______     __    __    ______    ___
                      |  __ \\\\   /  __  \\\\  |  \\\\ |  ||  |  ___||  |  ||
                      | || \ \\\\  | || | ||  |   \\\\|  ||  | ||__    |  ||
                      | || | ||  | || | ||  |        ||  |  __||   |__||
                      | ||_/ //  | ||_| ||  |  ||\   ||  | ||___    __
                      |_____//   \_____//   |__|| \__||  |_____||  |__||

==================================================================================================
 """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if logterminaloutput:
        setupdic  = tu.load_setup(setupfile,verbose=False)
        outputlog = setupdic['spec1D_directory']+setupfile.split('/')[-1].replace('.txt','_logged_output.txt')

        bufsize = 0
        f = open(outputlog, 'a', bufsize)
        sys.stdout = f
        f.write('\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LOG FROM RUN STARTED ON '+tu.get_now_string()+
                ' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n')

        tdosefunction(setupfile,performcutout,generatesourcecat,modelrefimage,refimagemodel2cubewcs,
                      definePSF,modeldatacube,createsourcecube,store1Dspectra,plot1Dspectra,
                      plotS2Nspectra,save_init_model_output,clobber,verbose,verbosefull)

        sys.stdout = sys.__stdout__
        f.close()
    else:
        tdosefunction(setupfile,performcutout,generatesourcecat,modelrefimage,refimagemodel2cubewcs,
                      definePSF,modeldatacube,createsourcecube,store1Dspectra,plot1Dspectra,
                      plotS2Nspectra,save_init_model_output,clobber,verbose,verbosefull)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_cutouts(setupdic,extractids,sourceids_init,sourcedat_init,
                performcutout=True,generatesourcecat=True,clobber=False,verbose=True,verbosefull=True,start_time=0.0):
    """
    Generate cutouts of reference image and data cube

    --- INPUT ---
    setupdic              Dictionary containing the setup parameters read from the TDOSE setup file
    extractids            IDs of objects to extract spectra for
    sourceids_init        The initial source IDs
    sourcedat_init        The initial source data
    performcutout         Set to true to actually perform cutouts.
    generatesourcecat     To generate a (sub) source catalog corresponding to the objects in the cutout
    clobber               Overwrite existing files if they exist
    verbose               Toggle verbosity
    verbosefull           Toggle extended verbosity
    start_time            Start time of wrapper cutout generation is embedded in

    """
    Nextractions = len(extractids)
    cut_images = []
    cut_cubes  = []
    for oo, cutoutid in enumerate(extractids):
        if verbose:
            infostr = ' - Cutting out object '+str("%4.f" % (oo+1))+' / '+\
                      str("%4.f" % Nextractions)+' with ID = '+str(cutoutid)+'           '+tu.get_now_string()
            if verbosefull:
                print infostr
            else:
                sys.stdout.write("%s\r" % infostr)
                sys.stdout.flush()

        objent = np.where(sourceids_init == cutoutid)[0]
        if len(objent) != 1:
            sys.exit(' ---> More than one (or no) match in source catalog to ID '+str(cutoutid))

        ra          = sourcedat_init[setupdic['sourcecat_racol']][objent]
        dec         = sourcedat_init[setupdic['sourcecat_deccol']][objent]


        cutstr, cutoutsize, cut_img, cut_cube, cut_variance, cut_sourcecat = tdose.get_datinfo(cutoutid,setupdic)

        if setupdic['wht_image'] is None:
            imgfiles = [setupdic['ref_image']]
            imgexts  = [setupdic['img_extension']]
            cut_images.append(cut_img)
        else:
            imgfiles = [setupdic['ref_image'],setupdic['wht_image']]
            imgexts  = [setupdic['img_extension'],setupdic['wht_extension']]
            cut_images.append(cut_img[0])



        if performcutout:
            if setupdic['data_cube'] == setupdic['variance_cube']:
                cutouts   = tu.extract_subcube(setupdic['data_cube'],ra,dec,cutoutsize,cut_cube,
                                               cubeext=[setupdic['cube_extension'],setupdic['variance_extension']],clobber=clobber,
                                               imgfiles=imgfiles,imgexts=imgexts,
                                               imgnames=cut_img,verbose=verbosefull)
            else:
                cutouts   = tu.extract_subcube(setupdic['data_cube'],ra,dec,cutoutsize,cut_cube,
                                               cubeext=[setupdic['cube_extension']],clobber=clobber,
                                               imgfiles=imgfiles,imgexts=imgexts,
                                               imgnames=cut_img,verbose=verbosefull)

                cutouts   = tu.extract_subcube(setupdic['variance_cube'],ra,dec,cutoutsize,cut_variance,
                                               cubeext=[setupdic['variance_extension']],clobber=clobber,
                                               imgfiles=None,imgexts=None,imgnames=None,verbose=verbosefull)
        else:
            if verbose: print ' >>> Skipping cutting out images and cubes (assuming they exist)                                 '

        # --- SUB-SOURCE CAT ---
        if generatesourcecat:
            obj_in_cut_fov = np.where( (sourcedat_init[setupdic['sourcecat_racol']] < (ra + cutoutsize[0]/2./3600.)) &
                                       (sourcedat_init[setupdic['sourcecat_racol']] > (ra - cutoutsize[0]/2./3600.)) &
                                       (sourcedat_init[setupdic['sourcecat_deccol']] < (dec + cutoutsize[1]/2./3600.)) &
                                       (sourcedat_init[setupdic['sourcecat_deccol']] > (dec - cutoutsize[1]/2./3600.)) )[0]
            Ngoodobj      = len(obj_in_cut_fov)
            cutout_hdr    = pyfits.open(cut_images[oo])[setupdic['img_extension']].header
            cut_sourcedat = sourcedat_init[obj_in_cut_fov].copy()
            storearr      = np.zeros(Ngoodobj,dtype=cut_sourcedat.columns) # define structure array to store to fits file
            for ii in np.arange(Ngoodobj):
                striphdr   = tu.strip_header(cutout_hdr.copy())
                wcs_in     = wcs.WCS(striphdr)
                skycoord   = SkyCoord(cut_sourcedat[ii][setupdic['sourcecat_racol']],
                                      cut_sourcedat[ii][setupdic['sourcecat_deccol']], frame='icrs', unit='deg')
                pixcoord   = wcs.utils.skycoord_to_pixel(skycoord,wcs_in)
                cut_sourcedat[ii][setupdic['sourcecat_xposcol']] = pixcoord[0]
                cut_sourcedat[ii][setupdic['sourcecat_yposcol']] = pixcoord[1]

                storearr[ii] = np.vstack(cut_sourcedat)[ii,:]

            astropy.io.fits.writeto(cut_sourcecat,storearr,header=None,clobber=clobber)
        else:
            if verbose: print ' >>> Skipping generating the cutout source catalogs (assume they exist)'

    if not verbosefull:
        if verbose: print '\n   done'

    if verbose:
        print '=================================================================================================='
        print ' TDOSE: Done cutting out sub cubes and postage stamps       '+\
              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        print ' - To open resulting images in DS9 execute the following command '
        ds9string = ' ds9 '+setupdic['ref_image']+' xxregionxx '+\
                    ' '.join(cut_images)+' -lock frame wcs -tile grid layout '+str(len(cut_images)+1)+' 1 &'
        regname   = setupdic['source_catalog'].replace('.fits','.reg')
        if os.path.isfile(regname):
            ds9string = ds9string.replace('xxregionxx',' -region '+regname+' ')
        else:
            ds9string = ds9string.replace('xxregionxx',' ')
        print ds9string
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_refimage(setupdic,refimg,img_hdr,sourcecat,modelimg,modelparam,regionfile,img_wcs,img_data,names,
                   save_init_model_output=True,clobber=True,verbose=True,verbosefull=True):
    """
    Modeling the refernce image

    --- INPUT ---
    setupdic                Dictionary containing the setup parameters read from the TDOSE setup file
    refimg                  Name of fits reference image to model
    img_hdr                 Fits header of of reference image
    sourcecat               Source catalog providing coordinates of objects in reference image to model
    modelimg                Name of output file to store model to
    modelparam              Fits table to contain the model parameters (which will be turned into a DS9 region file)
    regionfile              The name of the reegionfile to generate with model parameter regions
    img_wcs                 WCS of image to model
    img_data                Data of image array
    names                   Names of individual objects used in DS9 region
    save_init_model_output  Set to true to save the initial model to files
    clobber                 Overwrite files if the already exist
    verbose                 Toggle verbosity
    verbosefull             Toggle extended verbosity

    """
    if setupdic['source_model'].lower() == 'gauss':
        sigysigxangle = None
        fluxscale     = setupdic['sourcecat_fluxcol']
        if setupdic['gauss_guess'] is None:
            param_initguess = None
        else:
            objects   = pyfits.open(sourcecat)[1].data['id'].tolist()

            if save_init_model_output:
                saveDS9region = True
                savefitsimage = True
                savefitstable = True
                ds9regionname = refimg.replace('.fits','_tdose_initial_model_ds9region.reg')
                fitsimagename = refimg.replace('.fits','_tdose_initial_model_image.fits')
                fitstablename = refimg.replace('.fits','_tdose_initial_model_objparam.fits')
            else:
                saveDS9region = False
                savefitsimage = False
                savefitstable = False
                ds9regionname = ' '
                fitsimagename = ' '
                fitstablename = ' '

            paramlist = tu.gen_paramlist_from_SExtractorfile(setupdic['gauss_guess'],imgheader=img_hdr,clobber=clobber,
                                                             objects=objects,
                                                             idcol=setupdic['gauss_guess_idcol'],
                                                             racol=setupdic['gauss_guess_racol'],
                                                             deccol=setupdic['gauss_guess_deccol'],
                                                             aimg=setupdic['gauss_guess_aimg'],
                                                             bimg=setupdic['gauss_guess_bimg'],
                                                             angle=setupdic['gauss_guess_angle'],
                                                             fluxscale=setupdic['gauss_guess_fluxscale'],
                                                             fluxfactor=setupdic['gauss_guess_fluxfactor'],
                                                             Nsigma=setupdic['gauss_guess_Nsigma'],
                                                             verbose=verbosefull,
                                                             saveDS9region=saveDS9region,ds9regionname=ds9regionname,
                                                             savefitsimage=savefitsimage,fitsimagename=fitsimagename,
                                                             savefitstable=savefitstable,fitstablename=fitstablename)
            param_initguess = paramlist
    elif setupdic['source_model'].lower() == 'galfit':
        sys.exit(' ---> source_model == galfit is not enabled yet; sorry...')
    elif setupdic['source_model'].lower() == 'aperture':
        sigysigxangle    = None
        param_initguess  = None
        pixscales        = wcs.utils.proj_plane_pixel_scales(img_wcs)*3600.0
        pixscaleunique   = np.unique(pixscales)
        if len(pixscaleunique) != 1:
            sys.exit(' ---> The pixel scale in the x and y direction of image are different')
        else:
            sigysigxangle =  setupdic['aperture_size'] / pixscaleunique          # radius in pixels
            fluxscale     =  pyfits.open(sourcecat)[1].data['id'].astype(float)  # pixel values
    else:
        sys.exit(' ---> Setting source_model == '+setupdic['source_model']+' is not a valid entry')


    pinit, fit    = tmf.gen_fullmodel(img_data,sourcecat,modeltype=setupdic['source_model'],verbose=verbosefull,
                                      xpos_col=setupdic['sourcecat_xposcol'],ypos_col=setupdic['sourcecat_yposcol'],
                                      datanoise=None,sigysigxangle=sigysigxangle,
                                      fluxscale=fluxscale,generateimage=modelimg,
                                      generateresidualimage=True,clobber=clobber,outputhdr=img_hdr,
                                      param_initguess=param_initguess)

    tu.model_ds9region(modelparam,regionfile,img_wcs,color='cyan',width=2,Nsigma=2,textlist=names,
                       fontsize=12,clobber=clobber)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_datacube(setupdic,extid,modcubename,rescubename,cube_data,cube_variance,paramCUBE,cube_hdr,paramPSF,
                   clobber=False,verbose=True,verbosefull=True):
    """
    Modeling the data cube

    --- INPUT ---
    setupdic                Dictionary containing the setup parameters read from the TDOSE setup file
    extid                   ID of cube to model
    modcubename             Name of model cube to generate
    rescubename             Name of residual cube to generate
    cube_data               Data cube
    cube_variance           Variance for data cube (cube_data)
    paramCUBE               Parmaters of objects in data cube
    cube_hdr                Header of data cube
    paramPSF                Parameters of PSF
    clobber                 Overwrite files if they exist
    verbose                 Toggle verbosity
    verbosefull             Toggle extended verbosity

    """
    if setupdic['model_cube_layers'] == 'all':
        layers = None
    elif type(setupdic['model_cube_layers']) == list:
        layers = np.arange(setupdic['model_cube_layers'][0],setupdic['model_cube_layers'][1]+1,1)
    else:
        layerinfo = np.genfromtxt(setupdic['model_cube_layers'],dtype=None,comments='#')

        try:
            layer_ids            = layerinfo[:,0].astype(float)
            structuredlayerarray = False
        except:
            layer_ids            = layerinfo['f0'].astype(float)
            structuredlayerarray = True

        objent = np.where(layer_ids == extid)[0]

        if len(objent) > 1:
            sys.exit(' ---> More than one match in '+setupdic['model_cube_layers']+' for object '+str(extid))
        elif len(objent) == 0:
            sys.exit(' ---> No match in '+setupdic['model_cube_layers']+' for object '+str(extid))
        else:
            if structuredlayerarray:
                if layerinfo['f1'][objent] == 'all':
                    layers = None
                else:
                    layers = [int(layerinfo['f1'][objent]),int(layerinfo['f2'][objent])]
                    layers = np.arange(layers[0],layers[1]+1,1)
            else:
                layers = layerinfo[objent,1:][0].astype(float).tolist()
                layers = np.arange(layers[0],layers[1]+1,1)

    optimizer    = setupdic['model_cube_optimizer']
    paramtype    = setupdic['source_model']
    psfparamtype = setupdic['psf_type']

    cube_noise = np.sqrt(cube_variance) # turn variance cube into standard deviation
    cube_model, layer_scales = tmc.gen_fullmodel(cube_data,paramCUBE,paramPSF,paramtype=paramtype,
                                                 psfparamtype=psfparamtype,noisecube=cube_noise,save_modelcube=True,
                                                 cubename=modcubename,clobber=clobber,
                                                 fit_source_scales=True,outputhdr=cube_hdr,verbose=verbosefull,
                                                 returnresidual=rescubename,optimize_method=optimizer,model_layers=layers)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def define_psf(setupdic,datacube,cube_data,cube_scales,cube_hdr,cube_waves,clobber=False,verbose=True,verbosefull=True):
    """
    Defining the PSF model to convolve reference image with

    --- INPUT ---
    setupdic                Dictionary containing the setup parameters read from the TDOSE setup file
    datacube                Name of data used for printing.
    cube_data               Data from datacube to base PSF cube dimensions on
    cube_scales             The pixel scale of the data cube
    cube_hdr                The data cube fits header
    cube_waves              The wavelengths corresponding to the layers of the data cube
    clobber                 Overwrite files if they exist
    verbose                 Toggle verbosity
    verbosefull             Toggle extended verbosity

    """
    if setupdic['psf_FWHM_evolve'].lower() == 'linear':
        fwhm_p0     = setupdic['psf_FWHMp0']
        fwhm_p1     = setupdic['psf_FWHMp1']
        fwhm_vec    = fwhm_p0 + fwhm_p1 * (cube_waves - 7000.0)
        sigmas      = fwhm_vec/2.35482/cube_scales[0]
    else:
        sys.exit(' ---> '+setupdic['psf_FWHM_evolve']+' is an invalid choice for the psf_FWHM_evolve setup parameter ')

    if setupdic['psf_type'].lower() == 'gauss':
        xpos,ypos,fluxscale,angle = 0.0, 0.0, 1.0, 0.0
        paramPSF                  = []
        for layer in np.arange(cube_data.shape[0]):
            sigma = sigmas[layer]
            paramPSF.append([xpos,ypos,fluxscale,sigma,sigma,angle])
        paramPSF  = np.asarray(paramPSF)
    else:
        sys.exit(' ---> '+setupdic['psf_type']+' is an invalid choice for the psf_type setup parameter ')

    if setupdic['psf_savecube']:
        psfcubename = setupdic['models_directory']+'/'+datacube.split('/')[-1].replace('.fits','_tdose_psfcube.fits')
        if verbose: print ' - Storing PSF cube to fits file \n   '+psfcubename

        if os.path.isfile(psfcubename) & (clobber == False):
            if verbose: print ' ---> TDOSE WARNING: PSF cube already exists and clobber = False so skipping step'
        else:
            psfcube = cube_data*0.0
            for ll in np.arange(len(cube_waves)):
                if verbosefull:
                    infostr = '   Building PSF in layer '+str("%6.f" % (ll+1))+' / '+str("%6.f" % len(cube_waves))+''
                    sys.stdout.write("%s\r" % infostr)
                    sys.stdout.flush()

                if setupdic['psf_type'].lower() == 'gauss':
                    mu_psf    = paramPSF[ll][0:2]
                    cov_psf   = tu.build_2D_cov_matrix(paramPSF[ll][4],paramPSF[ll][3],paramPSF[ll][5],verbose=False)
                    psfimg    = tu.gen_2Dgauss(np.asarray(cube_data.shape[1:]).tolist(),cov_psf,1.0,
                                               show2Dgauss=False,verbose=False)
                else:
                    sys.exit(' ---> '+setupdic['psf_type']+' is an invalid choice for the psf_type setup parameter ')

                psfcube[ll,:,:] = psfimg

            if 'XTENSION' in cube_hdr.keys():
                hduprim        = pyfits.PrimaryHDU()  # default HDU with default minimal header
                hducube        = pyfits.ImageHDU(psfcube,header=cube_hdr)
                hducube.header.append(('PSF_P0',    setupdic['psf_FWHMp0'],' '),end=True)
                hducube.header.append(('PSF_P1',    setupdic['psf_FWHMp1'],' '),end=True)
                hdus           = [hduprim,hducube]
            else:
                hducube = pyfits.PrimaryHDU(psfcube,header=cube_hdr)
                hducube.header.append(('PSF_P0',    setupdic['psf_FWHMp0'],' '),end=True)
                hducube.header.append(('PSF_P1',    setupdic['psf_FWHMp1'],' '),end=True)
                hdus           = [hducube]

            hdulist = pyfits.HDUList(hdus)
            hdulist.writeto(psfcubename,clobber=clobber)

    return paramPSF

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def plot_spectra(setupdic,SAD,specoutputdir,plot1Dspectra=True,plotS2Nspectra=True,verbose=True):
    """

    --- INPUT ---
    setupdic                Dictionary containing the setup parameters read from the TDOSE setup file
    SAD                     Source association dictionary difining what sources should be combined into objects
                            (spectra) when plotting.
    specoutputdir           Directory to store plots in
    plot1Dspectra           Plot 1D spectra?
    plotS2Nspectra          Plot signal-to-noise spectra of the 1D spectra?
    verbose                 Toggle verbosity

    """
    showspec        = False

    for key in SAD.keys():
        spec = specoutputdir+setupdic['spec1D_name']+'_'+setupdic['source_model']+'_'+key+'.fits'
        id   = spec.split('_')[-1].split('.')[0]

        if plot1Dspectra:
            xrange = setupdic['plot_1Dspec_xrange']
            yrange = setupdic['plot_1Dspec_yrange']

            tes.plot_1Dspecs([spec],colors=['green'],labels=[id],plotSNcurve=False,
                             plotname=spec.replace('.fits','_'+setupdic['plot_1Dspec_ext']+'.pdf'),showspecs=showspec,
                             shownoise=setupdic['plot_1Dspec_shownoise'],xrange=xrange,yrange=yrange,
                             comparisonspecs=None,comp_colors=['dummy'],comp_labels=['dummy'],
                             comp_wavecol='dummy',comp_fluxcol='dummy',comp_errcol='dummy')
        else:
            if verbose: print ' >>> Skipping plotting 1D spectra '

        if plotS2Nspectra:
            xrange = setupdic['plot_S2Nspec_xrange']
            yrange = setupdic['plot_S2Nspec_yrange']

            tes.plot_1Dspecs([spec],colors=['green'],labels=[id],plotSNcurve=True,
                             plotname=spec.replace('.fits','_'+setupdic['plot_S2Nspec_ext']+'.pdf'),showspecs=showspec,
                             shownoise='dummy',xrange=xrange,yrange=yrange,
                             comparisonspecs=None,comp_colors=['dummy'],comp_labels=['dummy'],
                             comp_wavecol='dummy',comp_fluxcol='dummy',comp_errcol='dummy')
        else:
            if verbose: print ' >>> Skipping plotting S/N spectra '

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def modify_cube(modifysetupfile='./tdose_setup_template_modify.txt',verbose=True):
    """
    Wrapper for modyfying data cube based on sourcemodelcube

    --- INPUT ---
    modifysetupfile    Setup file for modifying the data cubes based on the source model cubes
    verbose            Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose
    tdose.modify_cube(modifysetupfile='./tdose_setup_template_modify.txt',verbose=True)

    tdose.modify_cube(modifysetupfile='/Users/kschmidt/work/TDOSE/tdose_setup_candels-cdfs-02_modify.txt',verbose=True)

    """
    if verbose: print '====================== TDOSE: Modifying data cube by removing objects ======================'
    setupdic        = tu.load_setup(modifysetupfile,verbose=verbose)

    datacube        = setupdic['data_cube']
    sourcemodelcube = setupdic['source_model_cube']
    modcubename     = datacube.replace('.fits','_'+setupdic['modyified_cube']+'.fits')

    if type(setupdic['modify_sources_list']) == str:
        modellist    = 'loop over ids in input file list'
        sys.exit(' ---> input sources to modify via list in file not enabled ')
    else:
        modellist    = [int(sourcenumber) for sourcenumber in setupdic['modify_sources_list']]

    if setupdic['sources_action'].lower() == 'keep':
        remkey = False
    else:
        remkey = True

    modified_cube   = tmoc.remove_object(datacube,sourcemodelcube,objects=modellist,remove=remkey,
                                         dataext=setupdic['cube_extension'],sourcemodelext=setupdic['source_extension'],
                                         savecube=modcubename,verbose=verbose)

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
