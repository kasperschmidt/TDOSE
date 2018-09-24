# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import pyfits
import pdb
import time
import os
import sys
import glob
import numpy as np
import collections
import astropy
import shutil
import collections
import multiprocessing
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units
import astropy.convolution
from astropy.nddata import Cutout2D
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
                       logterminaloutput=False,skipextractedobjects=False,skipspecificobjects=None):
    """
    Perform extraction of spectra from data cube based on information in TDOSE setup file

    --- INPUT ---
    setupfile              TDOSE setup file. Template can be generated with tu.generate_setup_template()
    performcutout          To skip cutting out subcubes and images (i.e., if the cutouts have already been
                           genereated and exist) set performcutout=False
    generatesourcecat      To skip generating the cutout source catalogs from the main source catalog of sources
                           to model (e.g., after editing the source catalog) set generatesourcecat=False
                           Note however, that these catalogs are needed to produce the full FoV source model cube with
                           tdose.gen_fullFoV_from_cutouts()
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
    skipextractedobjects   To skip modeling and extraction of objects which were already extracted, i.e. object IDs with
                           a matching 'spec1D_name'*.fits file in the 'spec1D_directory', set this keyword to True.
                           NB This keyword does not apply to the cutouts; to ignore this process use the
                              performcutout and generatesourcecat keywords.
                           NB Note that spectra extracted with parent ids will not be recognized and therefore skipped.
                              Hence only a standard TDOSE extraction will work in combinations with skipextractedobjects
                              However, this generates all nescessary models and files for post-modeling parent extractions.
    skipspecificobjects    In addition to skipextractedobjects to skip specific objects (irrespective of whether they
                           have already been extracted or not) provide a list of source IDs to this keyword.
                           The same causions mentioned under skipextractedobjects applies to skipspecificobjects as well.

    --- EXAMPLE OF USE ---
    import tdose

    # full extraction with minimal text output to prompt
    tdose.perform_extraction(setupfile='./tdose_setup_candels-cdfs-02.txt',verbose=True,verbosefull=False)

    # only plotting:
    tdose.perform_extraction(setupfile='./tdose_setup_candels-cdfs-02.txt',performcutout=False,generatesourcecat=False,modelrefimage=False,refimagemodel2cubewcs=False,definePSF=False,modeldatacube=False,createsourcecube=False,store1Dspectra=False,plot1Dspectra=True,clobber=True,verbosefull=False)


    """
    # defining function within the routine that can be called by the output logger at the end
    def tdosefunction(setupfile,performcutout,generatesourcecat,modelrefimage,refimagemodel2cubewcs,
                       definePSF,modeldatacube,createsourcecube,store1Dspectra,plot1Dspectra,
                       plotS2Nspectra,save_init_model_output,clobber,verbose,verbosefull,skipextractedobjects,skipspecificobjects):
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
        loopnames = [-9999] # Default: Extracting objects from full FoV.
                            # If to be done in cutouts, loopnames will be replaced with individual IDs

        if setupdic['model_cutouts']:
            Nloops    = Nextractions
            loopnames = extractids

        if skipspecificobjects is None:
            skipidlist = []
        else:
            skipidlist = [int(id) for id in skipspecificobjects]

        for oo, extid in enumerate(loopnames):
            if verbose:
                infostr = ' - Starting extraction for object '+str("%4.f" % (oo+1))+' / '+\
                          str("%4.f" % Nloops)+' with ID = '+str(extid)+'           '+tu.get_now_string()

            start_time_obj = time.clock()
            imgstr, imgsize, refimg, datacube, variancecube, sourcecat = tu.get_datinfo(extid,setupdic)

            skipthisobj = False
            if skipextractedobjects or (int(extid) in skipidlist):
                if int(extid) in skipidlist:
                    skipthisobj = True
                    infostr = infostr+'  -> skipping per request           '
                else:
                    nameext2check = setupdic['spec1D_name']+'_'+setupdic['source_model']
                    id2check      = str("%.10d" % extid)
                    specdir2check = setupdic['spec1D_directory']
                    file2check    = specdir2check+nameext2check+'_'+id2check+'.fits'
                    if os.path.isfile(file2check):
                        skipthisobj = True
                        infostr = infostr+'  -> skipping as spectrum exists    '
                    else:
                        infostr = infostr+'                                            '

            if verbose:
                if verbosefull:
                    print infostr
                else:
                    sys.stdout.write("%s\r" % infostr)
                    sys.stdout.flush()

            if skipthisobj:
                continue

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

                cube_model_file = model_file.replace('.fits','_cube.fits')
                if os.path.isfile(cube_model_file):
                    if verbosefull: print 'found a cube model, so will use that (instead of any model files)'
                    FoV_modelexists     = True
                    FoV_modelfile       = cube_model_file
                    FoV_modeldata       = pyfits.open(FoV_modelfile)[setupdic['modelimg_extension']].data
                elif os.path.isfile(model_file):
                    if verbosefull: print 'found it, so it will be used'
                    FoV_modelexists     = True
                    FoV_modelfile       = model_file
                    FoV_modeldata       = pyfits.open(FoV_modelfile)[setupdic['modelimg_extension']].data
                else:
                    if verbosefull: print 'did not find any model or cube model:\n    '+model_file+'\n    '+cube_model_file+\
                                          '\n   so will skip object '+str(extid)
                    continue

                try:
                    if FoV_modeldata == None:
                        print('\n WARNING - No model data found in extension '+
                              str(setupdic['modelimg_extension'])+' of '+FoV_modelfile+'\n')
                except:
                    pass
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

                if setupdic['nondetections'] is None:
                    centralpointsource = False
                elif type(setupdic['nondetections']) == np.str_ or (type(setupdic['nondetections']) == str):
                    if setupdic['nondetections'].lower() == 'all':
                        centralpointsource = True
                    else:
                        nondetids = np.genfromtxt(setupdic['nondetections'],dtype=None,comments='#')
                        nondetids = list(nondetids.astype(float))
                        if extid in nondetids:
                            centralpointsource = True
                else:
                    if extid in setupdic['nondetections']:
                        centralpointsource = True

                if centralpointsource:
                    if verbosefull: print ' - Object in list of non-detections. Adjusting model to contain central point source  '

                if modelrefimage:
                    tdose.model_refimage(setupdic,refimg,img_hdr,sourcecat,modelimg,modelparam,regionfile,img_wcs,img_data,names,
                                         save_init_model_output=save_init_model_output,centralpointsource=centralpointsource,
                                         clobber=clobber,verbose=verbose,verbosefull=verbosefull)
                else:
                    if verbose: print ' >>> Skipping modeling reference image (assume models exist)'
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
                    if len(FoV_modeldata.shape) == 2:
                        FoV_modeldata_reproject = FoV_modeldata
                    elif len(FoV_modeldata.shape) == 3:
                        FoV_modeldata_reproject = np.sum(FoV_modeldata, axis=0)
                    else:
                        sys.exit(' ---> Shape of model data array is not 2 (image) or 3 (cube) ')

                    projected_image, footprint = reproject_interp( (FoV_modeldata_reproject, img_wcs), cube_wcs2D,
                                                                   shape_out=cube_data.shape[1:])
                    projected_image[np.isnan(projected_image)] = 0.0 # replacing NaNs from reprojection with 0s
                    paramCUBE  = projected_image/np.sum(projected_image)*np.sum(FoV_modeldata) # normalize and scale to match FoV_modeldata

                tmf.save_modelimage(cubewcsimg,paramCUBE,modelimgsize,modeltype=setupdic['source_model'].lower(),
                                    param_init=False,clobber=clobber,outputhdr=cubehdu.header,verbose=verbosefull)

                if FoV_modelexists:
                    if (len(FoV_modeldata.shape) == 3):
                        paramCUBE = np.zeros([FoV_modeldata.shape[0],cube_data.shape[1],cube_data.shape[2]])
                        if verbose: print ' - Reprojecting and normalizing individual components in object model cube to use for extraction '
                        for component in xrange(FoV_modeldata.shape[0]):
                            projected_comp, footprint_comp = reproject_interp( (FoV_modeldata[component,:,:], img_wcs), cube_wcs2D,
                                                                               shape_out=cube_data.shape[1:])
                            projected_comp[np.isnan(projected_comp)] = 0.0
                            paramCUBE[component,:,:] = projected_comp/ np.sum(projected_comp)
            else:
                if verbose: print ' >>> Skipping converting reference image model to cube WCS frame (assume models exist)'
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbosefull: print '--------------------------------------------------------------------------------------------------'
            if verbosefull: print ' TDOSE: Defining PSF as FWHM = p0 + p1(lambda-'+str(setupdic['psf_FWHMp2'])+'A)        '+\
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

            psfcubename  = setupdic['models_directory']+'/'+datacube.split('/')[-1].replace('.fits','_tdose_psfcube_'+
                                                                                            setupdic['source_model']+'.fits')
            if modeldatacube:
                tdose.model_datacube(setupdic,extid,modcubename,rescubename,cube_data,cube_variance,paramCUBE,cube_hdr,paramPSF,
                                     psfcubename=psfcubename,clobber=clobber,verbose=verbose,verbosefull=verbosefull)
            else:
                if verbose: print ' >>> Skipping modeling of data cube (assume it exists)'

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            sourcecubename  = setupdic['models_directory']+'/'+\
                              datacube.split('/')[-1].replace('.fits','_'+setupdic['source_model_cube_ext']+'_'+
                                                              setupdic['source_model']+'.fits')

            if createsourcecube:
                if verbosefull: print '--------------------------------------------------------------------------------------------------'
                if verbosefull: print ' TDOSE: Creating source model cube                          '+\
                                      '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
                model_cube        = pyfits.open(modcubename)[setupdic['cube_extension']].data
                layer_scales      = pyfits.open(modcubename)['WAVESCL'].data
                if setupdic['source_model'].lower() != 'aperture':
                    psfcube       = pyfits.open(psfcubename)[setupdic['cube_extension']].data
                else:
                    psfcube       = None

                source_model_cube = tmc.gen_source_model_cube(layer_scales,model_cube.shape,paramCUBE,paramPSF,
                                                              psfcube=psfcube,paramtype=setupdic['source_model'],
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

            # - - - - - - - - - - - - Putting together source association dictionary - - - - - - - - - - - - -
            SAD    = collections.OrderedDict()

            if FoV_modelexists:
                sourceids = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_IDcol']]

            if extid == -9999: # If full FoV is modeled
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

            else:  # If cutouts are modeled instead of full FoV
                if setupdic['sourcecat_parentIDcol'] is not None:
                    parentids = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_parentIDcol']]
                    sourceent = np.where(sourceids == extid)[0]
                    parent    = parentids[sourceent][0]
                    if float(parent) < 0: # ignoring parents with negative IDs. Looking for object IDs in parent list instead
                        sourceent = np.where(parentids == extid)[0]
                        parent    = parentids[sourceent][0]

                    groupent  = np.where(parentids == parent)[0]
                    SAD[str("%.10d" % int(parent))+'-'+str("%.10d" % int(extid))] =  groupent.tolist()
                else:
                    sourceent = np.where(sourceids == extid)[0]
                    SAD[str("%.10d" % int(extid))] =  sourceent.tolist()
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
            if verbose: print ' TDOSE: Plotting extracted spectra                    '+\
                              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
            if setupdic['plot_generate']:
                tdose.plot_spectra(setupdic,SAD,specoutputdir,plot1Dspectra=plot1Dspectra,plotS2Nspectra=plotS2Nspectra,
                                   verbose=verbosefull)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbose:
                print '=================================================================================================='
                print ' TDOSE: Modeling and extraction done for object '+str(extid)+\
                      '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )       '
                print '                                                       '+\
                      '   --> Object runtime = '+str("%10.4f" % (time.clock() - start_time_obj))+' seconds <--   '
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
                      plotS2Nspectra,save_init_model_output,clobber,verbose,verbosefull,
                      skipextractedobjects,skipspecificobjects)

        sys.stdout = sys.__stdout__
        f.close()
    else:
        tdosefunction(setupfile,performcutout,generatesourcecat,modelrefimage,refimagemodel2cubewcs,
                      definePSF,modeldatacube,createsourcecube,store1Dspectra,plot1Dspectra,
                      plotS2Nspectra,save_init_model_output,clobber,verbose,verbosefull,
                      skipextractedobjects,skipspecificobjects)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def perform_extractions_in_parallel(setupfiles,Nsessions=0,verbose=True,generateFullFoVmodel=True,generateOverviewPlots=True,
                                    # - - - - - - - Inputs passed to tdose.perform_extraction() - - - - - - -
                                    performcutout=True,generatesourcecat=True,modelrefimage=True,refimagemodel2cubewcs=True,
                                    definePSF=True,modeldatacube=True,createsourcecube=True,store1Dspectra=True,plot1Dspectra=True,
                                    plotS2Nspectra=True,save_init_model_output=False,clobber=False,verbosePE=True,verbosefull=False,
                                    logterminaloutput=True,skipextractedobjects=False,skipspecificobjects=None):
    """
    Run multiple TDOSE setups in parallel

    --- INPUT ---
    setupfiles               List of setup files to run in parallel
    Nsessions                The number of parallel sessions to launch (the list of setupfiles will be bundled up in
                             Nsessions bundles to run). The default is 0 which will run Nsetupfiles sessions with
                             1 setup file per parallel session.
    verbose                  Toggle verbosity
    generateFullFoVmodel     Combine cutouts (if the run is based on cutouts) into full FoV model cube with
                             tdose.gen_fullFoV_from_cutouts()
    generateOverviewPlots    Generate overview plots of each of the extracted objects with tu.gen_overview_plot()

    **remaining input**      Input passed to tdose.perform_extraction();
                             see tdose.perform_extraction() header for details

    --- EXAMPLE OF USE ---
    import tdose, glob
    setupfiles           = ['setup01','setup02','setup03','setup04','setup05','setup06','setup07','setup08','setup09']
    setupfiles           = glob.glob('/Users/kschmidt/work/TDOSE/tdose_setup_candels-cdfs-*[0-99].txt')
    bundles, paralleldic = tdose.perform_extractions_in_parallel(setupfiles,Nsessions=2,clobber=True,performcutout=False,store1Dspectra=False,plot1Dspectra=False,generateFullFoVmodel=False,generateOverviewPlots=True,skipextractedobjects=True,logterminaloutput=True)

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def parallel_worker(setupfiles,performcutout,generatesourcecat,modelrefimage,refimagemodel2cubewcs,definePSF,
                        modeldatacube,createsourcecube,store1Dspectra,plot1Dspectra,plotS2Nspectra,
                        save_init_model_output,clobber,verbose,verbosefull,logterminaloutput,
                        generateFullFoVmodel=True,generateOverviewPlots=True,skipextractedobjects=False,skipspecificobjects=None):
        """
        Multiprocessing worker function
        """
        for setupfile in setupfiles:
            tdose.perform_extraction(setupfile=setupfile,performcutout=performcutout,generatesourcecat=generatesourcecat,
                                     modelrefimage=modelrefimage,refimagemodel2cubewcs=refimagemodel2cubewcs,
                                     definePSF=definePSF,modeldatacube=modeldatacube,createsourcecube=createsourcecube,
                                     store1Dspectra=store1Dspectra,plot1Dspectra=plot1Dspectra,plotS2Nspectra=plotS2Nspectra,
                                     save_init_model_output=save_init_model_output,clobber=clobber,
                                     verbose=verbose,verbosefull=verbosefull,logterminaloutput=logterminaloutput,
                                     skipextractedobjects=skipextractedobjects,skipspecificobjects=skipspecificobjects)
            if generateFullFoVmodel:
                tdose.gen_fullFoV_from_cutouts(setupfile,clobber=clobber)

            if generateOverviewPlots:
                tu.gen_overview_plot('all',setupfile)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    bundles     = collections.OrderedDict()
    Nsetups     = len(setupfiles)
    if (type(Nsessions) is not int) or (Nsessions < 0):
        sys.exit(' ---> Nsessions must be a positive integer; it was not: '+Nsessions)
    if Nsessions == 0:
        Nbundle = Nsetups

        for ii in xrange(Nbundle):
            string          = 'bundleNo'+str(ii+1)
            bundles[string] = [setupfiles[ii]]
    else:
        Nbundle     = int(Nsessions)
        bundlesize  = int(np.ceil(float(Nsetups)/float(Nbundle)))

        for ii in xrange(Nbundle):
            string = 'bundleNo'+str(ii+1)
            if ii == Nbundle: # Last bundle
                bundles[string] = setupfiles[ii*bundlesize:]
            else:
                bundles[string] = setupfiles[bundlesize*ii:bundlesize*(ii+1)]

    if verbose: print ' - Found '+str(Nsetups)+' setup files to bundle up and run '+str(Nbundle)+' parallel sessions for'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' ---- Starting multiprocess parallel run of the '+str(Nsetups)+' TDOSE setups ---- '
    tstart  = tu.get_now_string(withseconds=True)

    mngr = multiprocessing.Manager() # initialize Manager too keep track of worker function output
    return_dict = mngr.dict()        # define Manager dictionary to store output from Worker function in
    jobs = []

    for ii in xrange(Nbundle):
        bundlekey = 'bundleNo'+str(ii+1)
        if len(bundles[bundlekey]) == 1:
            jobname = bundles[bundlekey][0].split('/')[-1]
        else:
            jobname = bundlekey

        job = multiprocessing.Process(target=parallel_worker,
                                      args  = (bundles[bundlekey],performcutout,generatesourcecat,modelrefimage,
                                               refimagemodel2cubewcs,definePSF,modeldatacube,createsourcecube,store1Dspectra,
                                               plot1Dspectra,plotS2Nspectra,save_init_model_output,clobber,
                                               verbose,verbosefull,logterminaloutput,
                                               generateFullFoVmodel,generateOverviewPlots,skipextractedobjects,skipspecificobjects),
                                      name  = jobname)

        jobs.append(job)
        job.start()
        #job.join() # wait until job has finished

    for job in jobs:
        job.join()

    tend = tu.get_now_string(withseconds=True)

    if verbose:
        print '\n ---- The perform_extractions_in_parallel finished running the jobs for all TDOSE setups ----'
        print '      Start        : '+tstart
        print '      End          : '+tend
        print '      Exitcode = 0 : job produced no error '
        print '      Exitcode > 0 : job had an error, and exited with that code (signal.SIGTERM)'
        print '      Exitcode < 0 : job was killed with a signal of -1 * exitcode (signal.SIGTERM)'

        for job in jobs:
            print ' - The job running field ',job.name,' exited with exitcode: ',job.exitcode

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Adding output from parallelized run to dictionary'
    dict = {}
    for key in return_dict.keys():
        dict[key] = return_dict[key]  # filling dictionary

    return bundles, dict
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_cutouts(setupdic,extractids,sourceids_init,sourcedat_init,
                performcutout=True,generatesourcecat=True,clobber=False,verbose=True,verbosefull=True,start_time=0.0,
                check4modelcat=True):
    """
    Generate cutouts of reference image and data cube

    --- INPUT ---
    setupdic              Dictionary containing the setup parameters read from the TDOSE setup file
    extractids            IDs of objects to extract spectra for
    sourceids_init        The initial source IDs
    sourcedat_init        The initial source data
    performcutout         Set to true to actually perform cutouts.
    generatesourcecat     To generate a (sub) source catalog corresponding to the objects in the cutout
    clobber               Overwrite existing files
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


        cutstr, cutoutsize, cut_img, cut_cube, cut_variance, cut_sourcecat = tu.get_datinfo(cutoutid,setupdic)

        if setupdic['wht_image'] is None:
            imgfiles = [setupdic['ref_image']]
            imgexts  = [setupdic['img_extension']]
            cut_img  = [cut_img]
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
            foundmodelcat = False
            if check4modelcat:
                if setupdic['modelimg_directory'] is not None:
                    if not performcutout: # only print if info from cutting out is not printed
                        print(' - Looking for source catalogs in the "modelimg_directory" ')
                    checkstring = 'noModelComponent'
                    if checkstring in cut_sourcecat:
                        print(' - '+checkstring+' source catalog; using (ra,dec) match to "source_catalog" instead')
                    else:
                        model_sourcecat_str = setupdic['modelimg_directory']+'/*'+\
                                              cut_sourcecat.split('_id')[-1].replace('.fits','_sourcecatalog.fits')
                        model_sourcecat = glob.glob(model_sourcecat_str)
                        if len(model_sourcecat) == 1:
                            if not performcutout: # only print if info from cutting out is not printed
                                print('   Found a unqie match for the objects -> using it instead of a (ra,dec) match to "source_catalog" ')
                            shutil.copyfile(model_sourcecat[0],cut_sourcecat)
                            foundmodelcat = True
                        elif len(model_sourcecat) > 1:
                            if not performcutout: # only print if info from cutting out is not printed
                                print('   Found '+str(len(model_sourcecat))+
                                      ' matches for the object -> using (ra,dec) match to "source_catalog" instead')
                        else:
                            if not performcutout: # only print if info from cutting out is not printed
                                print('   Did not find any generating cutout source catalog from (ra,dec) match to "source_catalog" ')

            if not foundmodelcat:
                if not performcutout: # only print if info from cutting out is not printed
                    print(' - Generating cutout source catalog from (ra,dec) match to main "source_catalog" ')
                obj_in_cut_fov = np.where( (sourcedat_init[setupdic['sourcecat_racol']] <
                                            (ra + cutoutsize[0]/2./3600. / np.cos(np.deg2rad(dec)))) &
                                           (sourcedat_init[setupdic['sourcecat_racol']] >
                                            (ra - cutoutsize[0]/2./3600. / np.cos(np.deg2rad(dec)))) &
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
                                          cut_sourcedat[ii][setupdic['sourcecat_deccol']], frame='fk5', unit='deg')
                    pixcoord   = wcs.utils.skycoord_to_pixel(skycoord,wcs_in,origin=1)
                    cut_sourcedat[ii][setupdic['sourcecat_xposcol']] = pixcoord[0]
                    cut_sourcedat[ii][setupdic['sourcecat_yposcol']] = pixcoord[1]

                    storearr[ii] = np.vstack(cut_sourcedat)[ii,:]

                astropy.io.fits.writeto(cut_sourcecat,storearr,header=None,overwrite=clobber)
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
def gen_fullFoV_from_cutouts(setupfile,store_sourcemodelcube=False,store_modelcube=True,clobber=False,verbose=True):
    """
    This routine combines the 3D scaled model cubes obtained from individual cutouts to a
    source model cube of the full FoV the cutouts were extracted from so the full FoV IFU
    cube can be modified based on the individual cutouts.

    --- INPUT ---
    setupfile              TDOSE setup file used to run tdose.perform_extraction() with  model_cutouts=True
    store_sourcemodelcube  Save the 4D source model cube to a fits file (it's large: ~ size of 3Dcube * Nsources).
                           Hence, if too little memory is available on the system python will likely crash.
    store_modelcube        If true a model cube (woudl be the same as summing over the source model cube) will be
                           stored as a seperate fits file. This only requieres memory enough to handle two cubes
                           as opposed to Nsources * cube when manipulating the 4D source model cube.
    clobber                Overwrite existing files
    verbose                Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose
    tdose.gen_fullFoV_from_cutouts('/Users/kschmidt/work/TDOSE/tdose_setup_candels-cdfs-02.txt')

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Loading setup file and getting IDs that were extracted (and hence cutout)'
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
    if verbose: print '   Will combine models of '+str(Nextractions)+' extracted objects (if models exists) into full FoV cubes '
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Checking that source model cubes exist in models_directory '
    modeldir = setupdic['models_directory']
    basename = setupdic['data_cube'].split('/')[-1].split('.fit')[0]

    for objid in extractids:
        sourcemodelcube = glob.glob(modeldir+basename+'*id'+str(int(objid))+'*cutout*'+
                                    setupdic['source_model_cube_ext']+'_'+setupdic['psf_type']+'.fits')

        if len(sourcemodelcube) == 0:
            if verbose: print '   WARNING: did not find a source model cube for object '+str(objid)
        elif len(sourcemodelcube) > 1:
            if verbose: print '   WARNING: found more than one source model cube for object '+str(objid)+\
                              '\n   Using '+sourcemodelcube[0]

    if verbose: print '   If no WARNINGs raised, all cubes were found'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Build template full FoV cubes to fill with models'
    cube_data     = pyfits.open(setupdic['data_cube'])[setupdic['cube_extension']].data
    cube_data_hdr = pyfits.open(setupdic['data_cube'])[setupdic['cube_extension']].header
    striphdr      = tu.strip_header(cube_data_hdr.copy())
    cubewcs       = wcs.WCS(striphdr)
    cubewcs_2D    = tu.WCS3DtoWCS2D(cubewcs.copy())
    cube_shape    = cube_data.shape

    if store_sourcemodelcube:
        smc_out      = np.zeros([Nextractions,cube_shape[0],cube_shape[1],cube_shape[2]])
    if store_modelcube:
        cube_out     = np.zeros([cube_shape[0],cube_shape[1],cube_shape[2]])
        cube_model   = np.zeros([cube_shape[0],cube_shape[1],cube_shape[2]])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Adding individual models to full FoV output cubes '
    for oo, objid in enumerate(extractids):
        cutstr, cutoutsize, cut_img, cut_cube, cut_variance, cut_sourcecat = tu.get_datinfo(objid,setupdic)

        sourcemodelcube = glob.glob(modeldir+basename+'*id'+str(int(objid))+'*cutout*'+
                                    setupdic['source_model_cube_ext']+'_'+setupdic['psf_type']+'.fits')

        if len(sourcemodelcube) > 0:
            subsourcecat_file = setupdic['source_catalog'].replace('.fits',cutstr+'.fits')
            if not os.path.isfile(subsourcecat_file):
                sys.exit(' ---> did not find the source catalog \n                  '+subsourcecat_file+
                         '\n                  need it to locate model and define region of insertion in full FoV cube. ')
            subsourcecat    = pyfits.open(subsourcecat_file)[1].data
            objid_modelent  = np.where(subsourcecat['id'] == objid)[0][0]

            if setupdic['sourcecat_parentIDcol'] is None:
                parent_id       = None
                Nparent         = 1
            else:
                parent_id       = subsourcecat[setupdic['sourcecat_parentIDcol']][objid_modelent]
                parent_ent      = np.where(subsourcecat[setupdic['sourcecat_parentIDcol']] == parent_id)[0]
                source_ids      = subsourcecat['id'][parent_ent]
                Nparent         = len(parent_ent)

            sourcemodelhdu = pyfits.open(sourcemodelcube[0])
            if Nparent == 1:
                infostr = '   > Getting object model for '+str(int(objid))+' (source model no. '+str(int(objid_modelent))+')'+\
                          '   (obj '+str("%.5d" % (oo+1))+' / '+str("%.5d" % (Nextractions))+')       '
                sourcemodel = sourcemodelhdu[setupdic['cube_extension']].data[objid_modelent,:,:,:]
            else:
                infostr = '   > Getting object model for '+str(int(parent_id))+'\n     (combining source models: '+\
                          ','.join([str(int(id)) for id in source_ids])+', i.e. source model no. '+\
                          ','.join([str(int(ent)) for ent in parent_ent])+')'+\
                          '   (obj '+str("%.5d" % (oo+1))+' / '+str("%.5d" % (Nextractions))+')       '

                sourcemodel = np.sum(sourcemodelhdu[setupdic['cube_extension']].data[parent_ent,:,:,:],axis=0)

            if verbose:
                sys.stdout.write("%s\r" % infostr)
                sys.stdout.flush()

            ra_obj        = subsourcecat[setupdic['sourcecat_racol']][objid_modelent]
            dec_obj       = subsourcecat[setupdic['sourcecat_deccol']][objid_modelent]
            skyc          = SkyCoord(ra_obj, dec_obj, frame='fk5', unit=(units.deg,units.deg))
            size          = units.Quantity((  cutoutsize[1], cutoutsize[0]), units.arcsec)
            cutout_layer  = Cutout2D(cube_data[0,:,:], skyc, size, wcs=cubewcs_2D, mode='partial')

            if store_sourcemodelcube:
                smc_out[oo,:,cutout_layer.bbox_original[0][0]:cutout_layer.bbox_original[0][1]+1,
                             cutout_layer.bbox_original[1][0]:cutout_layer.bbox_original[1][1]+1] = sourcemodel
            if store_modelcube:
                if store_sourcemodelcube:
                    continue
                else:
                    cube_model = cube_model*0.0 # reset to zeros
                    cube_model[:,cutout_layer.bbox_original[0][0]:cutout_layer.bbox_original[0][1]+1,
                                 cutout_layer.bbox_original[1][0]:cutout_layer.bbox_original[1][1]+1] = sourcemodel
                    cube_out = cube_out + cube_model
            sourcemodelhdu.close()
    if verbose: print '\n   ... done'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if store_sourcemodelcube:
        fullfov_smc = modeldir+basename+'_'+setupdic['source_model_cube_ext']+'_'+setupdic['psf_type']+'_fullFoV.fits'
        if verbose: print ' - Storing final full FoV source model cube to:\n   '+fullfov_smc
        if 'XTENSION' in cube_data_hdr.keys():
            hduprim        = pyfits.PrimaryHDU()  # default HDU with default minimal header
            hducube        = pyfits.ImageHDU(smc_out,header=cube_data_hdr)
            hdus           = [hduprim,hducube]
        else:
            hducube = pyfits.PrimaryHDU(smc_out,header=cube_data_hdr)
            hdus           = [hducube]

        hdulist = pyfits.HDUList(hdus)       # turn header into to hdulist
        hdulist.writeto(fullfov_smc,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if store_modelcube:
        fullfov_cube = modeldir+basename+'_'+setupdic['model_cube_ext']+'_'+setupdic['psf_type']+'_fullFoV.fits'
        if store_sourcemodelcube:
            cube_out = np.sum(smc_out,axis=0)
        if verbose: print ' - Producing FoV model cube from full FoV source model cube and storing it in:\n   '+fullfov_cube
        if 'XTENSION' in cube_data_hdr.keys():
            hduprim        = pyfits.PrimaryHDU()  # default HDU with default minimal header
            hducube        = pyfits.ImageHDU(cube_out,header=cube_data_hdr)
            hdus           = [hduprim,hducube]
        else:
            hducube = pyfits.PrimaryHDU(cube_out,header=cube_data_hdr)
            hdus           = [hducube]

        hdulist = pyfits.HDUList(hdus)       # turn header into to hdulist
        hdulist.writeto(fullfov_cube,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_refimage(setupdic,refimg,img_hdr,sourcecat,modelimg,modelparam,regionfile,img_wcs,img_data,names,
                   save_init_model_output=True,centralpointsource=False,clobber=True,verbose=True,verbosefull=True):
    """
    Modeling the refernce image

    --- INPUT ---
    setupdic                Dictionary containing the setup parameters read from the TDOSE setup file
    refimg                  Name of fits reference image to model
    img_hdr                 Fits header of reference image
    sourcecat               Source catalog providing coordinates of objects in reference image to model
    modelimg                Name of output file to store model to
    modelparam              Fits table to contain the model parameters (which will be turned into a DS9 region file)
    regionfile              The name of the regionfile to generate with model parameter regions
    img_wcs                 WCS of image to model
    img_data                Data of image array
    names                   Names of individual objects used in DS9 region
    save_init_model_output  Set to true to save the initial model to files
    centralpointsource      To insert central point source set to true
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
            objects   = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_IDcol']].tolist()
            objxpos   = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_xposcol']].tolist()
            objypos   = pyfits.open(sourcecat)[1].data[setupdic['sourcecat_yposcol']].tolist()

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
                                                             objects=objects,objxpos=objxpos,objypos=objypos,
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
        pixscaleunique   = np.unique(np.round(pixscales,8))
        if len(pixscaleunique) != 1:
            sys.exit(' ---> The pixel scale in the x and y direction of image are different (pixscales='+str(pixscales)+')')
        else:
            sigysigxangle =  setupdic['aperture_size'] / pixscaleunique          # radius in pixels
            fluxscale     =  pyfits.open(sourcecat)[1].data['id'].astype(float)  # pixel values
    else:
        sys.exit(' ---> Setting source_model == '+setupdic['source_model']+' is not a valid entry')

    # checking if constraint is set on centroid positioning in setup file for modeling
    try:
        maxcenshift = setupdic['max_centroid_shift']
    except:
        maxcenshift = None

    if setupdic['nondetections']:
        pixscales  = wcs.utils.proj_plane_pixel_scales(img_wcs)*3600.0
        if type(setupdic['ignore_radius']) == float:
            ignore_radius_pix = np.asarray([setupdic['ignore_radius']]*2) / pixscales
        else:
            ignore_radius_pix = np.asarray(setupdic['ignore_radius']) / pixscales
    else:
        ignore_radius_pix = 'dummy'

    pinit, fit    = tmf.gen_fullmodel(img_data,sourcecat,modeltype=setupdic['source_model'],verbose=verbosefull,
                                      xpos_col=setupdic['sourcecat_xposcol'],ypos_col=setupdic['sourcecat_yposcol'],
                                      datanoise=None,sigysigxangle=sigysigxangle,
                                      fluxscale=fluxscale,generateimage=modelimg,
                                      generateresidualimage=True,clobber=clobber,outputhdr=img_hdr,
                                      param_initguess=param_initguess,max_centroid_shift=maxcenshift,
                                      centralpointsource=centralpointsource,ignore_radius=ignore_radius_pix)

    tu.model_ds9region(modelparam,regionfile,img_wcs,color='cyan',width=2,Nsigma=2,textlist=names,
                       fontsize=12,clobber=clobber)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_datacube(setupdic,extid,modcubename,rescubename,cube_data,cube_variance,paramCUBE,cube_hdr,paramPSF,
                   psfcubename=False,clobber=False,verbose=True,verbosefull=True):
    """
    Modeling the data cube

    --- INPUT ---
    setupdic                Dictionary containing the setup parameters read from the TDOSE setup file
    extid                   ID of cube to model
    modcubename             Name of model cube to generate
    rescubename             Name of residual cube to generate
    cube_data               Data cube
    cube_variance           Variance for data cube (cube_data)
    paramCUBE               Parameters of objects in data cube
    cube_hdr                Header of data cube
    paramPSF                Parameters of PSF
    psfcubename             Name of PSF cube to use for numerical convolutions
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
    if paramtype.lower() != 'aperture':
        psfcube  = pyfits.open(psfcubename)[setupdic['cube_extension']].data
    else:
        psfcube  = None

    cube_noise = np.sqrt(cube_variance) # turn variance cube into standard deviation
    cube_model, layer_scales = tmc.gen_fullmodel(cube_data,paramCUBE,paramPSF,paramtype=paramtype,
                                                 psfparamtype=psfparamtype,noisecube=cube_noise,save_modelcube=True,
                                                 cubename=modcubename,clobber=clobber,psfcube=psfcube,
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
        fwhm_p2     = setupdic['psf_FWHMp2']
        fwhm_vec    = fwhm_p0 + fwhm_p1 * (cube_waves - fwhm_p2)
        sigmas      = fwhm_vec/2.35482/cube_scales[0]
    else:
        sys.exit(' ---> '+setupdic['psf_FWHM_evolve']+' is an invalid choice for the psf_FWHM_evolve setup parameter ')

    if (setupdic['psf_type'].lower() == 'gauss') or (setupdic['psf_type'].lower() == 'kernel_gauss'):
        xpos,ypos,fluxscale,angle = 0.0, 0.0, 1.0, 0.0
        paramPSF                  = []
        for layer in np.arange(cube_data.shape[0]):
            sigma = sigmas[layer]
            paramPSF.append([xpos,ypos,fluxscale,sigma,sigma,angle])
        paramPSF  = np.asarray(paramPSF)
    else:
        sys.exit(' ---> '+setupdic['psf_type']+' is an invalid choice for the psf_type setup parameter ')

    if setupdic['psf_savecube']:
        psfcubename = setupdic['models_directory']+'/'+datacube.split('/')[-1].replace('.fits','_tdose_psfcube_'+
                                                                                       setupdic['source_model']+'.fits')
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
                    #mu_psf    = paramPSF[ll][0:2]
                    cov_psf   = tu.build_2D_cov_matrix(paramPSF[ll][4],paramPSF[ll][3],paramPSF[ll][5],verbose=False)
                    psfimg    = tu.gen_2Dgauss(np.asarray(cube_data.shape[1:]).tolist(),cov_psf,1.0,
                                               show2Dgauss=False,verbose=False)
                elif setupdic['psf_type'].lower() == 'kernel_gauss':
                    kernel_sigma = paramPSF[ll][3]
                    kernel       = astropy.convolution.Gaussian2DKernel(kernel_sigma,x_size=cube_data.shape[2],y_size=cube_data.shape[1])
                    psfimg       = kernel.array
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
