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
                       plotS2Nspectra=True,save_init_model_output=False,clobber=False,verbose=True,verbosefull=False):
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
    save_init_model_output If a SExtractor catalog is provide to the keyword gauss_guess in the setup file
                           An initial guess including the SExtractor fits is generated for the Gaussian model.
                           To save a ds9 region, image and paramater list (the two latter is available from the default
                           output of the TDOSE modeling) set save_init_model_output=True
    clobber                If True existing output files will be overwritten
    verbose                Toggle verbosity
    verbose                Toggle extended verbosity

    --- EXAMPLE OF USE ---
    import tdose

    # full extraction w/o generating cutouts
    tdose.perform_extraction(setupfile='./tdose_setup_candels-cdfs-02.txt',performcutout=False)

    # only plotting:
    tdose.perform_extraction(setupfile='./tdose_setup_candels-cdfs-02.txt',verbosefull=True,performcutout=False,generatesourcecat=False,modelrefimage=False,refimagemodel2cubewcs=False,definePSF=False,modeldatacube=False,createsourcecube=False,store1Dspectra=False,plot1Dspectra=True,clobber=True)


    """
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
    if setupdic['model_cutouts']:
        if verbose: print '=================================================================================================='
        if verbose: print ' TDOSE: Generate cutouts around sources to extract          '+\
                          '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
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


            cutstr, cutoutsize, cut_img, cut_cube, cut_noise, cut_sourcecat = tdose.get_datinfo(cutoutid,setupdic)
            cut_images.append(cut_img)

            if performcutout:
                if setupdic['data_cube'] == setupdic['noise_cube']:
                    cutouts   = tu.extract_subcube(setupdic['data_cube'],ra,dec,cutoutsize,cut_cube,
                                                   cubeext=[setupdic['cube_extension'],setupdic['noise_extension']],clobber=True,
                                                   imgfiles=[setupdic['ref_image']],imgexts=[setupdic['img_extension']],
                                                   imgnames=[cut_img],verbose=verbosefull)
                else:
                    cutouts   = tu.extract_subcube(setupdic['data_cube'],ra,dec,cutoutsize,cut_cube,
                                                   cubeext=[setupdic['cube_extension']],clobber=True,
                                                   imgfiles=[setupdic['ref_image']],imgexts=[setupdic['img_extension']],
                                                   imgnames=[cut_img],verbose=verbosefull)

                    cutouts   = tu.extract_subcube(setupdic['noise_cube'],ra,dec,cutoutsize,cut_noise,
                                                   cubeext=[setupdic['noise_extension']],clobber=True,
                                                   imgfiles=None,imgexts=None,imgnames=None,verbose=verbosefull)
            else:
                if verbose: print ' >>> Skipping cutting out images and cubes (assuming they exist)'

            # --- SUB-SOURCE CAT ---
            if generatesourcecat:
                obj_in_cut_fov = np.where( (sourcedat_init[setupdic['sourcecat_racol']] < (ra + cutoutsize[0]/2./3600.)) &
                                           (sourcedat_init[setupdic['sourcecat_racol']] > (ra - cutoutsize[0]/2./3600.)) &
                                           (sourcedat_init[setupdic['sourcecat_deccol']] < (dec + cutoutsize[1]/2./3600.)) &
                                           (sourcedat_init[setupdic['sourcecat_deccol']] > (dec - cutoutsize[1]/2./3600.)) )[0]
                Ngoodobj      = len(obj_in_cut_fov)

                cutout_hdr    = pyfits.open(cut_img)[setupdic['img_extension']].header
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

                # Create region file... ?

            else:
                if verbose: print ' >>> Skipping generating the cutout source catalogs (assume they exist)'

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
            print '=================================================================================================='

        if not verbosefull:
            if verbose: print '\n   done'
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

        imgstr, imgsize, refimg, datacube, noisecube, sourcecat = tdose.get_datinfo(extid,setupdic)

        cube_data     = pyfits.open(datacube)[setupdic['cube_extension']].data
        cube_noise    = np.sqrt(pyfits.open(noisecube)[setupdic['noise_extension']].data)
        cube_hdr      = pyfits.open(datacube)[setupdic['cube_extension']].header
        cube_wcs2D    = tu.WCS3DtoWCS2D(wcs.WCS(tu.strip_header(cube_hdr.copy())))
        cube_scales   = wcs.utils.proj_plane_pixel_scales(cube_wcs2D)*3600.0

        img_data      = pyfits.open(refimg)[setupdic['img_extension']].data
        img_hdr       = pyfits.open(refimg)[setupdic['img_extension']].header
        img_wcs       = wcs.WCS(tu.strip_header(img_hdr.copy()))
        img_scales    = wcs.utils.proj_plane_pixel_scales(img_wcs)*3600.0

        modelimg      = setupdic['models_directory']+'/'+\
                        refimg.split('/')[-1].replace('.fits','_'+setupdic['model_image_ext']+'.fits')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbosefull: print '--------------------------------------------------------------------------------------------------'
        if verbosefull: print ' TDOSE: Model reference image                               '+\
                              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        regionfile    = setupdic['models_directory']+'/'+\
                        refimg.split('/')[-1].replace('.fits','_'+setupdic['model_param_reg']+'.reg')
        modelparam    = modelimg.replace('.fits','_objparam.fits') # output from refernce image modeling

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

            pinit, fit    = tmf.gen_fullmodel(img_data,sourcecat,verbose=verbosefull,
                                              xpos_col=setupdic['sourcecat_xposcol'],ypos_col=setupdic['sourcecat_yposcol'],
                                              datanoise=None,sigysigxangle=None,
                                              fluxscale=setupdic['sourcecat_fluxcol'],generateimage=modelimg,
                                              generateresidualimage=True,clobber=clobber,outputhdr=img_hdr,
                                              param_initguess=param_initguess)

            tu.model_ds9region(modelparam,regionfile,img_wcs,color='cyan',width=2,Nsigma=2,textlist=names,
                               fontsize=12,clobber=clobber)

        else:
            if verbosefull: print ' >>> Skipping modeling reference image (assume models exist)'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbosefull: print '--------------------------------------------------------------------------------------------------'
        if verbosefull: print ' TDOSE: Convert ref. image model to cube WCS                '+\
                              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        cubewcsimg       = setupdic['models_directory']+'/'+\
                           refimg.split('/')[-1].replace('.fits','_'+setupdic['model_image_cube_ext']+'.fits')
        paramREF      = tu.build_paramarray(modelparam,verbose=verbosefull)
        paramCUBE     = tu.convert_paramarray(paramREF,img_hdr,cube_hdr,verbose=verbosefull)

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

            tmf.save_modelimage(cubewcsimg,paramCUBE,cube_data.shape[1:],param_init=False,clobber=clobber,
                                outputhdr=cubehdu.header,verbose=verbosefull)

        else:
            if verbosefull: print ' >>> Skipping converting reference image model to cube WCS frame (assume models exist)'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbosefull: print '--------------------------------------------------------------------------------------------------'
        if verbosefull: print ' TDOSE: Defining PSF                                        '+\
                              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        if definePSF or modeldatacube:
            print ' >>>>>>> KBS: Enable generating PSF using setup file parameters <<<<<<<<'
            xpos,ypos,fluxscale,angle = 0.0, 0.0, 1.0, 0.0
            paramPSF                  = []
            # according to Bacon+15 (HUDF paper) Moffat PSF FWHM goes from 0.76 arcsec in the blue to 0.61 arcsec in the red
            # with the MUSE pixel scale this gives a Gauss sigma-dependence (FWHM = 2.35482sigma) of:
            sigma_blue                = 0.76/2.35482/cube_scales[0]
            sigma_red                 = 0.61/2.35482/cube_scales[0]
            sigmas                    = np.arange(sigma_red,sigma_blue,(sigma_blue-sigma_red)/cube_data.shape[0])[::-1]
            for layer in np.arange(cube_data.shape[0]):
                sigma = sigmas[layer]
                paramPSF.append([xpos,ypos,fluxscale,sigma,sigma,angle])
            paramPSF                  = np.asarray(paramPSF)
        else:
            if verbosefull: print ' >>> Skipping defining PSF of data cube (assume it is defined)'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbosefull: print '--------------------------------------------------------------------------------------------------'
        if verbosefull: print ' TDOSE: Modelling data cube                                 '+\
                              '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        modcubename = setupdic['models_directory']+'/'+\
                      datacube.split('/')[-1].replace('.fits','_'+setupdic['model_cube_ext']+'.fits')
        rescubename = setupdic['models_directory']+'/'+\
                      datacube.split('/')[-1].replace('.fits','_'+setupdic['residual_cube_ext']+'.fits')

        if modeldatacube:

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

            cube_model, layer_scales = tmc.gen_fullmodel(cube_data,paramCUBE,paramPSF,paramtype=paramtype,
                                                         psfparamtype=psfparamtype,noisecube=cube_noise,save_modelcube=True,
                                                         cubename=modcubename,clobber=clobber,
                                                         fit_source_scales=True,outputhdr=cube_hdr,verbose=verbosefull,
                                                         returnresidual=rescubename,optimize_method=optimizer,model_layers=layers)

        else:
            if verbosefull: print ' >>> Skipping modeling of data cube (assume it exists)'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        sourcecubename  = setupdic['models_directory']+'/'+\
                          datacube.split('/')[-1].replace('.fits','_'+setupdic['source_model_cube']+'.fits')

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
            if verbosefull: print ' >>> Skipping generating source model cube (assume it exists)'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print '=================================================================================================='
        if verbose: print ' TDOSE: Storing extracted 1D spectra to files               '+\
                          '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        specoutputdir   = setupdic['spec1D_directory']

        model_cube_file = modcubename
        noise_cube_file = noisecube
        noise_cube_ext  = setupdic['noise_extension']
        smc_file        = sourcecubename
        smc_ext         = setupdic['cube_extension']

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
                                             layer_scale_ext='WAVESCL',clobber=clobber,nameext=setupdic['spec1D_name'],
                                             source_association_dictionary=SAD,outputdir=specoutputdir,
                                             noise_cube_file=noise_cube_file,noise_cube_ext=noise_cube_ext,
                                             source_model_cube_file=smc_file,source_cube_ext=smc_ext,verbose=True)

        else:
            if verbosefull: print ' >>> Skipping storing 1D spectra to binary fits tables '

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print '=================================================================================================='
        if verbose: print ' TDOSE: Plotting extracted spectra                          '+\
                          '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
        showspec        = False

        for key in SAD.keys():
            spec = specoutputdir+setupdic['spec1D_name']+'_'+key+'.fits'
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
                if verbosefull: print ' >>> Skipping plotting 1D spectra '

            if plotS2Nspectra:
                xrange = setupdic['plot_S2Nspec_xrange']
                yrange = setupdic['plot_S2Nspec_yrange']

                tes.plot_1Dspecs([spec],colors=['green'],labels=[id],plotSNcurve=True,
                                 plotname=spec.replace('.fits','_'+setupdic['plot_S2Nspec_ext']+'.pdf'),showspecs=showspec,
                                 shownoise='dummy',xrange=xrange,yrange=yrange,
                                 comparisonspecs=None,comp_colors=['dummy'],comp_labels=['dummy'],
                                 comp_wavecol='dummy',comp_fluxcol='dummy',comp_errcol='dummy')
            else:
                if verbosefull: print ' >>> Skipping plotting S/N spectra '

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose:
            print '=================================================================================================='
            print ' TDOSE: Modeling and extraction done for object '+str(extid)+\
                  '      ( Total runtime = '+str("%10.4f" % (time.clock() - start_time))+' seconds )'
            print ' - To open resulting files in DS9 execute the following command '
            print ' ds9 '+refimg+' -region '+setupdic['source_catalog'].replace('.fits','.reg')+' '+\
                  modelimg+' -region '+regionfile+' '+cubewcsimg+' '+datacube+' '+modcubename+' '+rescubename+\
                  ' '+sourcecubename+' -lock frame wcs -tile grid layout 7 1 &'
            print '============================================================================================'

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
                 |__    __|| |  __ \\    /  __  \\  / ____\\  |  ___||
                    |  ||    | || \ \\   | || | ||  \ \\__    | ||__
                    |  ||    | || | ||   | || | ||   \__  \\  |  __||
                    |  ||    | ||_/ //   | ||_| ||   ___\  \\ | ||___
                    |__||    |_____//    \_____//   |______// |_____||

                    _____      ______     __    __    ______    ___
                   |  __ \\   /  __  \\  |  \\ |  ||  |  ___||  |  ||
                   | || \ \\  | || | ||  |   \\|  ||  | ||__    |  ||
                   | || | ||  | || | ||  |        ||  |  __||   |__||
                   | ||_/ //  | ||_| ||  |  ||\   ||  | ||___    __
                   |_____//   \_____//   |__|| \__||  |_____||  |__||

==================================================================================================
 """

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def modify_cube(modifysetupfile='./tdose_setup_template_modify.txt',verbose=True):
    """
    Wrapper for modyfying data cube based on sourcemodelcube

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

    """
    if cutoutid == -9999:
        cutstr     = None
        imgsize    = setupdic['cutout_sizes']
        refimg     = setupdic['ref_image']
        datacube   = setupdic['data_cube']
        noisecube  = setupdic['noise_cube']
        sourcecat  = setupdic['source_catalog']
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
        #noise_init_base = setupdic['noise_cube'].split('/')[-1]

        cut_img         = setupdic['cutout_directory']+img_init_base.replace('.fits',cutstr+'.fits')
        cut_cube        = setupdic['cutout_directory']+cube_init_base.replace('.fits',cutstr+'.fits')
        cut_noise       = cut_cube.replace('.fits','_noise.fits')
        cut_sourcecat   = setupdic['source_catalog'].replace('.fits',cutstr+'.fits')

        refimg     = cut_img
        datacube   = cut_cube
        if setupdic['data_cube'] == setupdic['noise_cube']:
            noisecube  = cut_cube
        else:
            noisecube  = cut_noise
        sourcecat = cut_sourcecat

    return cutstr, imgsize, refimg, datacube, noisecube, sourcecat

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
