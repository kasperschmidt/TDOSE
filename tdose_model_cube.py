# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import os
import sys
import pyfits
import scipy
import scipy.ndimage
import scipy.optimize as opt
import tdose_utilities as tu
import tdose_model_cube as tmc
import matplotlib as mpl
import matplotlib.pylab as plt
import tdose_model_FoV as tmf
import astropy.convolution
import pdb
import warnings
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_fullmodel(datacube,sourceparam,psfparam,paramtype='gauss',psfparamtype='gauss',fit_source_scales=True,
                  noisecube=None,save_modelcube=True,cubename='tdose_model_cube_output_RENAME.fits',clobber=True,
                  outputhdr=None,model_layers=None,optimize_method='matrix',returnresidual=None,verbose=True,
                  loopverbose=False):
    """
    Generate full model of data cube

    --- INPUT ---
    datacube           Data cube to generate model for
    sourceparam        List of parameters describing sources to generate model for.
                       The expected format of the list is set by paramtype
                       A complete model can also be provide in a 2D numby array.
                       In this case paramtype should be set to 'model'
    psfparam           Wavelength dependent psf paramters to use for modeling.
    paramtype          The format expected in the sourceparam keyword is determined by this value.
                       Choose between:
                          'gauss'    Expects a source paramter list dividable by 6 containing the paramters
                                     [yposition,xposition,fluxscale,sigmay,sigmax,angle] for each source. I.e,
                                     the length of the list should be Nobjects * 6
                          'model'    A full 2D model of the field-of-view is provided to source param
    psfparamtype       The format expected in the psfparam keyword is determined by this value
                       Choose between:
                          'gauss'    Expects a (Nlayer,6) array of paramters
                                     [yposition,xposition,fluxscale,sigmay,sigmax,angle] for each layer.
    fit_source_scales  To optimize the source scale of each source set to true. If False the optimization
                       is done by scaling the convolved layer as a whole.
    noisecube          If a noise cube for the data with estimated sqrt(variances) exist, provide it here
                       so it can be used in the flux optimization in each layer.
    save_modelcube     If true, the resulting model cube will be saved to an output fits file.
    cubename           Name of fits file to save model cube to
    clobber            If true any existing fits file will be overwritten
    outputhdr          Header to use for output fits file. If none provided a default header will be used.
    model_layers       Proivde array of layer number (starting from 0) to model if full cube should not be modelled
    optimize_method    Method to use for optimizing the flux scales in each layers. Possible choices are:
                            curvefit        Numerical optimization using a chi2 approach with curve_fit
                            matrix          Minimizing the chi2 expression manually solving the matrix equation
                            lstsq           Minimizing the chi2 expression using the scipy.linalg.lstsq function
                            all             Run all three methods and compare them (for trouble shooting)
    returnresidual     Provide file name of residual cube (data-model) to return this as well
    verbose            Toggle verbosity
    loopverbose        Toggle verbosity in loop over objects

    --- EXAMPLE OF USE ---


    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Determining parameters and assembling source information'
    if (paramtype == 'gauss') or (paramtype == 'modelimg'):
        if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if paramtype == 'gauss':
            Nsource   = int(len(sourceparam)/6.0)
            params    = np.reshape(sourceparam,[Nsource,6])
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if verbose: print ' - Assembling covariance matrixes for '+str(Nsource)+' Gaussian source in parameter list'
            mu_objs   = np.zeros([Nsource,2])
            cov_objs  = np.zeros([Nsource,2,2])
            for nn in xrange(Nsource):
                mu_objs[nn,:]    = params[nn][0:2]
                cov_obj          = tu.build_2D_cov_matrix(params[nn][4],params[nn][3],params[nn][5],verbose=False)
                cov_objs[nn,:,:] = cov_obj
        elif paramtype == 'modelimg':
            Nsource = 1
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        datashape = datacube.shape
        if verbose: print ' - Looping over '+str(datashape[0])+' wavelength layers, convolve sources with'
        if verbose: print '   Gaussian PSF and optimize flux scaling for each of them.'
        layer_scales   = np.zeros([Nsource,datashape[0]])
        model_cube_out = np.zeros(datashape)

        if noisecube is None:
            if verbose: ' - WARNING No sqrt(variance) cube provide for the data cube so using a ' \
                        'cube of 1s in flux optimization'
            noisecube = np.ones(datashape)


        if model_layers is None:
            layerlist = np.arange(datashape[0])
        else:
            layerlist = np.asarray(model_layers).astype(int)

        if optimize_method =='all':
            optimize_method_list = ['curvefit','matrix','lstsq']
        else:
            optimize_method_list = [optimize_method]

        for ll in layerlist:
            if verbose:
                infostr = '   Matching layer '+str("%6.f" % (ll+1))+' / '+str("%6.f" % datashape[0])+''
                sys.stdout.write("%s\r" % infostr)
                sys.stdout.flush()

            analytic_conv = False
            if (paramtype == 'gauss') & (psfparamtype == 'gauss'):
                analytic_conv = True

            if psfparamtype == 'gauss':
                mu_psf    = psfparam[ll][0:2]
                cov_psf   = tu.build_2D_cov_matrix(psfparam[ll][4],psfparam[ll][3],psfparam[ll][5],verbose=loopverbose)
                # if not analytic_conv:
                #     psfscale  = 1 # 1 returns normalized gaussian
                #     img_psf   = tu.gen_2Dgauss(mu_psf,cov_psf,psfscale,show2Dgauss=False)
            elif psfparamtype == 'moffat':
                sys.exit(' ---> Numerical convolution and parameter handling of a "'+psfparamtype+'" PSF is not enabled yet; sorry')
            else:
                sys.exit(' ---> PSF parameter type "'+psfparamtype+'" not enabled')

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if analytic_conv:
                if verbose & (ll==0): print ' - Performing analytic convolution of Gaussian sources; Build convolved covariance matrixes'
                mu_objs_conv   = np.zeros([Nsource,2])
                cov_objs_conv  = np.zeros([Nsource,2,2])
                for nn in xrange(Nsource):
                    muconv, covarconv     = tu.analytic_convolution_gaussian(mu_objs[nn,:],cov_objs[nn,:,:],mu_psf,cov_psf)
                    mu_objs_conv[nn,:]    = muconv
                    cov_objs_conv[nn,:,:] = covarconv
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            else:
                if verbose & (ll==0): print ' - Performing numerical convolution of sources in-loop'
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            if loopverbose: print ' - Build layer image'

            if fit_source_scales:
                if loopverbose: print ' - Checking if any pixels need to be masked out '
                if ( len(np.where(np.isfinite(datacube[ll,:,:].ravel()) == True )[0]) != len(datacube[ll,:,:].ravel()) ) or \
                        ( len(np.where(np.isfinite(noisecube[ll,:,:].ravel()) == True )[0]) != len(noisecube[ll,:,:].ravel()) ):
                    if loopverbose: print '   Found non-finite values in cubes so generating mask to ignore those in modeling '
                    invalid_mask1   = np.ma.masked_invalid(datacube[ll,:,:]).mask
                    invalid_mask2   = np.ma.masked_invalid(noisecube[ll,:,:]).mask
                    comb_mask       = (invalid_mask1 | invalid_mask2)

                    datacube_layer  = np.ma.array(datacube[ll,:,:],mask=comb_mask)
                    noisecube_layer = np.ma.array(noisecube[ll,:,:],mask=comb_mask)

                    # filling masked arrays; curve_fit can't handle masked arrays as the np.asarray_chkfinite(datacube_layer)
                    # used to chekc for NaNs in curve_fit still returns an error even though the array is masked
                    datacube_layer  = datacube_layer.filled(fill_value=0.0)
                    noisecube_layer = noisecube_layer.filled(fill_value=1.0)
                else:
                    comb_mask       = None
                    datacube_layer  = datacube[ll,:,:]
                    noisecube_layer = noisecube[ll,:,:]

                #-------------------------------------------------------------------------------------------------------
                if ('curvefit' in optimize_method_list) & (analytic_conv == True):
                    if loopverbose: print ' - Optimize flux scaling of each source in full image numerically '
                    scalesCFIT, covsCFIT  = tmc.optimize_source_scale_gauss(datacube_layer,
                                                                            np.ones(datashape[1:]), # noise always ones
                                                                            mu_objs_conv,cov_objs_conv,
                                                                            optimizer='curve_fit',verbose=loopverbose)

                    output_layerCFIT      = tmc.gen_image(datashape[1:],mu_objs_conv,cov_objs_conv,
                                                          sourcescale=scalesCFIT,verbose=loopverbose)

                    output_layer   = output_layerCFIT
                    output_scales  = scalesCFIT
                #-------------------------------------------------------------------------------------------------------
                if ('matrix' in optimize_method_list) or ('lstsq' in optimize_method_list):
                    if loopverbose: print ' - Optimize flux scaling of each source in full image analytically ',
                    for ss in xrange(Nsource):
                        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        if analytic_conv:
                            img_model  = tmc.gen_image(datashape[1:],mu_objs_conv[ss],cov_objs_conv[ss],sourcescale=[1.0],
                                                       verbose=False)
                        else:
                            if psfparamtype == 'gauss':
                                psfscale   = 1 # 1 returns normalized gaussian
                                img_psf    = tu.gen_2Dgauss(datashape[1:],cov_psf,psfscale,show2Dgauss=False)
                                kerneltype = img_psf
                            elif psfparamtype == 'kernel_gauss':
                                kernel_sigma = mu_psf[0]
                                kerneltype   = astropy.convolution.Gaussian2DKernel(kernel_sigma)
                            elif psfparamtype == 'kernel_moffat':
                                kernel_alpha = 1
                                kernel_gamma = 2
                                kerneltype   = astropy.convolution.Moffat2DKernel(kernel_gamma,kernel_alpha)
                                sys.exit(' ---> psfparamtype=kernel_moffat in tdose_model_cube() not enabled')
                            else:
                                sys.exit(' ---> Invalid psfparamtype in tdose_model_cube() not enabled')

                            if paramtype != 'model':
                                inputmodel = sourceparam
                            else:
                                sys.exit(' ---> Building of model for numerical intergration is not enabled yet')

                            sys.exit(' ---> Numerical convolution is not enabled yet, sorry')
                            img_model  = tu.numerical_convolution_image(inputmodel,kerneltype,saveimg=True,imgmask=comb_mask,
                                                                        fill_value=0.0,norm_kernel=True,convolveFFT=False,
                                                                        verbose=loopverbose)
                        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        img_model  = img_model/noisecube_layer
                        modelravel = img_model.ravel()
                        if ss == 0:
                            Atrans = modelravel
                        else:
                            Atrans = np.vstack([Atrans,modelravel])

                    if Nsource == 1:
                        Atrans = np.atleast_2d(Atrans)

                    A          = np.transpose(Atrans)
                    dataravel  = (datacube_layer/noisecube_layer).ravel()

                    #---------------------------------------------------------------------------------------------------
                    if 'matrix' in optimize_method_list:
                        if loopverbose: print 'using matrix algebra '
                        ATA        = Atrans.dot(A)
                        ATd        = Atrans.dot(dataravel)
                        ATAinv     = np.linalg.inv(ATA)
                        scalesMTX  = ATAinv.dot(ATd)

                        output_layerMTX   = tmc.gen_image(datashape[1:],mu_objs_conv,cov_objs_conv,
                                                          sourcescale=scalesMTX,verbose=loopverbose)

                        output_layer   = output_layerMTX
                        output_scales  = scalesMTX
                    #---------------------------------------------------------------------------------------------------
                    if 'lstsq' in optimize_method_list:
                        if loopverbose: print 'using scipy.linalg.lstsq() '
                        LSQout     = scipy.linalg.lstsq(A,dataravel)
                        scalesLSQ  = LSQout[0]

                        output_layerLSQ  = tmc.gen_image(datashape[1:],mu_objs_conv,cov_objs_conv,
                                                         sourcescale=scalesLSQ,verbose=loopverbose)

                        output_layer   = output_layerLSQ
                        output_scales  = scalesLSQ
                #-------------------------------------------------------------------------------------------------------
                if optimize_method == 'all':
                    resCFIT    = (datacube_layer-output_layerCFIT).ravel()
                    resMTX     = (datacube_layer-output_layerMTX).ravel()
                    resLSQ     = (datacube_layer-output_layerLSQ).ravel()

                    medianCFIT = np.median(resCFIT)
                    medianMTX  = np.median(resMTX)
                    medianLSQ  = np.median(resLSQ)

                    meanCFIT   = np.mean(resCFIT)
                    meanMTX    = np.mean(resMTX)
                    meanLSQ    = np.mean(resLSQ)

                    import pylab as plt
                    plt.hist(resCFIT,label='dat-CFITmodel (med='+str("%.2f" % medianCFIT)+', mea='+str("%.2f" % meanCFIT)+')',
                             bins=50,alpha=0.3)
                    plt.hist(resMTX,label='dat-MTXmodel (med='+str("%.2f" % medianMTX)+', mea='+str("%.2f" % meanMTX)+')',
                             bins=50,alpha=0.3)
                    plt.hist(resLSQ,label='dat-LSQmodel (med='+str("%.2f" % medianLSQ)+', mea='+str("%.2f" % meanLSQ)+')',
                             bins=50,alpha=0.3)
                    plt.legend()
                    plt.show()

                    print ' ---> Stopping in gen_fullmodel() for further investigation'
                    pdb.set_trace()
            else:
                if loopverbose: print ' - Optimize flux scaling of full image numerically '
                layer_img      = tmc.gen_image(datashape[1:],mu_objs_conv,cov_objs_conv,
                                               sourcescale=params[:,2],verbose=loopverbose)
                scale, cov     = tmc.optimize_img_scale(datacube[ll,:,:],noisecube[ll,:,:],layer_img,
                                                        verbose=loopverbose)
                output_layer   = layer_img * scale
                output_scales  = np.asarray(scale.tolist * Nsource)

            layer_scales[:,ll]     = output_scales
            model_cube_out[ll,:,:] = output_layer
        if verbose: print '\n   ----------- Finished on '+tu.get_now_string()+' ----------- '
    elif paramtype == 'aperture':
        if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
        Nsource        = int(len(sourceparam)/4.0)
        datashape      = datacube.shape
        xgrid, ygrid   = tu.gen_gridcomponents(datashape[1:])
        model_img      = tmf.modelimage_aperture((xgrid,ygrid), sourceparam, showmodelimg=False, verbose=loopverbose)
        layer_scales   = np.ones([Nsource,datashape[0]])
        model_cube_out = np.repeat(model_img[np.newaxis, :, :], datashape[0], axis=0)
        if verbose: print '\n   ----------- Finished on '+tu.get_now_string()+' ----------- '
    else:
        sys.exit(' ---> Invalid parameter type ('+paramtype+') provided to gen_fullmodel()')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if save_modelcube:
        tmc.save_cube(cubename,model_cube_out,layer_scales,outputhdr=outputhdr,clobber=clobber,verbose=verbose)

        if returnresidual is not None:
            residualcube = datacube - model_cube_out
            tmc.save_cube(returnresidual,residualcube,layer_scales,outputhdr=outputhdr,clobber=clobber,verbose=verbose)

    return model_cube_out, layer_scales
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def save_cube(cubename,datacube,layer_scales,outputhdr=None,clobber=False,verbose=True):
    """
    Saveing data cube to fits file

    """
    if verbose: print ' - Saving model cube to \n   '+cubename
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if outputhdr is None:
        hducube = pyfits.PrimaryHDU(datacube)       # default HDU with default minimal header
        if verbose: print ' - No header provided so will generate one '
        # writing hdrkeys:    '---KEY--',                       '----------------MAX LENGTH COMMENT-------------'
        hducube.header.append(('BUNIT  '                      ,'(10**(-20)*erg/s/cm**2/Angstrom)**2'),end=True)
        hducube.header.append(('OBJECT '                      ,'model_cube'),end=True)
        hducube.header.append(('CRPIX1 ',    201.043514357863 ,' Pixel coordinate of reference point'),end=True)
        hducube.header.append(('CRPIX2 ',    201.629151352493 ,' Pixel coordinate of reference point'),end=True)
        hducube.header.append(('CD1_1  ',-5.55555555555556E-05,' Coordinate transformation matrix element'),end=True)
        hducube.header.append(('CD1_2  ',                   0.,' Coordinate transformation matrix element'),end=True)
        hducube.header.append(('CD2_1  ',                   0.,' Coordinate transformation matrix element'),end=True)
        hducube.header.append(('CD2_2  ',5.55555555555556E-05 ,' Coordinate transformation matrix element'),end=True)
        hducube.header.append(('CUNIT1 ','deg     '           ,' Units of coordinate increment and value'),end=True)
        hducube.header.append(('CUNIT2 ','deg     '           ,' Units of coordinate increment and value'),end=True)
        hducube.header.append(('CTYPE1 ','RA---TAN'           ,' Right ascension, gnomonic projection'),end=True)
        hducube.header.append(('CTYPE2 ','DEC--TAN'           ,' Declination, gnomonic projection'),end=True)
        hducube.header.append(('CSYER1 ',   1.50464086916E-05 ,' [deg] Systematic error in coordinate'),end=True)
        hducube.header.append(('CSYER2 ',   6.61226954775E-06 ,' [deg] Systematic error in coordinate'),end=True)
        hducube.header.append(('CRVAL1 ',          53.1078417 ,' '),end=True)
        hducube.header.append(('CRVAL2 ',         -27.8267356 ,' '),end=True)
        hducube.header.append(('CTYPE3 ','AWAV    '           ,' '),end=True)
        hducube.header.append(('CUNIT3 ','Angstrom'           ,' '),end=True)
        hducube.header.append(('CD3_3  ',                1.25 ,' '),end=True)
        hducube.header.append(('CRPIX3 ',                  1. ,' '),end=True)
        hducube.header.append(('CRVAL3 ',    4800             ,' '),end=True)
        hducube.header.append(('CD1_3  ',                  0. ,' '),end=True)
        hducube.header.append(('CD2_3  ',                  0. ,' '),end=True)
        hducube.header.append(('CD3_1  ',                  0. ,' '),end=True)
        hducube.header.append(('CD3_2  ',                  0. ,' '),end=True)

        hdus = [hducube]
    else:
        if verbose: print ' - Using header provided with "outputhdr" for output fits file '
        if 'XTENSION' in outputhdr.keys():
            hduprim        = pyfits.PrimaryHDU()  # default HDU with default minimal header
            hducube        = pyfits.ImageHDU(datacube,header=outputhdr)
            hdus           = [hduprim,hducube]
        else:
            hducube = pyfits.PrimaryHDU(datacube,header=outputhdr)
            hdus           = [hducube]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    hduscales = pyfits.ImageHDU(layer_scales)       # default HDU with default minimal header
    # writing hdrkeys:       '---KEY--',                      '----------------MAX LENGTH COMMENT-------------'
    hduscales.header.append(('EXTNAME ','WAVESCL'            ,' None'),end=True)
    hduscales.header.append(('BUNIT   '                      ,' None'),end=True)
    hduscales.header.append(('OBJECT  '                      ,' layer_scales'),end=True)
    hduscales.header.append(('CRPIX1  ',                   99,' '),end=True)
    hduscales.header.append(('CRVAL1  ',                   99,' '),end=True)
    hduscales.header.append(('CDELT1  ',                 1.25,' '),end=True)
    hduscales.header.append(('CUNIT1  ', 'Angstrom',' '),end=True)
    hduscales.header.append(('CTYPE1  ', 'WAVE    ',' '),end=True)
    hduscales.header.append(('CRPIX2  ',                  0.0,' '),end=True)
    hduscales.header.append(('CRVAL2  ',                    0,' '),end=True)
    hduscales.header.append(('CDELT2  ',                  1.0,' '),end=True)
    hduscales.header.append(('CUNIT2  ', 'Number  ',' '),end=True)
    hduscales.header.append(('CTYPE2  ', 'SOURCE  ',' '),end=True)

    hduscales.header['CRPIX1'] = hducube.header['CRPIX3']
    hduscales.header['CRVAL1'] = hducube.header['CRVAL3']
    hduscales.header['CDELT1'] = hducube.header['CD3_3']
    hdus.append(hduscales)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    hdulist = pyfits.HDUList(hdus)       # turn header into to hdulist
    hdulist.writeto(cubename,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def optimize_source_scale_gauss(img_data,img_std,mu_objs,cov_objs,optimizer='curve_fit',verbose=True):
    """
    optimize the (flux) scaling of an image by scaling each individual source (assumed to be a
    multi-variate Gaussian with mu and covariance) with respect to a (noisy) data image

    --- INPUT ---
    img_data        The (noisy) data image to scale model image provide in img_model to
    img_std         Standard deviation image for data to use in optimization
    mu_objs         Mean vectors for multivariate Gaussian sources to scale         Dimensions: [Nobj,2]
    cov_objs        Covariance matrixes for multivariate Gaussian sources to scale. Dimensions: [Nobj,2,2]
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_cube as tmc
    scale, cov = tmc.optimize_img_scale()

    """
    if verbose: print ' - Optimize residual between model (multiple Gaussians) and data with least squares in 2D'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if optimizer == 'leastsq':
        sys.exit('optimizer = "leastsq" not enabled')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif optimizer == 'curve_fit':
        scales_initial_guess   = np.ones(mu_objs.shape[0])
        imgsize                = img_data.shape
        xgrid, ygrid           = tu.gen_gridcomponents(imgsize)
        with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
            scale_best, scale_cov  = opt.curve_fit(lambda (xgrid, ygrid), *scales:
                                                   curve_fit_fct_wrapper_sourcefit((xgrid, ygrid),mu_objs,
                                                                                   cov_objs,*scales),(xgrid, ygrid),
                                                   img_data.ravel(), p0 = scales_initial_guess, sigma=img_std.ravel() )
        output = scale_best, scale_cov
    else:
        sys.exit(' ---> Invalid optimizer ('+optimizer+') chosen for optimize_source_scale_gauss()')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print '   ----------- Finished on '+tu.get_now_string()+' ----------- '
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return output
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def optimize_img_scale(img_data,img_std,img_model,optimizer='curve_fit',show_residualimg=False,verbose=True):
    """
    optimize the (flux) scaling of an image with respect to a (noisy) data image

    --- INPUT ---
    img_data        The (noisy) data image to scale model image provide in img_model to
    img_std         Standard deviation image for data to use in optimization
    img_model       Model image to (flux) scale to match img_data
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_cube as tmc
    scale, cov = tmc.optimize_img_scale()

    """
    if verbose: print ' - Optimize residual between model (multiple Gaussians) and data with least squares in 2D'
    if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if optimizer == 'leastsq':
        sys.exit('optimizer = "leastsq" no enabled')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif optimizer == 'curve_fit':
        imgsize                = img_data.shape
        xgrid, ygrid           = tu.gen_gridcomponents(imgsize)
        scale_best, scale_cov  = opt.curve_fit(lambda (xgrid, ygrid), scale:
                                               tmc.curve_fit_fct_wrapper_imgscale((xgrid, ygrid), scale, img_model),
                                               (xgrid, ygrid),
                                               img_data.ravel(), p0 = [1.0], sigma=img_std.ravel() )

        output = scale_best, scale_cov
    else:
        sys.exit(' ---> Invalid optimizer ('+optimizer+') chosen in optimize_img_scale()')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print '   ----------- Finished on '+tu.get_now_string()+' ----------- '
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if show_residualimg:
        if verbose: print ' - Displaying the residual image between data and scaled model image '
        res_img  = img_model-img_data
        plt.imshow(res_img,interpolation='none', vmin=1e-5, vmax=np.max(res_img), norm=mpl.colors.LogNorm())
        plt.title('Initial Residual = Initial Model Image - Data Image')
        plt.show()

        res_img  = img_model*scale_best-img_data
        plt.imshow(res_img,interpolation='none', vmin=1e-5, vmax=np.max(res_img), norm=mpl.colors.LogNorm())
        plt.title('Best Residual = Scaled (by '+str(scale_best)+') Model Image - Data Image')
        plt.show()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return output
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def curve_fit_fct_wrapper_sourcefit((x,y),mu_objs,cov_objs,*scales):
    """
    Wrapper for curve_fit optimizer function
    """
    imagedim  = x.shape
    scls      = np.asarray(scales)
    img_model = tmc.gen_image(imagedim,mu_objs,cov_objs,sourcescale=scls,verbose=False)

    return img_model.ravel()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def curve_fit_fct_wrapper_imgscale((x,y),scale,img_model):
    """
    Wrapper for curve_fit optimizer function
    """
    img_scaled = img_model*scale

    return img_scaled.ravel()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_image(imagedim,mu_objs,cov_objs,sourcescale='ones',verbose=True):
    """
    Performing analytic convolution of multiple  guassian sources with a single gaussian PSF in image (layer)

    --- INPUT ---
    imagedim        Image dimensions of output to contain convovled sources
    mu_objs         The mean values for each source to convolve in an (Nobj,2) array
    cov_objs        The covariance matrixes for the infividual sources in an (Nobj,2,2) array
    sourcescale     Scale to aply to sources. Default is one scaling, i.e., sourcesclae = [1]*Nobj
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_cube as tmc
    img_conv = tmc.gen_image()

    """
    img_out   = np.zeros(imagedim)
    if mu_objs.shape == (2,):
        Nmu       = 1
    else:
        Nmu       = mu_objs.shape[0]

    if cov_objs.shape == (2, 2):
        Ncov      = 1
    else:
        Ncov      = cov_objs.shape[0]

    if Nmu != Ncov:
        sys.exit(' ---> Number of mu-vectors ('+str(Nmu)+') and covariance matrixes ('+str(Ncov)+
                 ') does not agree in gen_image()')
    else:
        Nobj  = Nmu

    if sourcescale == 'ones':
        if verbose: print ' - Setting all source flux scales to 1.'
        scalings = [1]*Nobj
    else:
        if verbose: print ' - Applying user-defined source scalings provided with "sourcescale".'
        scalings = sourcescale

    for oo in xrange(Nobj):
        if Nobj == 1: # only one object
            if len(mu_objs.shape) == 2:
                muconv = mu_objs[oo,:]
            else:
                muconv = mu_objs[:]

            if len(cov_objs.shape) == 3:
                covarconv = cov_objs[oo,:,:]
            else:
                covarconv = cov_objs[:,:]
        else:
            muconv, covarconv = mu_objs[oo,:], cov_objs[oo,:,:]

        if verbose: print ' - Generating source in spatial dimensions '
        try:
            source_centered = tu.gen_2Dgauss(imagedim,covarconv,scalings[oo],show2Dgauss=False,verbose=verbose)
        except:
            print ' - ERROR: Something went wrong in tdose_model_cube.gen_image(); stopping to further investigate...'
            pdb.set_trace()

        if verbose: print ' - Positioning source at position according to mu-vector (x,y) = ('+\
                      str(muconv[1])+','+str(muconv[0])+') in output image'
        #source_positioned = tu.roll_2Dprofile(source_centered,muconv-1.0,showprofiles=False)
        source_positioned = tu.shift_2Dprofile(source_centered,muconv-1.0,showprofiles=False)

        if verbose: print ' - Adding convolved source to output image'
        img_out  = img_out + source_positioned

    return img_out
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_source_model_cube(layer_scales,cubeshape,sourceparam,psfparam,paramtype='gauss',psfparamtype='gauss',
                          save_modelcube=True,cubename='tdose_source_model_cube_output_RENAME.fits',
                          clobber=True,outputhdr='None',verbose=True):
    """
    Generate 4D cube with dimensions [Nobj,Nlayes,ydim,xdim] containing the source models
    for each source making up the model cube generated with gen_fullmodel()

    --- INPUT ---
    layer_scales    Layer scaling from chi2 optimization performed by gen_fullmodel() format [Nobj,Nlayers]
    cubeshape       Cube dimensions of source model cubes to generate for each object [Nlayes,ydim,xdim]
    sourceparam     List of parameters describing sources to generate model for.
                    The expected format of the list is set by paramtype
                    A complete model can also be provide in a 2D numby array.
                    In this case paramtype should be set to 'model'
    psfparam        Wavelength dependent psf paramters to use for modeling.
    paramtype       The format expected in the sourceparam keyword is determined by this value.
                    Choose between:
                        'gauss'    Expects a source paramter list dividable by 6 containing the paramters
                                   [yposition,xposition,fluxscale,sigmay,sigmax,angle] for each source. I.e,
                                   the length of the list should be Nobjects * 6
                        'aperture' The model is based on an aperture extraction
                        'model'    A full 2D model of the field-of-view is provided to source param
    psfparamtype    The format expected in the psfparam keyword is determined by this value
                    Choose between:
                       'gauss'    Expects a (Nlayer,6) array of paramters
                                  [yposition,xposition,fluxscale,sigmay,sigmax,angle] for each layer.
    verbose         Toggle verbosity
    save_modelcube  If true, the resulting model cube will be saved to an output fits file.
    cubename        Name of fits file to save model cube to
    clobber         If true any existing fits file will be overwritten
    outputhdr       Header to use for output fits file. If none provided a default header will be used.


    --- EXAMPLE OF USE ---


    """
    Nlayers = layer_scales.shape[1]

    if paramtype == 'gauss':
        if verbose: print ' - Set up output source model cube based on Gaussian parameters and sub-cube input shape  '
        Nsource   = int(len(sourceparam)/6.0)
        params    = np.reshape(sourceparam,[Nsource,6])

        out_cube  = np.zeros([Nsource,cubeshape[0],cubeshape[1],cubeshape[2]])

        if verbose: print ' - Loop over sources and layers to fill output cube with data '
        if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
        for ss in xrange(Nsource):
            for ll in xrange(Nlayers):
                if verbose:
                    infostr = '   Generating layer '+str("%6.f" % (ll+1))+' / '+str("%6.f" % cubeshape[0])+\
                              ' in source cube '+str("%6.f" % (ss+1))+' / '+str("%6.f" % Nsource)
                    sys.stdout.write("%s\r" % infostr)
                    sys.stdout.flush()

                if psfparamtype == 'gauss':
                    mu_psf    = psfparam[ll][0:2]
                    cov_psf   = tu.build_2D_cov_matrix(psfparam[ll][4],psfparam[ll][3],psfparam[ll][5],verbose=False)
                else:
                    sys.exit(' ---> PSF parameter type "'+psfparamtype+'" not enabled')

                mu_obj               = params[ss][0:2]
                cov_obj              = tu.build_2D_cov_matrix(params[ss][4],params[ss][3],params[ss][5],verbose=False)
                mu_conv, cov_conv    = tu.analytic_convolution_gaussian(mu_obj,cov_obj,mu_psf,cov_psf)
                layer_img            = tmc.gen_image(cubeshape[1:],np.asarray([mu_conv]),np.asarray([cov_conv]),
                                                     sourcescale=[1.0],verbose=False)
                out_cube[ss,ll,:,:]  = layer_img * layer_scales[ss,ll]
        if verbose: print '\n   ----------- Finished on '+tu.get_now_string()+' ----------- '
    elif paramtype == 'aperture':
        if verbose: print ' - Set up output source model cube based on aperture parameters and sub-cube input shape  '
        Nsource      = int(len(sourceparam)/4.0)
        params       = np.reshape(sourceparam,[Nsource,4])
        out_cube     = np.zeros([Nsource,cubeshape[0],cubeshape[1],cubeshape[2]])
        xgrid, ygrid = tu.gen_gridcomponents(cubeshape[1:])

        if verbose: print ' - Loop over sources and layers to fill output cube with data '
        if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
        for ss in xrange(Nsource):
            for ll in xrange(Nlayers):
                if verbose:
                    infostr = '   Generating layer '+str("%6.f" % (ll+1))+' / '+str("%6.f" % cubeshape[0])+\
                              ' in source cube '+str("%6.f" % (ss+1))+' / '+str("%6.f" % Nsource)
                    sys.stdout.write("%s\r" % infostr)
                    sys.stdout.flush()

                layer_img           = tmf.modelimage_aperture((xgrid,ygrid), params[ss], showmodelimg=False, verbose=False)
                out_cube[ss,ll,:,:] = layer_img * layer_scales[ss,ll]
        if verbose: print '\n   ----------- Finished on '+tu.get_now_string()+' ----------- '

    elif paramtype == 'model':
        if verbose: print ' - Set up output source model cube based on Gaussian parameters and sub-cube input shape  '
        Nsource   = 1
        out_cube  = np.zeros([Nsource,cubeshape[0],cubeshape[1],cubeshape[2]])

        if verbose: print ' - Loop over sources and layers to fill output cube with data '
        if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
        for ss in xrange(Nsource):
            for ll in xrange(Nlayers):
                if verbose:
                    infostr = '   Generating layer '+str("%6.f" % (ll+1))+' / '+str("%6.f" % cubeshape[0])+\
                              ' in source cube '+str("%6.f" % (ss+1))+' / '+str("%6.f" % Nsource)
                    sys.stdout.write("%s\r" % infostr)
                    sys.stdout.flush()
                layer_img            = sourceparam
                out_cube[ss,ll,:,:]  = layer_img * layer_scales[ss,ll]
        if verbose: print '\n   ----------- Finished on '+tu.get_now_string()+' ----------- '
    else:
        sys.exit(' ---> Invalid parameter type ('+paramtype+') provided to gen_source_model_cube()')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if save_modelcube:
        if verbose: print '\n - Saving source model cube to \n   '+cubename
        hducube = pyfits.PrimaryHDU(out_cube)       # default HDU with default minimal header
        if outputhdr == 'None':
            if verbose: print ' - No header provided so will generate one '
            # writing hdrkeys:    '---KEY--',                       '----------------MAX LENGTH COMMENT-------------'
            hducube.header.append(('BUNIT  '                      ,'(10**(-20)*erg/s/cm**2/Angstrom)**2'),end=True)
            hducube.header.append(('OBJECT '                      ,'mock_cube'),end=True)
            hducube.header.append(('CRPIX1 ',    201.043514357863 ,' Pixel coordinate of reference point'),end=True)
            hducube.header.append(('CRPIX2 ',    201.629151352493 ,' Pixel coordinate of reference point'),end=True)
            hducube.header.append(('CD1_1  ',-5.55555555555556E-05,' Coordinate transformation matrix element'),end=True)
            hducube.header.append(('CD1_2  ',                  0.,' Coordinate transformation matrix element'),end=True)
            hducube.header.append(('CD2_1  ',                  0. ,' Coordinate transformation matrix element'),end=True)
            hducube.header.append(('CD2_2  ',5.55555555555556E-05 ,' Coordinate transformation matrix element'),end=True)
            hducube.header.append(('CUNIT1 ','deg     '           ,' Units of coordinate increment and value'),end=True)
            hducube.header.append(('CUNIT2 ','deg     '           ,' Units of coordinate increment and value'),end=True)
            hducube.header.append(('CTYPE1 ','RA---TAN'           ,' Right ascension, gnomonic projection'),end=True)
            hducube.header.append(('CTYPE2 ','DEC--TAN'           ,' Declination, gnomonic projection'),end=True)
            hducube.header.append(('CSYER1 ',   1.50464086916E-05 ,' [deg] Systematic error in coordinate'),end=True)
            hducube.header.append(('CSYER2 ',   6.61226954775E-06 ,' [deg] Systematic error in coordinate'),end=True)
            hducube.header.append(('CRVAL1 ',          53.1078417 ,' '),end=True)
            hducube.header.append(('CRVAL2 ',         -27.8267356 ,' '),end=True)
            hducube.header.append(('CTYPE3 ','AWAV    '           ,' '),end=True)
            hducube.header.append(('CUNIT3 ','Angstrom'           ,' '),end=True)
            hducube.header.append(('CD3_3  ',                1.25 ,' '),end=True)
            hducube.header.append(('CRPIX3 ',                  1. ,' '),end=True)
            hducube.header.append(('CRVAL3 ',    4800             ,' '),end=True)
            hducube.header.append(('CD1_3  ',                  0. ,' '),end=True)
            hducube.header.append(('CD2_3  ',                  0. ,' '),end=True)
            hducube.header.append(('CD3_1  ',                  0. ,' '),end=True)
            hducube.header.append(('CD3_2  ',                  0. ,' '),end=True)
            hducube.header.append(('CTYPE4 ','SOURCE  '           ,' '),end=True)
            hducube.header.append(('CUNIT4 ','Number  '           ,' '),end=True)

            hdus = [hducube]
        else:
            if verbose: print ' - Using header provided with "outputhdr" for output fits file '
            if 'XTENSION' in outputhdr.keys():
                hduprim        = pyfits.PrimaryHDU()  # default HDU with default minimal header
                hducube        = pyfits.ImageHDU(out_cube,header=outputhdr)
                hdus           = [hduprim,hducube]
            else:
                hducube = pyfits.PrimaryHDU(out_cube,header=outputhdr)
                hdus           = [hducube]

            # if verbose: print ' - Using header provided with "outputhdr" for output fits file '
            # hducube.header = outputhdr

        hdulist = pyfits.HDUList(hdus)       # turn header into to hdulist
        hdulist.writeto(cubename,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return out_cube
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =