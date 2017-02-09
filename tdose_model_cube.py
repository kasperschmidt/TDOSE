# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import os
import sys
import pyfits
import scipy.ndimage
import scipy.optimize as opt
import tdose_utilities as tu
import tdose_model_cube as tmc
import matplotlib as mpl
import matplotlib.pylab as plt
import tdose_model_FoV as tmf
import pdb
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_fullmodel(datacube,sourceparam,psfparam,paramtype='gauss',psfparamtype='gauss',fit_source_scales=True,
                  noisecube='None',save_modelcube=True,cubename='tdose_model_cube_output_RENAME.fits',clobber=True,
                  outputhdr='None',verbose=True):
    """
    Generate full model of data cube

    --- INPUT ---
    datacube           Data cube to generate model for
    sourceparam        List of parameters describing sources to generate model for.
                       The expected format of the list is set by paramtype
    psfparam           Wavelength dependent psf paramters to use for modeling.
    paramtype          The format expected in the sourceparam keyword is determined by this value.
                       Choose between:
                          'gauss'    Expects a source paramter list dividable by 6 containing the paramters
                                     [yposition,xposition,fluxscale,sigmay,sigmax,angle] for each source. I.e,
                                     the length of the list should be Nobjects * 6
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
    verbose            Toggle verbosity

    --- EXAMPLE OF USE ---


    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Determining parameters and assembling source information'
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

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        datashape = datacube.shape
        if verbose: print ' - Looping over '+str(datashape[0])+' wavelength layers, convolve sources with'
        if verbose: print '   Gaussian PSF and optimize flux scaling for each of them.'
        layer_scales   = np.zeros([Nsource,datashape[0]])
        model_cube_out = np.zeros(datashape)

        if noisecube == 'None':
            if verbose: ' - WARNING No sqrt(variance) cube provide for the data cube so using a ' \
                        'cube of 1s in flux optimization'
            noisecube = np.ones(datashape)

        if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
        loopverbose = False
        for ll in xrange(datashape[0]):
            if verbose:
                infostr = '   Matching layer '+str("%6.f" % (ll+1))+' / '+str("%6.f" % datashape[0])+''
                sys.stdout.write("%s\r" % infostr)
                sys.stdout.flush()

            if psfparamtype == 'gauss':
                mu_psf    = psfparam[ll][0:2]
                cov_psf   = tu.build_2D_cov_matrix(psfparam[ll][4],psfparam[ll][3],psfparam[ll][5],verbose=loopverbose)
            else:
                sys.exit(' ---> PSF parameter type "'+psfparamtype+'" not enabled')

            if loopverbose: print ' - Build convolved covariance matrixes'
            mu_objs_conv   = np.zeros([Nsource,2])
            cov_objs_conv  = np.zeros([Nsource,2,2])
            for nn in xrange(Nsource):
                muconv, covarconv     = tu.analytic_convolution_gaussian(mu_objs[nn,:],cov_objs[nn,:,:],mu_psf,cov_psf)
                mu_objs_conv[nn,:]    = muconv
                cov_objs_conv[nn,:,:] = covarconv

            if loopverbose: print ' - Build layer image'

            if fit_source_scales:
                if loopverbose: print ' - Optimize flux scaling of each source in full image numerically '
                scales, covs   = optimize_source_scale_gauss(datacube[ll,:,:],noisecube[ll,:,:],
                                                             mu_objs_conv,cov_objs_conv,
                                                             optimizer='curve_fit',verbose=loopverbose)
                output_layer   = tmc.gen_image(datashape[1:],mu_objs_conv,cov_objs_conv,
                                               sourcescale=scales,verbose=loopverbose)
                output_scales  = scales

                # Instead of curve_fit above, this optimization should be possible to replace by matrix inversion...
                # solve   A^T A f = A^T d   where  A represents the 4D source model cube (Nobj,Nxpix,Nypix,Nlambda),
                #                                  f the 3D fluxes    (Nobj,Npix,Nlambda) and
                #                                  d is the data cube (Nxpix,Nypix,Nlambda)

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
    else:
        sys.exit(' ---> Invalid parameter type ('+paramtype+') provided to gen_fullmodel()')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if save_modelcube:
        if verbose: print ' - Saving model cube to \n   '+cubename
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        hducube = pyfits.PrimaryHDU(model_cube_out)       # default HDU with default minimal header
        if outputhdr == 'None':
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
        else:
            if verbose: print ' - Using header provided with "outputhdr" for output fits file '
            hducube.header = outputhdr

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

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        hdulist = pyfits.HDUList([hducube,hduscales])       # turn header into to hdulist
        hdulist.writeto(cubename,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)

    return model_cube_out, layer_scales
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
        sys.exit('optimizer = "leastsq" no enabled')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif optimizer == 'curve_fit':
        scales_initial_guess   = np.ones(mu_objs.shape[0])
        imgsize                = img_data.shape
        xgrid, ygrid           = tu.gen_gridcomponents(imgsize)
        scale_best, scale_cov  = opt.curve_fit(lambda (xgrid, ygrid), *scales:
                                               curve_fit_fct_wrapper_sourcefit((xgrid, ygrid),mu_objs,cov_objs,*scales),
                                               (xgrid, ygrid),
                                               img_data.ravel(), p0 = scales_initial_guess, sigma=img_std.ravel() )

        output = scale_best, scale_cov
    else:
        sys.exit(' ---> Invalid optimizer ('+optimizer+') chosen in optimize_img_scale()')
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
    img_model = gen_image(imagedim,mu_objs,cov_objs,sourcescale=scls,verbose=False)

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
    Nmu       = mu_objs.shape[0]
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
        muconv, covarconv = mu_objs[oo,:], cov_objs[oo,:,:]

        if verbose: print ' - Generating source in spatial dimensions '
        try:
            source_centered = tu.gen_2Dgauss(imagedim,covarconv,scalings[oo],show2Dgauss=False,verbose=verbose)
        except:
            pdb.set_trace()

        if verbose: print ' - Positioning source at position according to mu-vector (x,y) = ('+\
                      str(muconv[1])+','+str(muconv[0])+') in output image'
        source_positioned = tu.roll_2Dprofile(source_centered,muconv,showprofiles=False)

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
    psfparam        Wavelength dependent psf paramters to use for modeling.
    paramtype       The format expected in the sourceparam keyword is determined by this value.
                    Choose between:
                       'gauss'    Expects a source paramter list dividable by 6 containing the paramters
                                  [yposition,xposition,fluxscale,sigmay,sigmax,angle] for each source. I.e,
                                  the length of the list should be Nobjects * 6
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


        else:
            if verbose: print ' - Using header provided with "outputhdr" for output fits file '
            hducube.header = outputhdr

        hdulist = pyfits.HDUList([hducube])       # turn header into to hdulist
        hdulist.writeto(cubename,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return out_cube
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =