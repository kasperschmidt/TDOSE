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
def gen_fullmodel(datacube,sourceparam,psfparam,paramtype='gauss',psfparamtype='gauss',noisecube='None',
                  save_modelcube=True,cubename='tdose_model_cube_output_RENAME.fits',clobber=True,outputhdr='None',
                  verbose=True):
    """
    Generate full model of data cube

    --- INPUT ---
    datacube        Data cube to generate model for
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
    noisecube       If a noise cube for the data with estimated sqrt(variances) exist, provide it here
                    so it can be used in the flux optimization in each layer.
    save_modelcube  If true, the resulting model cube will be saved to an output fits file.
    cubename        Name of fits file to save model cube to
    clobber         If true any existing fits file will be overwritten
    outputhdr       Header to use for output fits file. If none provided a default header will be used.
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_cube as tmc
    path       = '/Users/kschmidt/work/TDOSE/'

    sourcecat  = path+'mock_cube_sourcecat161213_oneobj.fits'
    datacube   = pyfits.open(path+'mock_cube_sourcecat161213_oneobj_tdose_mock_cube.fits')[0].data

    output = tmc.gen_fullmodel(datacube,dourceparam,psfparam)

    dataimg    = pyfits.open(path+'mock_cube_sourcecat161213_all_tdose_mock_cube.fits')[0].data
    sourcecat  = path+'mock_cube_sourcecat161213_all.fits'
    output = tmc.gen_fullmodel()

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Determining paramters and assembling source information'
    if paramtype == 'gauss':
        Nsource   = int(len(sourceparam)/6.0)
        params    = np.reshape(sourceparam,[Nsource,6])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose: print ' - Assembling covariance matrixes'
        mu_objs   = np.zeros([Nsource,2])
        cov_objs  = np.zeros([Nsource,2,2])
        for nn in xrange(Nsource):
            mu_objs[nn,:]    = params[nn][0:2]
            cov_obj          = tu.build_2D_cov_matrix(params[nn][4],params[nn][3],params[nn][5],verbose=verbose)
            cov_objs[nn,:,:] = cov_obj

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        datashape = datacube.shape
        if verbose: print ' - Looping over '+str(datashape[0])+' wavelength layers, convolve sources with'
        if verbose: print '   Gaussian PSF and optimize flux scaling for each of them.'
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
                sys.exit(' ---> PSF parameter type "'+psfparam+'" not enabled')

            if loopverbose: print ' - Build layer image'
            layer_img = tmc.gen_convolved_image_gauss(datashape[1:],mu_objs,cov_objs,mu_psf,cov_psf,
                                                      sourcescale=params[:,2],verbose=loopverbose)

            if loopverbose: print ' - Optimize flux scaling numerically '
            scale, cov = tmc.optimize_img_scale(datacube[ll,:,:],noisecube[ll,:,:],layer_img,verbose=loopverbose)

            model_cube_out[ll,:,:] = layer_img * scale
        if verbose: print '\n   ----------- Finished on '+tu.get_now_string()+' ----------- '
    else:
        sys.exit(' ---> Invalid parameter type ('+paramtype+') provided to gen_fullmodel()')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if save_modelcube:
        if verbose: print ' - Saving model cube to \n   '+cubename
        hducube = pyfits.PrimaryHDU(model_cube_out)       # default HDU with default minimal header
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
        else:
            if verbose: print ' - Using header provided with "outputhdr" for output fits file '
            hducube        = pyfits.PrimaryHDU(model_cube_out)
            hducube.header = outputhdr

        hdulist = pyfits.HDUList([hducube])       # turn header into to hdulist
        hdulist.writeto(cubename,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)

    return model_cube_out
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
                                               tmc.curve_fit_function_wrapper((xgrid, ygrid), scale, img_model),
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
def curve_fit_function_wrapper((x,y),scale,img_model):
    """
    Wrapper for curve_fit optimizer call to be able to provide list of parameters to model_objects_gauss()
    """
    img_scaled = img_model*scale

    return img_scaled.ravel()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_convolved_image_gauss(imagedim,mu_objs,cov_objs,mu_psf,cov_psf,sourcescale='ones',verbose=True):
    """
    Performing analytic convolution of multiple  guassian sources with a single gaussian PSF in image (layer)

    --- INPUT ---
    imagedim        Image dimensions of output to contain convovled sources
    mu_objs         The mean values for each source to convolve in an (Nobj,2) array
    cov_objs        The covariance matrixes for the infividual sources in an (Nobj,2,2) array
    cov_psf         The covariance matric of the Gaussian PSF to convolve sources with
                    (assuming mean vetor is [0,0] in image to preserve source center)
    sourcescale     Scale to aply to sources. Default is one scaling, i.e., sourcesclae = [1]*Nobj
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_cube as tmc
    img_conv = tmc.gen_convolved_image_gauss()

    """
    img_out   = np.zeros(imagedim)
    Nmu       = mu_objs.shape[0]
    Ncov      = cov_objs.shape[0]
    if Nmu != Ncov:
        sys.exit(' ---> Number of mu-vectors ('+str(Nmu)+') and covariance matrixes ('+str(Ncov)+
                 ') does not agree in gen_convolved_image_gauss()')
    else:
        Nobj  = Nmu

    if sourcescale == 'ones':
        if verbose: print ' - Setting all source flux scales to 1.'
        scalings = [1]*Nobj
    else:
        if verbose: print ' - Applying user-defined source scalings provide with "sourcescale".'
        scalings = sourcescale

    for oo in xrange(Nobj):
        muconv, covarconv = tu.analytic_convolution_gaussian(mu_objs[oo],cov_objs[oo],mu_psf,cov_psf)

        if verbose: print ' - Generating source in spatial dimensions '
        source_centered = tu.gen_2Dgauss(imagedim,covarconv,scalings[oo],show2Dgauss=False,verbose=verbose)

        if verbose: print ' - Positioning source at position according to mu-vector (x,y) = ('+\
                      str(muconv[1])+','+str(muconv[0])+') in output image'
        source_positioned = tu.roll_2Dprofile(source_centered,muconv,showprofiles=False)

        if verbose: print ' - Adding convolved source to output image'
        img_out  = img_out + source_positioned


    return img_out
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =