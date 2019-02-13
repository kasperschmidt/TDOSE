# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import os
import sys
import pyfits
import scipy.ndimage
import scipy.optimize as opt
import tdose_utilities as tu
import matplotlib as mpl
mpl.use('Agg') # prevent pyplot from opening window; enables closing ssh session with detached screen running TDOSE
import matplotlib.pylab as plt
import tdose_model_FoV as tmf
import pdb
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_fullmodel(dataimg,sourcecatalog,modeltype='gauss',xpos_col='xpos',ypos_col='ypos',sigysigxangle=None,datanoise=None,
                  fluxscale='fluxscale',show_residualimg=False,generateimage=False,generateresidualimage=False,
                  optimizer='curve_fit',clobber=False,outputhdr=None,param_initguess=None,max_centroid_shift=None,
                  centralpointsource=False,ignore_radius=0.5,verbose=True):
    """
    Generate the full model of the FoV to extract spectra from

    --- INPUT ---
    dataimg                   The image to model sources in
    sourcecatalog             Source catalog of sources to model in dataimg
    modeltype                 The type of model to use. Select between:
                                  gauss       Generate a multivariate gaussian model at each source position
                                  aperture    Add apertures of radius sigysigxangle at the location of each source
                                  galfit      Not enabled yet
    xpos_col                  Column name of sourcecatalog column containing the x-coordinates of sources
    ypos_col                  Column name of sourcecatalog column containing the y-coordinates of sources
    sigysigxangle             Parameters for source models in image. For:
                                  modeltype = gauss    sigysigxangle    corresponds to the sigma_y, sigma_x and
                                                                        angle of the multivariate gaussian models
                                                                        is to be used provide it here. Expects either
                                                                        a list of stings where the three values are
                                                                        seperated by '_' or an array of size [N,3]
                                                                        where sigysigxangle[ii,:] = sigma_yi, sigma_xi angle_i
                                  modeltype = aperture sigysigxangle    corresponds to the radius of the aperture in pixels
                                                                        provided as a single value for all apertures or
                                                                        a list of different aperture sizes
                                  modeltype = galfit   sigysigxangle    NA
    datanoise                 Noise image corresponding to dataimg
    fluxscale                 Flux scale to apply to each source model given as either a list of values or
                              a single value. If:
                                  modeltype = gauss    fluxscale        corresponds to a scaling of the gaussian profile
                                  modeltype = aperture fluxscale        corresponds to the pixel values in the apertures,
                                                                        e.g., the ID of the objects.
                                  modeltype = galfit   sigysigxangle    NA
    show_residualimg          To show the residual image for each model, set this keyword to true
    generateimage             To store model to fits image provide the path and name of file to generate here.
    generateresidualimage     To store residual between image and model to fits image provide the path and name
                              of file to generate here.
    optimizer                 Optimizer to use for generating model. Chose between:
                                  leastsq     scipy.optimize.leastsq(); not ideal for 2D images...
                                              Tries to optimize the residual function
                                  curve_fit   scipy.optimize.curve_fit()
                                              Tries to optimize using the model function
    clobber                   Overwrite files if they already exist
    outputhdr                 Fits header to use as template for models
    param_initguess           To use a TDOSE parameter list as intial guess for image model provide it here
    max_centroid_shift        Maximum offset in pixels of (x,y) centroid position allowed when modeling
    centralpointsource        To add a point source (representing a non-detection) to the center of the model
                              (replacing the source model closest to this location) set to true.
                              Adding a point source, all source models within ignore_radius will be ignored
    ignore_radius             Radius (in pixels) around central point source (pointsource=True) where source models
                              will be ignored, i.e., set to 0 via a fluxscale=0 in the parameterlist
    verbose                   Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf
    path       = '/Users/kschmidt/work/TDOSE/'
    dataimg    = pyfits.open(path+'mock_cube_sourcecat161213_oneobj_tdose_mock_cube.fits')[0].data[0,:,:]
    sourcecat  = path+'mock_cube_sourcecat161213_oneobj.fits'
    param_init, fit_output = tmf.gen_fullmodel(dataimg,sourcecat,xpos_col='xpos',ypos_col='ypos',sigysigxangle=None,fluxscale='fluxscale',generateimage=path+'mock_cube_sourcecat161213_oneobj_modelimage.fits')

    dataimg    = pyfits.open(path+'mock_cube_sourcecat161213_all_tdose_mock_cube.fits')[0].data[0,:,:]
    sourcecat  = path+'mock_cube_sourcecat161213_all.fits'
    param_init, fit_output = tmf.gen_fullmodel(dataimg,sourcecat,xpos_col='xpos',ypos_col='ypos',sigysigxangle=None,fluxscale='fluxscale',generateimage=path+'mock_cube_sourcecat161213_all_modelimage.fits')

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if modeltype.lower() == 'gauss':
        if param_initguess is None:
            if verbose: print ' - Loading source catalog info to build inital guess of parameters for model fit'
            if verbose: print '   Will use x position, y position,',
            if fluxscale is not None:
                if verbose: print ' fluxscale',
            if sigysigxangle is not None:
                if verbose: print ' sigysigxangle',
            if verbose: print ' in initial guess'
            param_init   = tmf.gen_paramlist(sourcecatalog,xpos_col=xpos_col,ypos_col=ypos_col,
                                             sigysigxangle=sigysigxangle,fluxscale=fluxscale,verbose=verbose)
        else:
            if verbose: print ' - Using the intitial guess provided for model fit'
            param_init   = param_initguess

        fit_output      = tmf.model_objects_gauss(param_init,dataimg,optimizer=optimizer,max_centroid_shift=max_centroid_shift,
                                                  datanoise=datanoise,verbose=verbose,show_residualimg=show_residualimg)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if centralpointsource:
            if verbose: print ' - Requested to insert central point source; modifying parameter list from FoV Gauss model fit'
            ycen,xcen     = dataimg.shape[0]/2., dataimg.shape[1]/2.
            if type(ignore_radius) == float:
                ignore_radius = [ignore_radius]*2

            Nsources  = int(len(fit_output[0])/6.)
            for oo in xrange(Nsources):
                obj_xpix = fit_output[0][1::6][oo]
                obj_ypix = fit_output[0][0::6][oo]
                if ((obj_ypix-ycen)**2.0 < ignore_radius[0]**2.0) & ((obj_xpix-xcen)**2.0 < ignore_radius[1]**2.0):
                    fit_output[0][2::6][oo] = 0.0

            r_diffs = np.abs(np.sqrt((fit_output[0][0::6]-ycen)**2.0 + (fit_output[0][1::6]-xcen)**2.0))
            central_source = np.where(r_diffs == np.min(r_diffs))[0]
            fit_output[0][1::6][central_source]  = xcen # xposition
            fit_output[0][0::6][central_source]  = ycen # yposition
            fit_output[0][2::6][central_source]  = 1.0  # fluxscale
            fit_output[0][4::6][central_source]  = 1.0  # sigmax
            fit_output[0][3::6][central_source]  = 1.0  # sigmay
            fit_output[0][5::6][central_source]  = 0.0  # angle
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    elif modeltype.lower() == 'galfit':
        if param_initguess is not None:
            if verbose: print (' TDOSE WARNING: Initial guess is not enabled for modeltype = galfit; setting param_initguess = None')
            param_init  = None

        galfitparamfile = sourcecatalog
        fit_output      = tmf.model_objects_galfit(dataimg,galfitparamfile,verbose=verbose,show_residualimg=show_residualimg)

    elif modeltype.lower() == 'aperture':
        if verbose: print ' - Building paramter list from provided aperture parameters'
        paramlist     = tmf.gen_paramlist_aperture(sourcecatalog,sigysigxangle,pixval=fluxscale,
                                                   xpos_col=xpos_col,ypos_col=ypos_col,verbose=verbose)

        fit_output      = paramlist, 'dummy'
        param_init      = paramlist
    else:
        sys.exit(' ---> "modeltype"='+modeltype+' is an invalid choice of modeling setup so aborting')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if type(generateimage) == str:
        tmf.save_modelimage(generateimage,fit_output[0],dataimg.shape,modeltype=modeltype,param_init=param_init,
                            verbose=verbose,verbosemodel=verbose,clobber=clobber,outputhdr=outputhdr)
        if modeltype.lower() != 'aperture':
            tmf.save_modelimage(generateimage.replace('.fits','_initial.fits'),param_init,dataimg.shape,modeltype=modeltype,
                                param_init=False,verbose=verbose,verbosemodel=verbose,clobber=clobber,outputhdr=outputhdr)

        if generateresidualimage:
            tmf.save_modelimage(generateimage.replace('.fits','_residual.fits'),fit_output[0],dataimg.shape,modeltype=modeltype,
                                param_init=param_init,verbose=verbose,verbosemodel=verbose,clobber=clobber,outputhdr=outputhdr,
                                dataresidual=dataimg)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Storing fitted source paramters as fits table and returning output'
    tablename  = generateimage.replace('.fits','_objparam.fits')
    if modeltype.lower() == 'gauss': # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        objnumbers = np.arange(len(fit_output[0])/6)+1
        c01 = pyfits.Column(name='obj',            format='D', unit='',       array=objnumbers)
        c02 = pyfits.Column(name='xpos',           format='D', unit='PIXELS', array=fit_output[0][1::6])
        c03 = pyfits.Column(name='ypos',           format='D', unit='PIXELS', array=fit_output[0][0::6])
        c04 = pyfits.Column(name='fluxscale',      format='D', unit='',       array=fit_output[0][2::6])
        c05 = pyfits.Column(name='xsigma',         format='D', unit='PIXELS', array=fit_output[0][4::6])
        c06 = pyfits.Column(name='ysigma',         format='D', unit='PIXELS', array=fit_output[0][3::6])
        c07 = pyfits.Column(name='angle',          format='D', unit='DEGREES',array=fit_output[0][5::6])
        c08 = pyfits.Column(name='xpos_init',      format='D', unit='PIXELS', array=param_init[1::6])
        c09 = pyfits.Column(name='ypos_init',      format='D', unit='PIXELS', array=param_init[0::6])
        c10 = pyfits.Column(name='fluxscale_init', format='D', unit='',       array=param_init[2::6])
        c11 = pyfits.Column(name='xsigma_init',    format='D', unit='PIXELS', array=param_init[4::6])
        c12 = pyfits.Column(name='ysigma_init',    format='D', unit='PIXELS', array=param_init[3::6])
        c13 = pyfits.Column(name='angle_init',     format='D', unit='DEGREES',array=param_init[5::6])
        coldefs = pyfits.ColDefs([c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13])
    elif modeltype.lower() == 'aperture': # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        objnumbers = np.arange(len(fit_output[0])/4)+1
        c01 = pyfits.Column(name='obj',            format='D', unit='',       array=objnumbers)
        c02 = pyfits.Column(name='xpos',           format='D', unit='PIXELS', array=fit_output[0][1::4])
        c03 = pyfits.Column(name='ypos',           format='D', unit='PIXELS', array=fit_output[0][0::4])
        c04 = pyfits.Column(name='radius',         format='D', unit='PIXELS', array=fit_output[0][2::4])
        c05 = pyfits.Column(name='pixvalue',       format='D', unit='',       array=fit_output[0][3::4])
        c06 = pyfits.Column(name='xpos_init',      format='D', unit='PIXELS', array=param_init[1::4])
        c07 = pyfits.Column(name='ypos_init',      format='D', unit='PIXELS', array=param_init[0::4])
        c08 = pyfits.Column(name='radius_init',    format='D', unit='PIXELS', array=param_init[2::4])
        c09 = pyfits.Column(name='pixvalue_init',  format='D', unit='',       array=param_init[3::4])
        coldefs = pyfits.ColDefs([c01,c02,c03,c04,c05,c06,c07,c08,c09])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    th = pyfits.new_table(coldefs) # creating default header
    # writing hdrkeys:'---KEY--',                             '----------------MAX LENGTH COMMENT-------------'
    th.header.append(('MODTYPE ', modeltype.lower()          ,'The model type the parameters correspond to    '),end=True)

    tbHDU  = pyfits.new_table(coldefs, header=th.header)
    tbHDU.writeto(tablename, clobber=clobber)

    return param_init, fit_output
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def save_modelimage(outname,paramlist,imgsize,modeltype='gauss',param_init=False,dataresidual=None,
                    clobber=False,outputhdr=None,verbose=True, verbosemodel=False):
    """
    Generate and save a fits file containing the model image obtained from modeling multiple gaussians

    --- INPUT ---
    outname          File name to store model image to
    paramlist        Parameter list of sources in model to be strored in the header of the fits file
    imgsize          Size of the image model
    modeltype        The type of model which was used to generate the image
    param_init       If intitial parameters exists provide this here.
    dataresidual     To store the residual (data-model) instead of the model itself set this keyword to True
    clobber          Overwrite file if it already exists
    outputhdr        Header to use as template for fits image to generate
    verbose          Toggle verbosity
    verbosemodel     Toggle extended verbosity

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Generate model image from input paramters'

    if modeltype.lower() == 'gauss':
        xgrid, ygrid = tu.gen_gridcomponents(imgsize)
        modelimg     = tmf.modelimage_multigauss((xgrid,ygrid), paramlist, showmodelimg=False, verbose=verbosemodel)
    elif (modeltype.lower() == 'galfit') or (modeltype.lower() == 'modelimg'):
        modelimg     = paramlist
    elif modeltype.lower() == 'aperture':
        xgrid, ygrid = tu.gen_gridcomponents(imgsize)
        modelimg     = tmf.modelimage_aperture((xgrid,ygrid), paramlist, showmodelimg=False, verbose=verbosemodel)
    else:
        sys.exit(' ---> "modeltype"='+modeltype+' is an invalid choice of modeling setup so aborting')


    if dataresidual is not None:
        modelimg = dataresidual - modelimg

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Saving generated image to '+outname
    if outputhdr is None:
        hduimg = pyfits.PrimaryHDU(modelimg)       # creating default fits header
        if verbose: print ' - No header provided so will generate one '
        # writing hdrkeys:    '---KEY--',                      '----------------MAX LENGTH COMMENT-------------'
        hduimg.header.append(('BUNIT   '                      ,'(10**(-20)*erg/s/cm**2/Angstrom)**2'),end=True)
        hduimg.header.append(('OBJECT  '                      ,'model image'),end=True)
        hduimg.header.append(('CRPIX1  ',    201.043514357863 ,' Pixel coordinate of reference point'),end=True)
        hduimg.header.append(('CRPIX2  ',    201.629151352493 ,' Pixel coordinate of reference point'),end=True)
        hduimg.header.append(('CD1_1   ',-5.55555555555556E-05,' Coordinate transformation matrix element'),end=True)
        hduimg.header.append(('CD1_2   ',                  0.,' Coordinate transformation matrix element'),end=True)
        hduimg.header.append(('CD2_1   ',                  0. ,' Coordinate transformation matrix element'),end=True)
        hduimg.header.append(('CD2_2   ',5.55555555555556E-05 ,' Coordinate transformation matrix element'),end=True)
        hduimg.header.append(('CUNIT1  ','deg     '           ,' Units of coordinate increment and value'),end=True)
        hduimg.header.append(('CUNIT2  ','deg     '           ,' Units of coordinate increment and value'),end=True)
        hduimg.header.append(('CTYPE1  ','RA---TAN'           ,' Right ascension, gnomonic projection'),end=True)
        hduimg.header.append(('CTYPE2  ','DEC--TAN'           ,' Declination, gnomonic projection'),end=True)
        hduimg.header.append(('CSYER1  ',   1.50464086916E-05 ,' [deg] Systematic error in coordinate'),end=True)
        hduimg.header.append(('CSYER2  ',   6.61226954775E-06 ,' [deg] Systematic error in coordinate'),end=True)
        hduimg.header.append(('CRVAL1  ',          53.1078417 ,' '),end=True)
        hduimg.header.append(('CRVAL2  ',         -27.8267356 ,' '),end=True)
    else:
        hduimg = pyfits.PrimaryHDU(modelimg,header=outputhdr)       # creating default fits header
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if modeltype.lower() == 'gauss':
        Nparam = 6
        Nobj   = int(len(paramlist)/Nparam)
        if verbose: print ' - Adding gaussian parameters of '+str(Nobj)+' objects used to generate model to header '

        for oo in xrange(Nobj):
            objno = str("%.4d" % (oo+1))
            yposition,xposition,fluxscale,sigmay,sigmax,angle = paramlist[oo*Nparam:oo*Nparam+Nparam]
            hduimg.header.append(('xp'+objno, xposition, 'Obj'+objno+': x position'),end=True)
            hduimg.header.append(('yp'+objno, yposition, 'Obj'+objno+': y position'),end=True)
            hduimg.header.append(('fs'+objno, fluxscale, 'Obj'+objno+': flux scale applied to Gauss'),end=True)
            hduimg.header.append(('xs'+objno, sigmax,    'Obj'+objno+': Standard deviation in xpos'),end=True)
            hduimg.header.append(('ys'+objno, sigmay,    'Obj'+objno+': Standard deviation in ypos'),end=True)
            hduimg.header.append(('an'+objno, angle,     'Obj'+objno+': Rotation angle of Gauss [deg]'),end=True)

            if type(param_init) == np.ndarray:
                yposition,xposition,fluxscale,sigmay,sigmax,angle = param_init[oo*Nparam:oo*Nparam+Nparam]
                hduimg.header.append(('xp'+objno+'_i', xposition, 'Obj'+objno+': Initial x position'),end=True)
                hduimg.header.append(('yp'+objno+'_i', yposition, 'Obj'+objno+': Initial y position'),end=True)
                hduimg.header.append(('fs'+objno+'_i', fluxscale, 'Obj'+objno+': Initial flux scale applied to Gauss'),end=True)
                hduimg.header.append(('xs'+objno+'_i', sigmax,    'Obj'+objno+': Initial Standard deviation in xpos'),end=True)
                hduimg.header.append(('ys'+objno+'_i', sigmay,    'Obj'+objno+': Initial Standard deviation in ypos'),end=True)
                hduimg.header.append(('an'+objno+'_i', angle,     'Obj'+objno+': Initial Rotation angle of Gauss [deg]'),end=True)
    elif (modeltype.lower() == 'galfit') or (modeltype.lower() == 'modelimg'):
        hduimg.header.append(('model',imgsize , 'File model comes from'),end=True)
    elif modeltype.lower() == 'aperture':
        Nparam = 4
        Nobj   = int(len(paramlist)/Nparam)
        if verbose: print ' - Adding aperture parameters of '+str(Nobj)+' objects used to generate model to header '

        for oo in xrange(Nobj):
            objno = str("%.4d" % (oo+1))
            yposition,xposition,radius,pixval = paramlist[oo*Nparam:oo*Nparam+Nparam]
            hduimg.header.append(('xp'+objno, xposition, 'Obj'+objno+': x position'),end=True)
            hduimg.header.append(('yp'+objno, yposition, 'Obj'+objno+': y position'),end=True)
            hduimg.header.append(('ra'+objno, radius,    'Obj'+objno+': radius in pixels of aperture'),end=True)
            hduimg.header.append(('pv'+objno, pixval,    'Obj'+objno+': pixel value of aperture'),end=True)

            if type(param_init) == np.ndarray:
                yposition,xposition,radius,pixval = param_init[oo*Nparam:oo*Nparam+Nparam]
                hduimg.header.append(('xp'+objno+'_i', xposition, 'Obj'+objno+': Initial x position'),end=True)
                hduimg.header.append(('yp'+objno+'_i', yposition, 'Obj'+objno+': Initial y position'),end=True)
                hduimg.header.append(('ra'+objno+'_i', radius,    'Obj'+objno+': radius in pixels of aperture'),end=True)
                hduimg.header.append(('pv'+objno+'_i', pixval,    'Obj'+objno+': pixel value of aperture'),end=True)
    else:
        sys.exit(' ---> "modeltype"='+modeltype+' is an invalid choice of modeling setup so aborting')


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    hdulist = pyfits.HDUList([hduimg])       # turn header into to hdulist
    hdulist.writeto(outname,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_paramlist(sourcecatalog,xpos_col='xpos',ypos_col='ypos',sigysigxangle=None,fluxscale=None,verbose=True):
    """
    Generating parameter list for image modeling when the modeltype is set to 'gauss'

    --- INPUT ---
    sourcecatalog         The source catalog to generate a parameter list for
    xpos_col              Column name of column in sourcecatalog containing the x position of sources
    ypos_col              Column name of column in sourcecatalog containing the y position of sources
    sigysigxangle         The parameters to store in parameter list (see header for gen_fullmodel() for details)
    fluxscale             Flux scale of sources
    verbose               Toggle verbosity

    --- EXAMPLE OF USE ---
    # - - - Use model cube source file as input - - -
    import tdose_model_FoV as tmf
    sourcecat  = '/Users/kschmidt/work/TDOSE/mock_cube_sourcecat161213.fits'
    paramlist   = tmf.gen_paramlist(sourcecat) # only using x and y positions
    paramlist   = tmf.gen_paramlist(sourcecat,fluxscale='fluxscale') # only using x and y positions and fluxscale
    paramlist   = tmf.gen_paramlist(sourcecat,fluxscale='fluxscale',sigysigxangle='sourcetype') # using everything

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Loading source catalog information to build mock data cube for'
    try:
        sourcedat = pyfits.open(sourcecatalog)[1].data
    except:
        sys.exit(' ---> Problems loading fits source catalog for mock cube')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Nobjects   = len(sourcedat)
    if verbose: print ' - Assembling paramter list for '+str(Nobjects)+' sources found in catalog (tmf.gen_paramlist)'
    paramlist = []
    for oo in xrange(Nobjects):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        xpos       = sourcedat[xpos_col][oo]
        ypos       = sourcedat[ypos_col][oo]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if fluxscale is not None:
            if type(fluxscale) == np.ndarray:
                fs = fluxscale[oo]
            elif type(fluxscale) == str:
                fs = sourcedat[fluxscale][oo]
        else:
            fs = 1.0
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if sigysigxangle is not None:
            if type(sigysigxangle) == np.ndarray:
                sigy  = sigysigxangle[oo,0]
                sigx  = sigysigxangle[oo,1]
                angle = sigysigxangle[oo,2]
            elif type(sigysigxangle) == str:
                st    = sourcedat[sigysigxangle][oo]
                sigx  = float(st.split('_')[1])
                sigy  = float(st.split('_')[2])
                angle = float(st.split('_')[3])
        else:
            sigx,sigy,angle = 1.0,1.0,0.0
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #            [yposition,xposition,fluxscale,sigmay,sigmax,angle]
        objlist    = [ypos,     xpos,     fs,       sigy,  sigx,  angle]
        paramlist  = paramlist + objlist
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    paramlist_arr = np.asarray(paramlist)
    return paramlist_arr
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_paramlist_aperture(sourcecatalog,radius_pix,pixval=None,xpos_col='xpos',ypos_col='ypos',verbose=True):
    """
    Generating parameter list for image modeling when the model type is set to 'aperture'

    --- INPUT ---
    sourcecatalog         The source catalog to generate a parameter list for
    radius_pix            The radius of the source apartures
    pixval                Value of pixels in each aperture, e.g., the object id
    xpos_col              Column name of column in sourcecatalog containing the x position of sources
    ypos_col              Column name of column in sourcecatalog containing the y position of sources
    verbose               Toggle verbosity

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Loading source catalog information to build parameter list for'
    try:
        sourcedat = pyfits.open(sourcecatalog)[1].data
    except:
        sys.exit(' ---> Problems loading fits source catalog for mock cube')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Nobjects   = len(sourcedat)
    if verbose: print ' - Assembling paramter list for '+str(Nobjects)+' sources found in catalog (tmf.gen_paramlist_aperture)'
    paramlist = []
    for oo in xrange(Nobjects):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        xpos       = sourcedat[xpos_col][oo]
        ypos       = sourcedat[ypos_col][oo]

        if len(radius_pix) == 1:
            radius     = radius_pix
        else:
            radius     = radius_pix[oo]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if pixval is not None:
            pv = pixval[oo]
        else:
            pv = 1.0
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #            [yposition,xposition,radius,pixval]
        objlist    = [ypos,     xpos,     radius,    pv]
        paramlist  = paramlist + objlist
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    paramlist_arr = np.asarray(paramlist)
    return paramlist_arr
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_objects_gauss(param_init,dataimage,optimizer='curve_fit',max_centroid_shift=None,
                        datanoise=None,show_residualimg=True,verbose=True):
    """
    Optimize residual between model (multiple Gaussians) and data with least squares in 2D

    --- INPUT ---
    param_init            Initial guess on parameters (defines the number of gaussians to fit for)
    dataimage             Image to model with multiple Gaussians
    optimizer             Chose the optimizer to use
                              leastsq     scipy.optimize.leastsq(); not ideal for 2D images...
                                          Tries to optimize the residual function
                              curve_fit   scipy.optimize.curve_fit()
                                          Tries to optimize using the model function
    max_centroid_shift    Maximum offset in pixels of (x,y) centroid position of sources when modeling
                          I.e. impose  ypix_centroid +/-  max_centroid_shift and xpix_centroid +/-  max_centroid_shift
                          bounds om parameters when fitting.
    datanoise             Image of sigmas, i.e., sqrt(variance) to use as weights when optimizing fit
                          using curve_fit
    verbose               Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf
    param_init = [18,31,1*0.3,2.1*0.3,1.2*0.3,30*0.3,    110,90,200*0.5,20.1*0.5,15.2*0.5,0*0.5]
    dataimg    = pyfits.open('/Users/kschmidt/work/TDOSE/mock_cube_sourcecat161213_tdose_mock_cube.fits')[0].data[0,:,:]
    param_optimized, param_cov  = tmf.model_objects_gauss(param_init,dataimg,verbose=True)

    """
    if verbose: print ' - Optimize residual between model (multiple Gaussians) and data with the optimizer "'+optimizer+'"'
    if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if optimizer == 'leastsq':
        best_param, cov, info, message, succesflag = opt.leastsq(tmf.residual_multigauss,param_init, args=(dataimage),
                                                full_output=True)
        output = best_param, cov, info, message, succesflag
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif optimizer == 'curve_fit':
        imgsize      = dataimage.shape
        xgrid, ygrid = tu.gen_gridcomponents(imgsize)
        if datanoise is not None:
            sigma = datanoise.ravel()
        else:
            sigma = datanoise

        maxfctcalls  = 30000
        Nsources     = int(len(param_init)/6.)
        #               [yposition,xposition,fluxscale,sigmay  ,sigmax  ,angle]
        param_bounds = ([0        ,0        ,0        ,1./2.355,1./2.355,0    ]*Nsources,
                        [np.inf   , np.inf  ,np.inf   , np.inf , np.inf ,360  ]*Nsources)

        Nnonfinite = len(dataimage[np.where(~np.isfinite(dataimage))])
        if Nnonfinite > 0:
            dataimage[np.where(~np.isfinite(dataimage))] = 0.0
            if verbose: print(' WARNING: '+str(Nnonfinite)+' Pixels in dataimage that are not finite; '
                                                           'setting them to 0 to prevent curve_fit crash')

        if max_centroid_shift is not None:
            init_y       = param_init[0::6]
            init_x       = param_init[1::6]

            bound_xlow   = init_x-max_centroid_shift
            bound_xhigh  = init_x+max_centroid_shift

            bound_ylow   = init_y-max_centroid_shift
            bound_yhigh  = init_y+max_centroid_shift

            param_bounds[0][0::6] = bound_ylow
            param_bounds[1][0::6] = bound_yhigh

            param_bounds[0][1::6] = bound_xlow
            param_bounds[1][1::6] = bound_xhigh

        try:
            param_optimized, param_cov = opt.curve_fit(tmf.curve_fit_function_wrapper, (xgrid, ygrid),dataimage.ravel(),
                                                       p0 = param_init, sigma=sigma,maxfev=maxfctcalls, bounds=param_bounds)
            output = param_optimized, param_cov
        except:
            print ' WARNING: Curve_fit failed (likely using "maximum function call" of '+str(maxfctcalls)+\
                  ') so returning param_init; i.e. the intiial guess of the parameters'
            output = param_init, None
            #pdb.set_trace()

    else:
        sys.exit(' ---> Invalid optimizer ('+optimizer+') chosen in model_objects_gauss()')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print '\n   ----------- Finished on '+tu.get_now_string()+' ----------- '
    #if verbose: print ' - The returned best-fit parameters are \n   ',output[0]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if show_residualimg:
        if verbose: print ' - Displaying the residual image between initial guess and optimized parameters'
        init_img = tmf.modelimage_multigauss((xgrid,ygrid), param_init , showmodelimg=False)
        best_img = tmf.modelimage_multigauss((xgrid,ygrid), output[0]  , showmodelimg=False)
        res_img  = init_img-best_img

        plt.imshow(res_img,interpolation='none', vmin=1e-5, vmax=np.max(res_img), norm=mpl.colors.LogNorm())
        plt.title('Model Residual = Initial Parameter Image - Optimized Parameter Image')
        plt.show()

        res_img  = best_img-dataimage

        plt.imshow(res_img,interpolation='none', vmin=1e-5, vmax=np.max(res_img), norm=mpl.colors.LogNorm())
        plt.title('Data Residual = Optimized Parameter Image - Data Image')
        plt.show()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return output
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def curve_fit_function_wrapper((x,y),*args):
    """
    Wrapper for curve_fit optimizer call to be able to provide list of parameters to model_objects_gauss()
    """
    # infostr = '   curve_fit_function_wrapper call at '+tu.get_now_string(withseconds=True)
    # sys.stdout.write("%s\r" % infostr)
    # sys.stdout.flush()

    paramlist = np.asarray(args)
    modelimg  = tmf.modelimage_multigauss((x,y), paramlist, showmodelimg=False, verbose=False)
    #print paramlist

    return modelimg.ravel()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def residual_multigauss(param, dataimage, nonfinite = 0.0, ravelresidual=True, showimages=False, verbose=False):
    """
    Calculating the residual bestween the multigaussian model with the paramters 'param' and the data.

    --- INPUT ---
    param         Parameters of multi-gaussian model to generate. See modelimage_multigauss() header for details
    dataimage     Data image to take residual
    nonfinite     Value to replace non-finite entries in residual with
    ravelresidual To np.ravel() the residual image set this to True. Needed by scipy.optimize.leastsq()
                  optimizer function
    showimages    To show model and residiual images set to True
    verbose       Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf
    param      = [18,31,1*0.3,2.1*0.3,1.2*0.3,30*0.3,    110,90,200*0.5,20.1*0.5,15.2*0.5,0*0.5]
    dataimg    = pyfits.open('/Users/kschmidt/work/TDOSE/mock_cube_sourcecat161213_tdose_mock_cube.fits')[0].data[0,:,:]
    residual   = tmf.residual_multigauss(param, dataimg, showimages=True)

    """
    if verbose: ' - Estimating residual (= model - data) between model and data image'
    imgsize      = dataimage.shape
    xgrid, ygrid = tu.gen_gridcomponents(imgsize)
    modelimg     = tmf.modelimage_multigauss((xgrid, ygrid),param,imgsize,showmodelimg=showimages, verbose=verbose)

    residualimg  = modelimg - dataimage

    if showimages:
        plt.imshow(residualimg,interpolation='none', vmin=1e-5, vmax=np.max(residualimg), norm=mpl.colors.LogNorm())
        plt.title('Resdiaul (= model - data) image')
        plt.show()

    if nonfinite is not None:
        residualimg[~np.isfinite(residualimg)] = 0.0

    if ravelresidual:
        residualimg = np.ravel(residualimg)

    return residualimg
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def modelimage_multigauss((xgrid,ygrid), param, showmodelimg=False, useroll=False, verbose=True, verbosefull=False):
    """
    Build model image of N Gaussians where param contains the parameters

    --- INPUT ---
    param           N x 6 long vector with the paremeters for generated N Gaussians. The paremeters needed are:
                        [yposition,xposition,fluxscale,sigmay,sigmax,angle] x N
    imgsize         Size of image to model the Gaussians in
    showmodelimg    Display the model image?
    useroll         Position model image in full FoV using a roll in the array; no interpolation so no sub-pixel
                    precision (i.e., the pixel positions are rounded when performing the roll/positioning)
    verbose         Toggle verbosity
    verbosefull     Toggle verbosity showing all details despite lengthy

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf
    xgrid, ygrid = tu.gen_gridcomponents((500,1000))
    param      = np.asarray([305,515,1,40.1,4.2,21.69,    120,100,200,20.1,15.2,0])
    modelimage = tmf.modelimage_multigauss((xgrid,ygrid), param, showmodelimg=True, verbose=True)

    import tdose_model_FoV as tmf
    xgrid, ygrid = tu.gen_gridcomponents((3000,3000))
    param      = np.asarray([305,515,1,40.1,4.2,21.69,    120,100,200,20.1,15.2,0])
    modelimage = tmf.modelimage_multigauss((xgrid,ygrid), param, showmodelimg=True, verbose=True, verbosefull=True)
    """
    Ngauss  = len(param)/6.0
    if Ngauss != np.round(len(param)/6.0):
        sys.exit(' ---> The number of parameters is not a multiple of 6 in modelimage_multigauss()')

    if verbose: print ' - Generating model for multiple ('+str(Ngauss)+') gaussians '
    if xgrid.shape != ygrid.shape:
        sys.exit(' shapes of xgrid and ygrid in modelimage_multigauss do not matach')
    imgsize    = xgrid.shape
    modelimage = np.zeros(imgsize)
    for psets in np.arange(int(Ngauss)):
        if verbose:
            infostr = '   Inserting model of object '+str("%5.f" % (psets+1))+' / '+str("%5.f" % Ngauss)+'    '
            sys.stdout.write("%s\r" % infostr)
            sys.stdout.flush()

        paramset    = param[psets*6:psets*6+6]

        covmatrix          = tu.build_2D_cov_matrix(paramset[4],paramset[3],paramset[5],verbose=verbosefull)
        gauss2Dimg         = tu.gen_2Dgauss(imgsize,covmatrix,paramset[2],show2Dgauss=False,verbose=verbosefull,method='scipy')
        if useroll:
            gauss2D_positioned = tu.roll_2Dprofile(gauss2Dimg,paramset[0:2],showprofiles=False)
        else:
            gauss2D_positioned = tu.shift_2Dprofile(gauss2Dimg,paramset[0:2],showprofiles=False,origin=1)

        modelimage         = modelimage + gauss2D_positioned

    if verbose: print '\n   done'
    if showmodelimg:
        centerdot = modelimage*0.0
        center    = [int(modelimage.shape[0]/2.),int(modelimage.shape[1]/2.)]
        centerdot[center[1],center[0]] = 2.0*np.max(modelimage)
        print ' - Center of image:',center
        plt.imshow(modelimage-centerdot,interpolation=None,origin='lower')#, vmin=1e-5, vmax=np.max(modelimage), norm=mpl.colors.LogNorm())
        plt.colorbar()
        plt.title('Model Image')
        plt.savefig('./GeneratedModelImage.pdf')
        plt.clf()

    return modelimage
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def modelimage_aperture((xgrid,ygrid), param, showmodelimg=False, verbose=True, verbosefull=False):
    """
    Build model image of N apertures where param contains the parameters for each aperture

    --- INPUT ---
    param           N x 4 long vector with the paremeters for the N apertures to generate. The paremeters needed are:
                        [yposition,xposition,radius,pixelvalues] x N
    imgsize         Size of image to model the apertures in
    showmodelimg    Display the model image?
    useroll         Position model image in full FoV using a roll in the array; no interpolation so no sub-pixel
                    precision (i.e., the pixel positions are rounded when performing the roll/positioning)
    verbose         Toggle verbosity
    verbosefull     Toggle verbosity showing all details despite lengthy

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf
    xgrid, ygrid = tu.gen_gridcomponents((500,1000))
    param      = np.asarray([305,515,50,99,    120,100,100,999])
    modelimage = tmf.modelimage_aperture((xgrid,ygrid), param, showmodelimg=True, verbose=True)

    """
    Naper  = len(param)/4.0
    if Naper != np.round(len(param)/4.0):
        sys.exit(' ---> The number of parameters is not a multiple of 4 in modelimage_aperture()')

    if verbose: print ' - Generating model for multiple ('+str(Naper)+') apertures '
    if xgrid.shape != ygrid.shape:
        sys.exit(' shapes of xgrid and ygrid in modelimage_aperture do not matach')
    imgsize    = xgrid.shape
    modelimage = np.zeros(imgsize)
    for psets in np.arange(int(Naper)):
        if verbose:
            infostr = '   Inserting model of object '+str("%5.f" % (psets+1))+' / '+str("%5.f" % Naper)+'    '
            sys.stdout.write("%s\r" % infostr)
            sys.stdout.flush()

        paramset    = param[psets*4:psets*4+4]
        apertureimg = tu.gen_aperture(imgsize,paramset[0],paramset[1],paramset[2],pixval=paramset[3],
                                      showaperture=False,verbose=verbosefull)
        modelimage  = modelimage + apertureimg

    if verbose: print '\n   done'
    if showmodelimg:
        plt.imshow(modelimage,interpolation='none', vmin=1e-5, vmax=np.max(modelimage), norm=mpl.colors.LogNorm())
        plt.title('Model Image')
        plt.show()

    return modelimage
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_objects_galfit(dataimage,galfitparamfile,show_residualimg=False,verbose=True):
    """
    ~ ~ ~ ~ STILL UNDER CONSTRUCTION/TESTING ~ ~ ~ ~

    Function template for modeling of objects with galfit

    --- INPUT ---
    dataimage
    galfitparamfile
    show_residualimg
    verbose

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf


    """
    print ' # # # # # # # # # model_objects_galfit() still under development/testing # # # # # # # # #'
    if verbose: print ' - Use GALFIT model output to obtain model parameters for residual between (GALFIT) model and data'
    if verbose: print '   ----------- Started on '+tu.get_now_string()+' ----------- '

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    galfitparams = tu.galfit_loadoutput(galfitparamfile)
    Nmodels      = len(galfitparams)
    imgsize      = dataimage.shape
    xgrid, ygrid = tu.gen_gridcomponents(imgsize)

    for oo, obj in enumerate(Nmodels):

        xxx = yyy

        output = param_optimized, param_cov

        image       = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/acs_814w_candels-cdfs-02_cut_v1.0.fits'
        sexcatalog  = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/catalog_photometry_candels-cdfs-02.fits'
        fileout     = '/Volumes/DATABCKUP3/MUSE/candels-cdfs-02/galfit_inputfile_acs_814w_candels-cdfs-02-sextractor.txt'
        tu.galfit_buildinput_fromssextractoroutput(fileout,sexcatalog,image,objecttype='gaussian',verbose=verbose)

        galfitoutput = tu.galfit_run(fileout,verbose=verbose)
        param_optimized  = tu.galfit_results2paramlist(galfitoutput,verbose=verbose)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print '   ----------- Finished on '+tu.get_now_string()+' ----------- '
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if show_residualimg:
        if verbose: print ' - Displaying the residual image between data image and galfit model (assuming gaussians)'
        galfit_img = tmf.modelimage_multigauss((xgrid,ygrid), output[0]  , showmodelimg=False)
        res_img    = dataimage-galfit_img
        plt.imshow(res_img,interpolation='none', vmin=1e-5, vmax=np.max(res_img), norm=mpl.colors.LogNorm())
        plt.title('Data Residual = Data Image - Galfit Model Image ')
        plt.show()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return output
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_objects_MoGs(param,verbose=True):
    """
    ~ ~ ~ ~ STILL UNDER CONSTRUCTION/TESTING ~ ~ ~ ~

    modellng the sources as gaussian mixture models

    --- INPUT ---

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf


    """
    if verbose: print ' - Build 2D covariance matrix with varinaces (x,y)=('+str(sigmax)+','+str(sigmay)+\
                      ') and then rotated '+str(angle)+' degrees'
    return None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =