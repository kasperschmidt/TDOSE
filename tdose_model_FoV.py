# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import numpy as np
import os
import sys
import pyfits
import scipy.ndimage
import scipy.optimize as opt
import tdose_utilities as tu
import matplotlib as mpl
import matplotlib.pylab as plt
import tdose_model_FoV as tmf
import pdb
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_fullmodel(dataimg,sourcecatalog,xpos_col='xpos',ypos_col='ypos',sigysigxangle=None,
                  fluxscale='fluxscale',show_residualimg=False,generateimage=False,optimizer='curve_fit',
                  clobber=False,verbose=True):
    """
    model

    --- INPUT ---

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
    if verbose: print ' - Loading source catalog info to build inital guess of paramters for model fit'
    if verbose: print '   Will use x position, y position,',
    if fluxscale != None:
        if verbose: print '   fluxscale',
    if fluxscale != None:
        if verbose: print '   sigysigxangle',
    if verbose: print ' in initial guess'
    param_init   = tmf.gen_paramlist(sourcecatalog,xpos_col=xpos_col,ypos_col=ypos_col,
                                     sigysigxangle=sigysigxangle,fluxscale=fluxscale,verbose=verbose)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fit_output  = tmf.model_objects_gauss(param_init,dataimg,optimizer=optimizer,
                                          verbose=verbose,show_residualimg=show_residualimg)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if type(generateimage) == str:
        tmf.save_modelimage(generateimage,fit_output[0],dataimg.shape,param_init=param_init,
                            verbose=verbose,clobber=clobber)
        tmf.save_modelimage(generateimage.replace('.fits','_initial.fits'),param_init,dataimg.shape,param_init=False,
                            verbose=verbose,clobber=clobber)

    return param_init, fit_output
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def save_modelimage(outname,paramlist,imgsize,param_init=False,clobber=False,outputhdr=None,verbose=True):
    """
    Generate and save a fits file containing the model image obtained from modeling multiple gaussians

    --- INPUT ---

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Generate model image from input paramters'
    xgrid, ygrid = tu.gen_gridcomponents(imgsize)
    modelimg     = tmf.modelimage_multigauss((xgrid,ygrid), paramlist, showmodelimg=False, verbose=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Saving generated image to '+outname
    if outputhdr == None:
        if verbose: print ' - No header provided so will generate one '
        hduimg = pyfits.PrimaryHDU(modelimg)       # creating default fits header

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
        hduimg = outputhdr
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Nobj = int(len(paramlist)/6.0)
    if verbose: print ' - Adding parameters of '+str(Nobj)+' objects used to generate model to header '

    for oo in xrange(Nobj):
        objno = str("%.4d" % (oo+1))
        yposition,xposition,fluxscale,sigmay,sigmax,angle = paramlist[oo*6:oo*6+6]
        hduimg.header.append(('xp'+objno, xposition, 'Obj'+objno+': x position'),end=True)
        hduimg.header.append(('yp'+objno, yposition, 'Obj'+objno+': y position'),end=True)
        hduimg.header.append(('fs'+objno, fluxscale, 'Obj'+objno+': flux scale applied to Gauss'),end=True)
        hduimg.header.append(('xs'+objno, sigmax,    'Obj'+objno+': Standard deviation in xpos'),end=True)
        hduimg.header.append(('ys'+objno, sigmay,    'Obj'+objno+': Standard deviation in ypos'),end=True)
        hduimg.header.append(('an'+objno, angle,     'Obj'+objno+': Rotation angle of Gauss [deg]'),end=True)

        if type(param_init) == np.ndarray:
            yposition,xposition,fluxscale,sigmay,sigmax,angle = param_init[oo*6:oo*6+6]
            hduimg.header.append(('xp'+objno+'_i', xposition, 'Obj'+objno+': Initial x position'),end=True)
            hduimg.header.append(('yp'+objno+'_i', yposition, 'Obj'+objno+': Initial y position'),end=True)
            hduimg.header.append(('fs'+objno+'_i', fluxscale, 'Obj'+objno+': Initial flux scale applied to Gauss'),end=True)
            hduimg.header.append(('xs'+objno+'_i', sigmax,    'Obj'+objno+': Initial Standard deviation in xpos'),end=True)
            hduimg.header.append(('ys'+objno+'_i', sigmay,    'Obj'+objno+': Initial Standard deviation in ypos'),end=True)
            hduimg.header.append(('an'+objno+'_i', angle,     'Obj'+objno+': Initial Rotation angle of Gauss [deg]'),end=True)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    hdulist = pyfits.HDUList([hduimg])       # turn header into to hdulist
    hdulist.writeto(outname,clobber=clobber)  # write fits file (clobber=True overwrites excisting file)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def gen_paramlist(sourcecatalog,xpos_col='xpos',ypos_col='ypos',sigysigxangle=None,fluxscale=None,verbose=True):
    """
    Generating parameter list for image modeling

    --- INPUT ---
    sourcecatalog
    xpos_col='xpos'
    ypos_col='ypos'
    sigysigxangle=None
    fluxscale=None
    verbose=True

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
    if verbose: print ' - Assembling paramter list for '+str(Nobjects)+' found in catalog'
    paramlist = []
    for oo in xrange(Nobjects):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        xpos       = sourcedat[xpos_col][oo]
        ypos       = sourcedat[ypos_col][oo]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if fluxscale != None:
            if type(fluxscale) == np.ndarray:
                fs = fluxscale[oo]
            elif type(fluxscale) == str:
                fs = sourcedat[fluxscale][oo]
        else:
            fs = 1.0
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if sigysigxangle != None:
            if type(fluxscale) == np.ndarray:
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
def model_objects_gauss(param_init,dataimage,optimizer='curve_fit',show_residualimg=True,verbose=True):
    """
    Optimize residual between model (multiple Gaussians) and data with least squares in 2D

    --- INPUT ---
    param_init    Initial guess on parameters (defines the number of gaussians to fit for)
    dataimage     Image to model with multiple Gaussians
    optimizer     Chose the optimizer to use
                      leastsq     scipy.optimize.leastsq(); not ideal for 2D images...
                                  Tries to optimize the residual function
                      curve_fit   scipy.optimize.curve_fit()
                                  Tries to optimize using the model function

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf
    param_init = [18,31,1*0.3,2.1*0.3,1.2*0.3,30*0.3,    110,90,200*0.5,20.1*0.5,15.2*0.5,0*0.5]
    dataimg    = pyfits.open('/Users/kschmidt/work/TDOSE/mock_cube_sourcecat161213_tdose_mock_cube.fits')[0].data[0,:,:]
    param_optimized, param_cov  = tmf.model_objects_gauss(param_init,dataimg,verbose=True)

    """
    if verbose: print ' - Optimize residual between model (multiple Gaussians) and data with least squares in 2D'
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
        param_optimized, param_cov = opt.curve_fit(tmf.curve_fit_function_wrapper, (xgrid, ygrid),
                                                   dataimage.ravel(), p0 = param_init)

        output = param_optimized, param_cov
    else:
        sys.exit(' ---> Invalid optimizer ('+optimizer+') chose in model_objects_gauss()')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print '   ----------- Finished on '+tu.get_now_string()+' ----------- '
    #if verbose: print ' - The returned best-fit parameters are \n   ',output[0]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if show_residualimg:
        if verbose: print ' - Displaying the residual image between initial guess and optimized paramters'
        init_img = tmf.modelimage_multigauss((xgrid,ygrid), param_init , showmodelimg=False)
        best_img = tmf.modelimage_multigauss((xgrid,ygrid), output[0]  , showmodelimg=False)
        res_img  = init_img-best_img

        plt.imshow(res_img,interpolation='none', vmin=1e-5, vmax=np.max(res_img), norm=mpl.colors.LogNorm())
        plt.title('Model Residual = Initial Parameter Image - Optimized Paramter Image')
        plt.show()

        res_img  = best_img-dataimage

        plt.imshow(res_img,interpolation='none', vmin=1e-5, vmax=np.max(res_img), norm=mpl.colors.LogNorm())
        plt.title('Data Residual = Optimized Paramter Image - Data Image')
        plt.show()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return output
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def curve_fit_function_wrapper((x,y),*args):
    """
    Wrapper for curve_fit optimizer call to be able to provide list of parameters to model_objects_gauss()
    """
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

    if nonfinite != None:
        residualimg[~np.isfinite(residualimg)] = 0.0

    if ravelresidual:
        residualimg = np.ravel(residualimg)

    return residualimg
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def modelimage_multigauss((xgrid,ygrid), param, showmodelimg=False, verbose=True):
    """
    Build model image of N Gaussians where param contains the paremters

    --- INPUT ---
    param           N x 6 long vector with the paremeters for generated N Gaussians. The paremeters needed are:
                        [yposition,xposition,fluxscale,sigmay,sigmax,angle] x N
    imgsize         Size of image to model the Gaussians in
    showmodelimg    Displaye the model image?
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf

    param      = [35,15,1,2.1,1.2,30,    120,100,200,20.1,15.2,0]
    modelimage = tmf.modelimage_multigauss(param, [200,150], showmodelimg=True, verbose=True)

    """
    Ngauss  = len(param)/6.0
    if Ngauss != np.round(len(param)/6.0):
        sys.exit(' ---> The number of parameters is not a multiple of 6 in residual_multigauss()')

    if verbose: print ' - Generating model for multiple ('+str(Ngauss)+') gaussians '
    if xgrid.shape != ygrid.shape:
        sys.exit(' shapes of xgrid and ygrid in modelimage_multigauss do not matach')
    imgsize    = xgrid.shape
    modelimage = np.zeros(imgsize)
    for psets in np.arange(int(Ngauss)):
        paramset    = param[psets*6:psets*6+6]

        covmatrix          = tu.build_2D_cov_matrix(paramset[4],paramset[3],paramset[5],verbose=verbose)
        gauss2Dimg         = tu.gen_2Dgauss(imgsize,covmatrix,paramset[2],show2Dgauss=False,verbose=verbose)
        gauss2D_positioned = tu.roll_2Dprofile(gauss2Dimg,paramset[0:2])

        modelimage         = modelimage + gauss2D_positioned

    if showmodelimg:
        plt.imshow(modelimage,interpolation='none', vmin=1e-5, vmax=np.max(modelimage), norm=mpl.colors.LogNorm())
        plt.title('Model Image')
        plt.show()

    return modelimage
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_objects_galfit(param,verbose=True):
    """
    model

    --- INPUT ---

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf


    """
    if verbose: print ' - Build 2D covariance matrix with varinaces (x,y)=('+str(sigmax)+','+str(sigmay)+\
                      ') and then rotated '+str(angle)+' degrees'
    return None
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_objects_MoGs(param,verbose=True):
    """
    model

    --- INPUT ---

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf


    """
    if verbose: print ' - Build 2D covariance matrix with varinaces (x,y)=('+str(sigmax)+','+str(sigmay)+\
                      ') and then rotated '+str(angle)+' degrees'
    return None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =