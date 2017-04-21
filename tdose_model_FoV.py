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
def gen_fullmodel(dataimg,sourcecatalog,modeltype='gauss',xpos_col='xpos',ypos_col='ypos',sigysigxangle=None,datanoise=None,
                  fluxscale='fluxscale',show_residualimg=False,generateimage=False,optimizer='curve_fit',
                  clobber=False,outputhdr=None,param_initguess=None,verbose=True):
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
    if fluxscale is not None:
        if verbose: print ' fluxscale',
    if sigysigxangle is not None:
        if verbose: print ' sigysigxangle',
    if verbose: print ' in initial guess'
    if param_initguess is None:
        param_init   = tmf.gen_paramlist(sourcecatalog,xpos_col=xpos_col,ypos_col=ypos_col,
                                         sigysigxangle=sigysigxangle,fluxscale=fluxscale,verbose=verbose)
    else:
        param_init   = param_initguess
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if modeltype == 'gauss':
        fit_output      = tmf.model_objects_gauss(param_init,dataimg,optimizer=optimizer,datanoise=datanoise,
                                                  verbose=verbose,show_residualimg=show_residualimg)
    elif modeltype == 'galfit':
        galfitparamfile = sourcecatalog
        fit_output      = tmf.model_objects_galfit(dataimg,galfitparamfile,verbose=verbose,show_residualimg=show_residualimg)
    else:
        sys.exit(' ---> "modeltype"='+modeltype+' is an invalid choice of modeling setup so aborting')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if type(generateimage) == str:
        tmf.save_modelimage(generateimage,fit_output[0],dataimg.shape,param_init=param_init,
                            verbose=verbose,verbosemodel=verbose,clobber=clobber,outputhdr=outputhdr)
        tmf.save_modelimage(generateimage.replace('.fits','_initial.fits'),param_init,dataimg.shape,param_init=False,
                            verbose=verbose,verbosemodel=verbose,clobber=clobber,outputhdr=outputhdr)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Storing fitted source paramters as fits table and returning output'
    tablename = generateimage.replace('.fits','_objparam.fits')

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

    th = pyfits.new_table(coldefs) # creating default header

    # writing hdrkeys:'---KEY--',                             '----------------MAX LENGTH COMMENT-------------'
    #th.header.append(('ATRACE  ' , 1.00                      ,'Factor to scale trace to total flux'),end=True)
    #th.header.append(('RA      ' , spec2D[0].header['RA']    ,'Target R.A.'),end=True)

    tbHDU  = pyfits.new_table(coldefs, header=th.header)
    tbHDU.writeto(tablename, clobber=clobber)

    return param_init, fit_output
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def save_modelimage(outname,paramlist,imgsize,param_init=False,clobber=False,outputhdr=None,verbose=True, verbosemodel=False):
    """
    Generate and save a fits file containing the model image obtained from modeling multiple gaussians

    --- INPUT ---

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Generate model image from input paramters'
    xgrid, ygrid = tu.gen_gridcomponents(imgsize)
    modelimg     = tmf.modelimage_multigauss((xgrid,ygrid), paramlist, showmodelimg=False, verbose=verbosemodel)

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
        if fluxscale is not None:
            if type(fluxscale) == np.ndarray:
                fs = fluxscale[oo]
            elif type(fluxscale) == str:
                fs = sourcedat[fluxscale][oo]
        else:
            fs = 1.0
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if sigysigxangle is not None:
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
def model_objects_gauss(param_init,dataimage,optimizer='curve_fit',datanoise=None,show_residualimg=True,verbose=True):
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
    datanoise     Image of sigmas, i.e., sqrt(variance) to use as weights when optimizing fit
                  using curve_fit
    verbose       Toggle verbosity

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
        if datanoise is not None:
            sigma = datanoise.ravel()
        else:
            sigma = datanoise

        maxfctcalls = 30000
        try:
            param_optimized, param_cov = opt.curve_fit(tmf.curve_fit_function_wrapper, (xgrid, ygrid),
                                                       dataimage.ravel(), p0 = param_init, sigma=sigma,
                                                       maxfev=maxfctcalls)
            output = param_optimized, param_cov
        except:
            print ' WARNING: Curve_fit failed (using "maximum function call" of '+str(maxfctcalls)+\
                  ') so returning param_init; i.e. the intiial guess of the parameters'
            output = param_init, None

    else:
        sys.exit(' ---> Invalid optimizer ('+optimizer+') chose in model_objects_gauss()')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print '\n   ----------- Finished on '+tu.get_now_string()+' ----------- '
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
    infostr = '   curve_fit_function_wrapper call at '+tu.get_now_string(withseconds=True)
    sys.stdout.write("%s\r" % infostr)
    sys.stdout.flush()

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
    showmodelimg    Displaye the model image?
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
        sys.exit(' ---> The number of parameters is not a multiple of 6 in residual_multigauss()')

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
            gauss2D_positioned = tu.roll_2Dprofile(gauss2Dimg,paramset[0:2]-1.0,showprofiles=False)
        else:
            gauss2D_positioned = tu.shift_2Dprofile(gauss2Dimg,paramset[0:2]-1.0,showprofiles=False)
        modelimage         = modelimage + gauss2D_positioned

    if verbose: print '\n   done'
    if showmodelimg:
        plt.imshow(modelimage,interpolation='none', vmin=1e-5, vmax=np.max(modelimage), norm=mpl.colors.LogNorm())
        plt.title('Model Image')
        plt.show()

    return modelimage
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def model_objects_galfit(dataimage,galfitparamfile,show_residualimg=False,verbose=True):
    """
    model

    --- INPUT ---

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
    model

    --- INPUT ---

    --- EXAMPLE OF USE ---
    import tdose_model_FoV as tmf


    """
    if verbose: print ' - Build 2D covariance matrix with varinaces (x,y)=('+str(sigmax)+','+str(sigmay)+\
                      ') and then rotated '+str(angle)+' degrees'
    return None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =