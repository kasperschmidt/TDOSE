# TDOSE Examples

This directory contains the data, directory structure, and scripts to demonstrate TDOSE functionality and to perform a series of extractions. Spectra are extracted for two objects with original data taken from the MXDF presented for the first time by [Bacon et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021arXiv210205516B).

Jupyter python notebooks (as well as the corresponding `*.py` files) of the examples are available in the sub-directory [`examples_scripts`](https://github.com/kasperschmidt/TDOSE/tdose_examples/examples_scripts)
The Jupyter notebooks were generated with [`p2j`](https://github.com/remykarem/python2jupyter).

Hence, after installing and setting up TDOSE a test extraction of an `aperture`, `modelimg`, and `gauss` based run can be executed on the test data by simply running
```
python tdose_perform_spectral_extraction.py
```
from the command line or by going through the corresponding Jupyter notebook. In a similar way, other examples can be run/executed when they become available.

As of 210305 the full run of `tdose_perform_spectral_extraction.py` extracting an `aperture`-, `modelimg`-, and `gauss`-based TDOSE spectrum of the two objects with provided data cube cutouts takes just roughly 10 minutes on an old Mac from 2015 in non-parallel. The extractions ran succesfully with both Python 2.7.5 and Python 3.6.3.

Hopefully these examples can help people test new installations, updates or modifications to TDOSE made locally. With this in mind and for comparison purposes the sub-directory [`output_210305`](https://github.com/kasperschmidt/TDOSE/tdose_examples/output_210305/) contains the output from a full extraction performed by @kasperschmidt on March 5th 2021. Note however that the aperture and gauss `modelcubes` for 5082 were not moved to GitHub as these are larger than 100MB. These are instead available for downloard from [Google drive](https://drive.google.com/drive/folders/1xr_MHt6FdK39ylYpe1D0vPR7wY8N7sko?usp=sharing). The full extraction producing these outputs was performed using the Python 3 version of the TDOSE software from March 3rd, 2021. This version is essentially the same as TDOSE version 3.0 released with and described by [Schmidt et al. (2019)](http://ui.adsabs.harvard.edu/abs/2019arXiv190605891S). 

The [`tdose_examples`](https://github.com/kasperschmidt/TDOSE/tree/master/tdose_examples)  directory therefore compliments the appendices of [Schmidt et al. (2019)](http://ui.adsabs.harvard.edu/abs/2019arXiv190605891S) and the header of the individual TDOSE functions and scripts, which provide useful commands and approaches for using TDOSE.