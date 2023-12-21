## --- `innsight` 0.3.0 ------------------------------------------------------

### Test environments with LibTorch
* GitHub Actions (ubuntu-22.04): 4.1, 4.2, release, devel
* GitHub Actions (windows): release
* Github Actions (macOS): release

#### R CMD check results

There were no errors or warnings only one note which is not related to our 
package and caused by `keras` (in `keras`, these files are cleaned up after
the tests' execution; see [here](https://github.com/rstudio/keras/blob/eb5d21b9e37e918c2662eb6ec5bcc46a00054db6/tests/testthat/setup.R))

```
* checking for detritus in the temp directory ... NOTE
Found the following files/directories:
  ‘__autograph_generated_filejdw6cqsg.py’ ‘__pycache__’
```

**Note:** We can't run examples, tests or vignettes on CRAN, as this 
requires a successful installation of LibTorch/Lantern. Every implemented method 
relies on an instance of `Converter` that converts a passed model to a 
torch model, so any possibility of examples or (non-trivial) tests requires 
LibTorch/Lantern. In this regard, we have followed the recommendations 
of the authors of torch (see torch 
[issue #651](https://github.com/mlverse/torch/issues/651#issuecomment-896783144))
and disabled their execution on CRAN.

### Test environments without LibTorch
- winbuilder Windows Server 2022, R-devel, 64 bit
- winbuilder Windows Server 2022, R-release, 64 bit
- winbuilder Windows Server 2022, R-oldrel, 64 bit
- R-hub Ubuntu Linux 20.04.1 LTS, R-release
- R-hub Fedora Linux, R-devel, clang, gfortran
- macOS builder, R-release

#### R CMD check results

There were no errors or warnings, only some notes under R-Hub unrelated to the 
package: (see issues [#548](https://github.com/r-hub/rhub/issues/548), 
[#560](https://github.com/r-hub/rhub/issues/560),
[#503](https://github.com/r-hub/rhub/issues/503)):

```
* checking HTML version of manual ... NOTE
  Skipping checking HTML validation: no command 'tidy' found
  Skipping checking math rendering: package 'V8' unavailable
* checking for non-standard things in the check directory ... NOTE
  Found the following files/directories:
    ''NULL''
* checking for detritus in the temp directory ... NOTE
  Found the following files/directories:
    'lastMiKTeXException'
```
