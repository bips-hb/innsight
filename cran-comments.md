## --- `innsight` 0.3.2 --------------------------------------------------------

### Test environments with LibTorch
* GitHub Actions (ubuntu-22.04): 4.2, 4.3, release, devel
* GitHub Actions (windows): release
* Github Actions (macOS): release

#### R CMD check results

There were no errors or warnings only one note which is not related to our 
package and caused by `keras` (in `keras`, these files are cleaned up after
the tests' execution; see [here](https://github.com/rstudio/keras/blob/eb5d21b9e37e918c2662eb6ec5bcc46a00054db6/tests/testthat/setup.R))

```
* checking for detritus in the temp directory ... NOTE
Found the following files/directories:
  ‘__autograph_generated_filet4_ztm6r.py’ ‘__pycache__’
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
- macOS builder, R-release
- Windows builder, R-release
  
#### R CMD check results

There were no errors, warnings or notes (only the note on the runtime for the
example on ConnectionWeights, however, it is due to loading the `keras` package).
