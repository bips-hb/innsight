## --- `innsight` 0.3.1 --------------------------------------------------------

* The DOI in the CITATION is for a new JSS publication that will be registered 
after publication on CRAN.

### Test environments with LibTorch
* GitHub Actions (ubuntu-22.04): 4.2, 4.3, release, devel
* GitHub Actions (windows): release
* Github Actions (macOS): release

**Note:** The creation of vignettes using the `luz` package is currently 
failing on MacOS, but this is not due to our package (see [issue #1213](https://github.com/mlverse/torch/issues/1213)
in`torch` and [issue #143](https://github.com/mlverse/luz/issues/143) in `luz`).

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
- R-hub Ubuntu Linux 22.04 R-release
- R-hub Ubuntu Linux, R-devel
- R-hub Windows, R-devel
- R-hub macOS, R-devel
- macOS builder, R-release

#### R CMD check results

There were no errors, warnings or notes (only the already mentioned note on 
the DOI).
