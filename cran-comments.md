## --- `innsight` 0.2.1 ------------------------------------------------------

### Test environments with LibTorch
* GitHub Actions (ubuntu-22.04): 3.5, 3.6, 4.0, 4.1, release, devel
* GitHub Actions (windows): release
* Github Actions (macOS): release

#### R CMD check results

No warnings or errors occurred.

**Note:** We can't run examples, tests or vignettes on CRAN, as this 
requires a successful installation of LibTorch. Every implemented method 
relies on an instance of `Converter` that converts a passed model to a 
torch model, so any possibility of examples or (non-trivial) tests requires 
LibTorch. In this regard, we have followed the recommendations of the authors 
of torch (see torch 
[issue #651](https://github.com/mlverse/torch/issues/651#issuecomment-896783144))
and disabled their execution on CRAN.

### Test environments without LibTorch
- winbuilder Windows Server 2022, R-devel, 64 bit
- winbuilder Windows Server 2022, R-release, 64 bit
- R-hub Ubuntu Linux 20.04.1 LTS, R-release, GCC
- R-hub Fedora Linux, R-devel, clang, gfortran
- macOS builder, R-release

#### R CMD check results

There were no errors or warnings only some notes which are not related to
the package: 

```
* checking HTML version of manual ... NOTE
Skipping checking HTML validation: no command 'tidy' found
Skipping checking math rendering: package 'V8' unavailable
```
