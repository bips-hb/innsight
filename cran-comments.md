## Test environments
* GitHub Actions (ubuntu-18.04): 3.5, 3.6, 4.0, release, devel
* GitHub Actions (windows): 3.6, release
* Github Actions (macOS): release

## R CMD check results

There was no errors or notes and the following warning occurred only on the
operating system Windows (see 
[release](https://github.com/bips-hb/innsight/runs/4253363447?check_suite_focus=true#step:12:44) and [latest](https://github.com/bips-hb/innsight/runs/4253363511?check_suite_focus=true#step:12:44)):

```
Warning: Found the following significant warnings:
  Warning: Torch failed to start, restart your R session to try again. D:\a\_  temp\Library\torch\deps\lantern.dll - %1 is not a valid Win32 application.
```
This warning comes from the fact that the Windows machine has an old version 
of Microsoft Visual C++ Redistributable (version 10.0.40219 from 2010) 
preinstalled, but according to the torch 
[issue #246](https://github.com/mlverse/torch/issues/246#issuecomment-695097121), 
the latest version is required.

In addition, we can't run examples, tests or vignettes on CRAN, as this 
requires a successful installation of LibTorch. Every implemented method 
relies on an instance of 'Converter' that converts a passed model to a 
torch model, so any possibility of examples or (non-trivial) tests requires 
LibTorch. In this regard, we have followed the recommendations of the authors 
of torch (see torch 
[issue #651](https://github.com/mlverse/torch/issues/651#issuecomment-896783144))
and disabled their execution on CRAN.
