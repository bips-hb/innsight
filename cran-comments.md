## --- `innsight` 0.2.0 ------------------------------------------------------

### Test environments with LibTorch
* GitHub Actions (ubuntu-20.04): 3.5, 3.6, 4.1, release, devel
* GitHub Actions (windows): 3.6, 4.1, release
* Github Actions (macOS): release

#### R CMD check results

There were no errors or relevant notes and the following warning occurred only on the
operating system Windows:

```
Warning: Found the following significant warnings:
  Warning: i torch failed to start, restart your R session to try again.
```
This warning comes from the fact that the Windows machine has an old version 
of Microsoft Visual C++ Redistributable (version 10.0.40219 from 2010) 
preinstalled, but according to the torch 
[issue #246](https://github.com/mlverse/torch/issues/246#issuecomment-695097121), 
the latest version is required.

In addition, we can't run examples, tests or vignettes on CRAN, as this 
requires a successful installation of LibTorch. Every implemented method 
relies on an instance of `Converter` that converts a passed model to a 
torch model, so any possibility of examples or (non-trivial) tests requires 
LibTorch. In this regard, we have followed the recommendations of the authors 
of torch (see torch 
[issue #651](https://github.com/mlverse/torch/issues/651#issuecomment-896783144))
and disabled their execution on CRAN.

### Test environments without LibTorch
- R-hub Windows Server 2022, R-devel, 64 bit
- R-hub Ubuntu Linux 20.04.1 LTS, R-release, GCC
- R-hub Fedora Linux, R-devel, clang, gfortran

#### R CMD check results

There were no errors or warnings only some notes which are not related to
the package: 

* Windows:
```
❯ checking HTML version of manual ... NOTE
  Skipping checking math rendering: package 'V8' unavailable

❯ checking for detritus in the temp directory ... NOTE
  Found the following files/directories:
    'lastMiKTeXException'
```

* Linux
```
❯ checking HTML version of manual ... NOTE
  Skipping checking HTML validation: no command 'tidy' found
  Skipping checking math rendering: package 'V8' unavailable
```


## --- `innsight` 0.1.1 ------------------------------------------------------

### Test environments with LibTorch
* GitHub Actions (ubuntu-20.04): 3.5, 3.6, 4.0, release, devel
* GitHub Actions (windows): 3.6, release
* Github Actions (macOS): release

#### R CMD check results

There were no errors or relevant notes and the following warning occurred only on the
operating system Windows:

```
Warning: Found the following significant warnings:
  Warning: Torch failed to start, restart your R session to try again.
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

### Test environments without LibTorch
- R-hub Windows Server 2022, R-devel, 64 bit
- R-hub Ubuntu Linux 20.04.1 LTS, R-release, GCC
- R-hub Fedora Linux, R-devel, clang, gfortran

#### R CMD check results

There were no errors or warnings only two notes: 

* One on Windows: a bug in `rhub` that can be ignored ([see here](https://github.com/r-hub/rhub/issues/503#issue-1105809885/))
```
  Build ID:   innsight_0.1.1.tar.gz-5b65ac0c5d8648e88cdc6f523893c731
  Platform:   Windows Server 2022, R-devel, 64 bit
  Submitted:  1h 1m 3.8s ago
  Build time: 4m 42.9s

❯ checking for detritus in the temp directory ... NOTE
  Found the following files/directories:
    'lastMiKTeXException'

0 errors ✔ | 0 warnings ✔ | 1 note ✖
```

* One note on Linux that is irrelevant for the submission
```
  Build ID:   innsight_0.1.1.tar.gz-831d4fbd980b4a5eaba89f9f69499522
  Platform:   Fedora Linux, R-devel, clang, gfortran
  Submitted:  1h 1m 3.8s ago
  Build time: 31m 21.3s

❯ checking HTML version of manual ... NOTE
  Skipping checking HTML validation: no command 'tidy' found
  Skipping checking math rendering: package 'V8' unavailable

0 errors ✔ | 0 warnings ✔ | 1 note ✖
```
## --- `innsight` 0.1.0 ------------------------------------------------------

### Test environments with LibTorch
* GitHub Actions (ubuntu-18.04): 3.5, 3.6, 4.0, release, devel
* GitHub Actions (windows): 3.6, release
* Github Actions (macOS): release

#### R CMD check results

There were no errors or notes and the following warning occurred only on the
operating system Windows (see 
[release](https://github.com/bips-hb/innsight/runs/4266780339?check_suite_focus=true#step:12:44) and [3.6](https://github.com/bips-hb/innsight/runs/4266780419?check_suite_focus=true#step:12:44)):

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

### Test environments without LibTorch
- R-hub windows-x86_64-devel (r-devel)
- R-hub ubuntu-gcc-release (r-release)
- R-hub fedora-clang-devel (r-devel)
- Win-builder (r-oldrelease)
- Win-builder (r-release)
- Win-builder (r-devel)

#### R CMD check results

There were no errors or warnings only one note: New submission and 
irrelevant misspellings in names in DESCRIPTION:

```
checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Niklas Koenen <niklas.koenen@gmail.com>'
  New submission
  
  Possibly misspelled words in DESCRIPTION:
    Baehrens (24:5)
  
    Interpretability (15:14)
    Shrikumar (21:31)
    Smilkov (22:63)
    al (17:53, 19:46, 21:44, 23:5, 24:17)
    et (17:50, 19:43, 21:41, 22:71, 24:14)

0 errors ✓ | 0 warnings ✓ | 1 note x
```

