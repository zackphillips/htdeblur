# htdeblur: High-throughput imaging using motion deblur
***Work in progress - API may change without warning***

## Requirements
numpy, scipy, [llops](http://www.github.com/zfphil/llops)

## Installation
```shell
git clone https://www.github.com/zfphil/htdeblur
cd illuminate_controller
python setup.py build
python setup.py install
```

## Submodule List
- ***htdeblur.blurkernel***: Functions for creating and manipulating blur kernels
- ***htdeblur.pgd***: Functions related to object reconstructions
- ***htdeblur.io***: [Depreciated]
- ***htdeblur.project_simplex_box***: Functions to project into a simplex
- ***htdeblur.visualization***: Functions for the visualization of blur kernels and reconstructions

## License
BSD 3-clause
