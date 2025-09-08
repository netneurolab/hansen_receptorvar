# Inter-individual variability of neurotransmitter receptor and transporter density in the human brain
This repository contains code and data in support of "Inter-individual variability of neurotransmitter receptor and transporter density in the human brain", available as a preprint on [bioRxiv]().
All code was written in Python 3.8.10.
Below I describe the contents of this repository.

## `code`
The [code](code/) folder contains a single file, [main.py](code/main.py), which was used to run the analyses and generate the figures.

## `data`
The [data](data/) folder contains data files used for the analyses. If you use this data in your own analyses, please cite the associated papers.

- [PET_volumes](data/PET_volumes) contains group mean and standard deviation volumetric images from PET tracer studies. Many of these images were originally shared in [Hansen et al 2022 Nat Neurosci](https://www.nature.com/articles/s41593-022-01186-3) (associated GitHub repo [here](https://github.com/netneurolab/hansen_receptors)). The file naming convention is _receptor name_ followed by _tracer name_ followed by _number of healthy controls_ followed by _last name of first author of original publication_ followed by either "mean" or "sd". Note that full reference details can be found in Table 1 of this manuscript.
- The two `Tian_Subcortex_S4_3T` files refer to the 54-region subcortical atlas presented in [Tian et al 2020 Nat Neurosci](https://www.nature.com/articles/s41593-020-00711-6). I use them to parcellate and plot the subcortex.

## `manuscript`
The [manuscript](manuscript/) folder contains the PDF of the manuscript.
