# AI-powered printability evaluation framework for 3D bioprinting using Hausdorff distance metrics

This repository contains scripts used for the paper "AI-powered printability evaluation framework for 3D bioprinting using Hausdorff distance metrics."

## Abstract
3D bioprinting enables rapid fabrication of complex biological structures for tissue engineering applications. However, optimizing bioink formulation remains challenging due to complex relationships among material properties, printability, and cell viability. While the perimeter ratio (Pr) is commonly used to evaluate printability, it cannot adequately capture the full geometric fidelity required for comprehensive printability assessments, thereby limiting robust bioink design. To address this limitation, a novel Hausdorff distance (HD) metric is employed to quantify printability, directly measuring the maximum deviation between the designed and printed structures. Furthermore, multiple machine-learning approaches were applied to alginate-hyaluronic acid composite inks and rat pheochromocytoma-derived PC12 cells to assess printability and cell viability. Rheological parameters were characterized using support vector regression (SVR) with R² ⩾ 0.974. Multi-layer perceptron (MLP) regressors achieved R² values of 0.932 and 0.945 when predicting HD values of printed grid structures and cell viability, respectively. A regression-based convolutional neural network (CNN) was developed to predict HD values directly from grid structure images, achieving an R² of 0.986. Through optimization, optimal as-extruded cell viability (⩾95%) was achieved while maintaining high printability (HD ⩽ 0.20). The optimal ink composition was further demonstrated with good long-term cell viability and proliferation potential. This proposed AI-integrated approach can dramatically reduce ink optimization time by rapidly predicting rheological properties, printability, and cell viability from minimal experimental data.

**Keywords:** Additive manufacturing; 3D bioprinting; Hausdorff distance; printability; rheology; machine learning; convolutional neural network

<a href="https://iopscience.iop.org/article/10.1088/1758-5090/ae288c">[Article Link]</a>
<br>
<a href="https://www.researchgate.net/publication/398396117_AI-powered_printability_evaluation_framework_for_3D_bioprinting_using_Hausdorff_distance_metrics">[Publisher Preview]</a>
<br>
<a href="https://ir.library.osaka-u.ac.jp/repo/ouka/all/104697/?lang=1">[Accepted Manuscript]</a>

## Graphical Abstract
<p align="center">
  <a href="https://doi.org/10.1080/17452759.2026.2671497">
    <img width="800" alt="GraphicalAbstract2" src="https://github.com/user-attachments/assets/dd0aeba6-2659-4e10-bcb3-eee5a5e65188" />
  </a>
  <br>
  <sub>Derived from author-prepared materials used in the pre-acceptance manuscript version of Zhang, C. et al., <i>Biofabrication</i> <b>18</b>, 015015 (2026).</sub>
</p>

## Citation
Colin Zhang, Kelum Elvitigala, and Shinji Sakai.
AI-powered printability evaluation framework for 3D bioprinting using Hausdorff distance metrics. <i>Biofabrication</i> <b>18</b>, 015015 (2026). <a href="https://doi.org/10.1088/1758-5090/ae288c">https://doi.org/10.1088/1758-5090/ae288c</a>.

```bibtex
@article{zhang_ai_2026,
author = {Colin Zhang and Kelum Elvitigala and Shinji Sakai},
title = {AI-powered printability evaluation framework for 3D bioprinting using Hausdorff distance metrics},
journal = {Biofabrication},
volume = {18},
number = {1},
pages = {015015},
year = {2026},
month = {mar},
publisher = {IOP Publishing},
doi = {10.1088/1758-5090/ae288c},
url = {https://doi.org/10.1088/1758-5090/ae288c},
}
```

## Disclaimer

Please note that some scripts in this repository might require the corresponding data files to run successfully. The data files are available upon reasonable request from the corresponding author. This repository is intended for educational and research purposes. The authors are not responsible for any misuse of the code or data provided herein. Users are encouraged to cite the original manuscript when using this code in their research.
