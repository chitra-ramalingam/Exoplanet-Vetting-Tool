# Exoplanet-Vetting-Tool [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abhmalik/Exoplanet-Vetting-Tool/master)
The vetting tool is to help people explore the results from our exoplanet detection model from the paper: [Exoplanet Detection using Machine Learning](https://arxiv.org/abs/2011.14135)
With the tool you can explore some of the predicted planet cases in an interactive way without any knowledge of the underlying machine learning methods and the feature calculation techniques.

## How to run the tool:
1. [open binder](https://mybinder.org/v2/gh/abhmalik/Exoplanet-Vetting-Tool/master) and run `Vetting-Tool.ipynb`
2. Run both the code blocks and the tools will initiate. After you successfully start the tool you should see the following buttons:
![preview](./data/tool.png)

## Structure of the tool:
- Global View: Full view of the lightcurve
- Folded View: Folded or local view i.e. lightcurve shown in phase
- SHAP Values: SHAP Values breaks down a prediction to show the impact of each feature. Here you'll see top 10 features that were important for this prediction.
