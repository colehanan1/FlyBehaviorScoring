#!/usr/bin/env python
"""
AGGRESSIVE optimization to reach >95% accuracy for proboscis extension scoring.

Techniques:
1. XGBoost (state-of-the-art gradient boosting)
2. More PCA components (capture more signal)
3. Probability calibration
4. Threshold optimization
5. Stacked ensemble
6. Feature engineering from raw traces
