# Active Learning

Implementation of active learning loop for binary classification on an highly
imbalanced dataset. The project implements Random Forest and SVM classifiers
which are used in the prolonged learning process. Furthermore, implementation
supports algorithm-independent methods that are used for active learning
initalization (random and cluster-based) and query selection (uncertainty,
diversity and random smapling). Predictive performacne is evaluated using the
area under the precision-recall curve, estimated by k-fold cross-validation, and
precision-recall curves at different stages of learning.

Detailed documentation and experiment results can be found in
[docs/docs.pdf](docs/docs.pdf).
