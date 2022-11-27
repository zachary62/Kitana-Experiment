# Kitana: A Data-as-a-Service Platform

## Introduction
AutoML services provide a way for non-expert users to benefit from high-quality ML models without worrying about model design and deployment, in exchange for a charge per hour ($\$21.252$ for VertexAI). However, ML models are only as good as the quality of training data. With the increasing volume of data available, both within enterprises and to the public, there is a huge opportunity for training data augmentation. For instance, vertical augmentation can add predictive features, while horizontal augmentation can add samples. These augmented training data can potentially yield much better ML models through AutoML at a lower cost. However, previous AutoML and data augmentation systems either forgo the augmentation opportunities that provide poor models, or apply expensive augmentation searching that drain users' budgets.

We present Kitana, an AutoML system with practical data augmentation searching. \sys manages a corpus of datasets from enterprises or public, and exposes an AutoML interface to users. To search for augmentation opportunities at minimum costs, \sys applies aggressive pre-computation  to train *factorized proxy model* and evaluate augmentation candidates within $0.1s$. Kitana also uses a cost model to limit the time spend on augmentation search, supports access controls, and performs request caching to benefit from past similar requests. 
Using a corpus of 513 open-source datasets, we show that Kitana produces higher quality models than existing AutoML systems in orders of magnitude less time. Across different user requests, we find Kitana can increase the model R2 from ${\sim}0.16\to {\sim}0.66$ while reducing the cost by ${>}100\times$.

## Running Kitana

The main implementation of Kitana is in datamarket.py. Please refer to Kitana.ipynb for the main experiments, factorized.ipynb for factorized learning micro benchmarks, Kudous.ipynb for horizontal augmentation benchmarks and request_cache.ipynb for request cache benchmars. Due to the size limit of github, datasets that are above the size of 100 MB are not inlcuded.
