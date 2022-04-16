# Kitana: A Data-as-a-Service Platform

## Introduction

Selling data is hard. It is hard to release data, because once released publicly, there is potentially no incentive for others to pay. It is hard to price data, because its value inherently depends on the task, as well as the data a buyer already holds. And it is simply hard to manually find, assess, and trade data in today's markets. In response, this paper proposes a Data-as-a-Service (DaaS) system called Kitana that uses seller data to improve a buyer's machine learning prediction task, and does not release seller data.

Kitana manages a corpus of datasets uploaded by sellers, and exposes an AutoML interface to a buyer. Buyers submit a budget t, training data T, and validation data V, and receive a prediction API. In contrast to existing AutoML services, which solely use compute to find a predictive model for T, Kitana first uses seller data to augment T with new rows and attributes, before calling AutoML on the augmented dataset T'. Kitana assesses augmentations by training and evaluating a proxy linear regression model.

The buyer's budget constraint forces a trade-off between finding augmentations and running AutoML. Unfortunately, naively assessing an augmentation easily consumes the entire budget because materializing T' and fully re-training the proxy model can be very expensive. Instead, Kitana combines prudent pre-computation and factorized learning to quickly retrain the proxy model without materializing T'. Kitana also uses a cost model to limit the time spend on augmentation, and performs request caching to benefit from past similar requests. 

## Running Kitana

Please refer to Kitana.ipynb for the main experiments, factorized.ipynb for factorized learning micro benchmarks, Kudous.ipynb for horizontal augmentation benchmarks and request_cache.ipynb for request cache benchmars. Due to the size limit of github, datasets that are above the size of 100 MB not inlcuded.
