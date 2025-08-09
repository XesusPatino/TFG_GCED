# LightGCN: Simplified and Powerful Graph Convolution for Recommendation

## Overview  
This project implements the **LightGCN (Light Graph Convolution Network)** model for collaborative filtering using the **MovieLens-1M dataset**. LightGCN improves over traditional graph convolution models like NGCF by simplifying the architecture—removing feature transformations and non-linear activations—while retaining the ability to capture high-order user-item relationships. The model’s performance is evaluated using metrics like **Recall** and **NDCG**.

## Features  
- **Dataset:** MovieLens-1M dataset containing 1 million user-movie interactions.  
- **Model:** LightGCN for efficient and scalable recommendation systems.  
- **Optimization:** Bayesian Personalized Ranking (BPR) loss.  
- **Evaluation:** Recall and NDCG as key evaluation metrics.  
- **Visualization:** Loss curves and performance plots for monitoring training progress.

## Dataset  
The **MovieLens-1M dataset** includes user-movie ratings and can be downloaded from [here](https://grouplens.org/datasets/movielens/1m/). It contains information about users, movies, and the ratings provided by users.

**Results and Visualizations:** View Recall and NDCG metrics, along with the training loss plots, to assess performance.

## Key Differences from NGCF  
LightGCN introduces several simplifications that improve both scalability and accuracy:  
- **No Feature Transformation or Non-linearity:** Unlike NGCF, LightGCN does not transform features through non-linear activation, simplifying the computation and improving model generalization.  
- **Layer Propagation:** LightGCN propagates embeddings across multiple layers to capture higher-order connectivity between users and items.  
- **Neighborhood Aggregation:** Embeddings from different layers are combined using a simplified aggregation mechanism, improving both training efficiency and recommendation accuracy.

## Optimization and Loss Function  
LightGCN optimizes embeddings using the **Bayesian Personalized Ranking (BPR) loss**, which focuses on maximizing the relative ranking between positive and negative user-item pairs. This loss function helps the model prioritize relevant recommendations while maintaining a scalable optimization process.

## Results  
- **Evaluation:** The model’s performance is assessed using Recall and NDCG across different layers and epochs.  
- **Loss Visualization:** The training process is monitored through loss-epoch plots to ensure convergence.  
- **Embedding Smoothness:** Analysis on how smooth embedding propagation improves recommendation performance and generalization.

## Future Improvements  
- Explore personalized aggregation weights across layers for dynamic neighbor contribution.  
- Introduce additional regularization techniques to further reduce overfitting.

## References  
- **Paper:** [LightGCN: Simplifying and Powering Graph Convolution Networks](https://arxiv.org/abs/2002.02126) by He et al.  
- **Dataset:** [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)
