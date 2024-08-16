# Recommender Service Research

Testing different ideas related to the Weaviate Recommender Service.

- `xWines_PCA_TRL.ipynb`: This notebook tests PCA (Principal Component Analysis) for TRL (Table Representation Learning). Most Weaviate users are probably familiar with using vector embedding models to represent their data objects. This typically works by passing a particular property such as `content` into the model. This works great for most applications of Vector and Hybrid Search, however, Recommendation Systems strive to integrate more structured metadata into the representation of objects. For example, we want to integrate a numeric valued `price` or a categorical valued `brand` into the object's vector. This notebook tests how well PCA can combine structured and unstructured features to represent objects with the `xWines` dataset!

Sign up for the Recommender Service beta testing [here](https://weaviate.io/workbench/recommender)!
