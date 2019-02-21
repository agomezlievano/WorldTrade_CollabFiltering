# Applied Neural Networks-based Collaborative Filtering to World Trade
At the Center for International Development at Harvard, where I do research as a postdoc, we have a particular view on the process of economic development. 

The short story is that Ricardo Hausmann (the director of CID) and Baily Klinger (former postdoc at CID, now Director at LenddoEFL) wrote a seminal and influential working paper in 2006, titled "Structural Transformation and Patterns of Comparative Advantage in the Product Space", where they developed the idea that economic development is a process of accumulation of productive capabilities (which are unobservable and difficult to trade). This is in contrast to more traditional theories, which would characterize development as an increase in either capital labor, or more importantly, technology. In these theories, "technology" was just a term that could only be inferred "after the fact". That is, by the residual (unexplained) deviations of the productivity of a country after accounting for capital and labor. Instead, Hausmann and Klinger assume that the differences in productivity across countries was not a matter of having more or less of these three aggregated quantities, but actually, of having a variety of qualitatively different capabilities. 

In their seminal paper, they argue the capabilities that a country has are revealed in the fact that they produce different varieties of products. Hence, by looking at the products countries export, one can get a sense of the distribution of capabilities across countries and products. Specifically they focus the relationships between products. Hence, by looking at many countries, they propose a method to infer how products share capabilities. Creating such matrix of "technological proximity" between products is very useful and informative, because one can then predict which products a country is more or less likely to start exporting in the future. They say, 
"[O]ur main idea is that the similarity of
capabilities (or the distance between trees) is heterogeneous, but is related to the
likelihood that countries have revealed comparative advantage in both goods. To develop
this measure we use product-level data of exports, which is appropriate as exports
represent products in which a country has a comparative advantage and must pass a rather
strict market test compared to production for the domestic market. For a country to have
revealed comparative advantage in a good it must have the right endowments and
capabilities to produce that good and export it successfully. If two goods need the same
capabilities, this should show up in a higher probability of a country having comparative
advantage in both. We calculate this probability across a large sample of countries."

This insight was then developed further with the help of Cesar Hidalgo (another former postdoc of CID, now professor at the Media Lab at MIT), in a Science publication "The product space conditions the development of nations". 

# This repo
The above papers opened an fruitful avenue of research that has become a field in itself: the field of Economic Complexity. Some people have realized that predicting which products a country would export in the future using the insights from Economic Complexity was akin to the process of predicting which movie a person would be likely to watch in the future using insights from Machine Learning. The latter exercise became famous by the Netflix Price. The winning solution to the competition became also a field in itself: the field of Collaborative Filtering. Hence, one can say that Economic Complexity is Collaborative Filtering applied to economics. It is, of course, more than than, since we also have mathematical models about why and exactly how some of these methods work. 

What I want to do here is to apply one of the latest, state-of-the-art, collaborative filtering techniques: neural networks. What is my goal? One the one hand, I want to quantify how easy/hard it is to train such a network given the fact that world trade data is a small data set compared to the typical datasets in other applications of neural networks (instead of having tens of millions of users, and tens of thousands of movies, we only have around a hundred countries, and at most a few thousands products). On the other hand, if neural networks are indeed appropriate, I expect them to be very predictive. Which we will have a benchmark for how easy it is to predict economic diversification. Hence, we will have learned something about the world.

### A note
The specific type of collaborative filtering technique implicit in Hausmann and Klinger's methodology is just one among many techniques that are part of a bigger family of methods: Low-rank models, or Matrix Factorization methods. I may post some of them in other repos. Here we will just focus on neural networks.

# What we will use
We will use FASTAI libraries, which are very convenient and easy to use. 

For the data, we will download the freely available data used for the Atlas of Economic Complexity. 
Details of the data: http://atlas.cid.harvard.edu/downloads
Where to download the data: https://intl-atlas-downloads.s3.amazonaws.com/index.html

