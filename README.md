<h1 align="center"> Turkish Review Comments Analysis with Deep Learning (RNN-GRU)</h1>



<img src="https://katalystcommunications.com/wp-content/uploads/2016/02/brand-sentiment2-880x470.jpg">


<h4 align="center">  HepsiBurada is most visited and largest E-commerce website in Turkey.An e-commerce site where you can find all kinds of products easily. Therefore, we can make comments for each product. In addition, we can rate the product out of 5 with our comment. Our data set determined the number 2.5 as the threshold value and defined the comments as negative and positive. If the score is below 2.5, the comment is negative and if it is over, it is positive. At the same time, a sensitivity analysis algorithm can be written to work with higher performance to reduce manpower on these data. This is what I did today to learn the GRU algorithm.</h4>

### Example product on Hepsiburada

![Hepsiburada](/img/product.png)

### Example comment on Hepsiburada

![Hepsiburada](/img/comment.png)




## How to Work GRU/LSTM?

LSTM ’s and GRU’s were created as the solution to short-term memory. They have internal mechanisms called gates that can regulate the flow of information.

![GRU](https://miro.medium.com/max/576/1*AQ52bwW55GsJt6HTxPDuMA.gif)

![Architecture](https://cdn-images-1.medium.com/max/800/1*9z1Jrl8K99TorEQfsOTjpA.png)



## Made with using Tools:

* Tensorflow / Keras
* Numpy
* Pandas


## Model Architecture:

* Many to one RNN
* Gated Recurrent Unit
* Based on LSTM model













