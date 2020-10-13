# Prophet预测库介绍

Prophet，或称“Facebook Prophet”，是一个由Facebook开发的用于单变量时间序列预测的开源库。

Prophet实现的是一个*可加的时间序列预测模型*，支持**趋势**、**季节性周期变化**及**节假日效应**。

“该模型所实现的是一个基于可加模型的时间序列数据预测过程，拟合了年度、周度、日度的季节性周期变化及节假日效应的非线性趋势。”
— Package ‘prophet’, 2019.

Prophet的设计初衷就是简单易用、完全自动，因此适合在公司内部场景中使用，例如预测销量、产能等。 

这里有一篇不错的概览，介绍了Prophet及它的功能：
[Prophet: forecasting at scale, 2017](https://research.fb.com/blog/2017/02/prophet-forecasting-at-scale/)