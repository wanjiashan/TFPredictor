## TFPredictor: A Multi-scale Selective State Space Model for Traffic Flow Prediction

### Abstract

> *Most advanced traffic-forecasting methods rely on fixed temporal resolutions to handle both short-term and long-term variations in traffic flow, concentrating on capturing fixed-scale relationships among nodes in spatiotemporal graph networks. However, this fixed approach can lead to incomplete scale modeling and limit the model's adaptability to complex traffic dynamics. To overcome these challenges, we introduce a novel multi-scale Selective State Space Model (SSSM) named TFPredictor for traffic forecasting. TFPredictor features four key components: (a) multi-scale decomposition of input sequences via Fast Fourier Transform (FFT) to capture diverse periodic information in the frequency domain; (b) adaptive spatiotemporal modeling through the Kalman Filtering Graph Neural Network (KFGNN); (c) dynamic system analysis using TFPredictor to simulate state evolution accurately over time in spatiotemporal graph networks; and (d) multi-scale fusion via the Scale-weighted Kolmogorov-Arnold Network (ScaKAN) to assess and capture the significance of each scale and inter-scale correlations. Extensive empirical evaluations on six benchmark traffic-forecasting datasets reveal that TFPredictor achieves significant improvements in predictive performance and computational efficiency. Additionally, by visualizing the learned spatial representations and predictions across different scales, we enhance the model's interpretability. Code is available at https://github.com/wanjiashan/TFPredictor**

Performance comparison of six selected methods in four types: GNNs, LLMs, Transformers, and Mixers, using the mean squared error as the metric.

![p1](./imgs/p1.png)

Potential interactions among variables in MTS prediction are critical. Most studies have used a pre-set static correlation (s0). However, in reality, the graph structure changes over time (s1 and s2), and these changes differ based on the scale of observation (s3). Therefore, it is crucial to consider the dynamic nature and scale effects of these inter-variable interactions when predicting MTS.

![p2](./imgs/p2.png)

Architecture of the MSPredictor. (a) Multi-Scale Decoupling Module (MDM): This module uses FFT to decompose the original sequence into different scales, capturing various periodic information in the frequency domain of the input sequence. (b) Evolving Graph Structure Learner (EGSL): This learner is responsible for learning and updating the multi-scale temporal graph structure to adapt to the dynamic changes in time series data. (c) Multi-Scale Spatiotemporal Module (MSTM): This module contains *k* GNNs and TCNs, designed to capture the dynamic and complex relationships between variables at specific scales. (d) Multi-Scale Fusion Module (MFM): This module integrates features and temporal pattern information from the MSTM. It effectively combines information from different scales through an *L*-layer KAN, significantly enhancing the accuracy and stability of predictions.

![p3](./imgs/p3.png)

The architecture of multi-scale fusion module, which primarily en- compasses three key operations: (a) Concatenation is employed to amalgamate representations from various scales into a singular, unified vector, ensuring a comprehensive aggregation of information. (b) Pooling is applied to reduce the dimensionality of the combined feature vector, highlighting the most crucial features. (c) Fully connected layer composed of *L*-layer KAN.

![p4](./imgs/p4.png)






# TFPredictor
首先需要加压数据集
例如运行PEMSBY
调参prepare.py里面的if speed_sequences.shape[2] > 325:
        speed_sequences = speed_sequences[:, :, :325]和train_STGmamba里面的mamba_features=325这个参数，这个对应具体数据集的特征，比如PESMSBY是325，
  然后那个metr-la是207，就需要调一下，运行代码 
```bash
#PESMSBY
  python main.py -dataset=PEMSBY -model=STGmamba -mamba_features=325 
```
```bash
#metr-la
  python main.py -dataset=metr-la -model=STGmamba -mamba_features=207
```
```bash
#PEMS04
  python main.py -dataset=PREMS04 -model=STGmamba -mamba_features=307
```
