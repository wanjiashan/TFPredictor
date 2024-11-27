![image](https://github.com/user-attachments/assets/4b0d9a4f-c7fa-4dfa-9540-cdbf19c845ac)## TFPredictor: A Multi-scale Selective State Space Model for Traffic Flow Prediction

### Abstract

> *Most advanced traffic-forecasting methods rely on fixed temporal resolutions to handle both short-term and long-term variations in traffic flow, concentrating on capturing fixed-scale relationships among nodes in spatiotemporal graph networks. However, this fixed approach can lead to incomplete scale modeling and limit the model's adaptability to complex traffic dynamics. To overcome these challenges, we introduce a novel multi-scale Selective State Space Model (SSSM) named TFPredictor for traffic forecasting. TFPredictor features four key components: (a) multi-scale decomposition of input sequences via Fast Fourier Transform (FFT) to capture diverse periodic information in the frequency domain; (b) adaptive spatiotemporal modeling through the Kalman Filtering Graph Neural Network (KFGNN); (c) dynamic system analysis using TFPredictor to simulate state evolution accurately over time in spatiotemporal graph networks; and (d) multi-scale fusion via the Scale-weighted Kolmogorov-Arnold Network (ScaKAN) to assess and capture the significance of each scale and inter-scale correlations. Extensive empirical evaluations on six benchmark traffic-forecasting datasets reveal that TFPredictor achieves significant improvements in predictive performance and computational efficiency. Additionally, by visualizing the learned spatial representations and predictions across different scales, we enhance the model's interpretability. Code is available at https://github.com/wanjiashan/TFPredictor**

Index Terms—Selective state space model, traffic flow, multiscale, Kolmogorov-Arnold network

![landslide](imgs/1.png)
PerformanceComparisonofDifferent.
(1) To address the limitations of existing methods that use
fixed temporal resolution, we propose a multi-scale Selective
State Space Model named TFPredictor. TFPredictor includes
multi-scale decoupling and fusion modules, a state graph structure generator, and the Graph-Mamba block, comprehensively
capturing multi-scale temporal patterns and enhancing the
model’s multi-scale modeling capability.
(2) We employ an SSSM to handle the traffic network,
treating it as a dynamic system to simulate state evolution
deeply along the temporal dimension, thereby enhancing the
understanding of the system-level dynamics of the traffic
network.
(3) We design a model with linear time o (n) complexity,
which not only improves prediction accuracy but also effectively shortens inference time, reduces computational costs,
and enhances the model’s practicality and scalability.
(4) Extensive empirical studies were conducted on six realworld public traffic datasets. The results demonstrate that
our model outperforms state-of-the-art models in terms of
performance, computational efficiency, and interpretability.

![landslide](imgs/2.png)   ![landslide](imgs/2-1.png)

Loss trend comparison between TFPredictor and other benchmark models at different epoch counts
provides a detailed comparison of loss trends with increasing epochs under different architectures. From the analysis, it is evident that the Mamba model has significant advantages over Transformer-based models. To quantify the performance advantages of the ScaKAN component in the model further, we present the loss trends for MLP and ScaKAN in our model in Fig. ~\ref{fig5}. ScaKAN outperforms MLP, with the 4-layer KAN structure performing the best. Its loss decreases rapidly to the minimum level after relatively few epochs and remains stable. This indicates that ScaKAN not only provides more accurate prediction results but also maintains consistency under different traffic conditions, which is crucial for real-world traffic prediction scenarios.


## 2. Training Procedure
### (1) Datasets
We evaluate the performance of TFPredictor on six major
traffic flow prediction benchmark datasets: METR-LA, PEMSBAY, PEMS03, PEMS04, PEMS07, and PEMS08. The time
interval for each dataset is 5 minutes, resulting in 12 time
frames per hour. These datasets are widely used in traffic
prediction research and cover various traffic conditions and
challenges. For more information about these datasets, please
refer to Table I.


<b>PEMS04</b>: PEMS04: PEMS04 dataset is among the most popular benchmark in ST Data Mining & Traffic Forecasting. You can find the source data at here


<b>Traffic</b>: Describes the occupancy rates (between 0 and 1) measured by 862 sensors on San Francisco, California,
freeways. The data were sampled every hour from 2015 to 2016. Following [38], we convert the data to reffect hourly
consumption.


<b>Weather</b>: Provided by the Max Planck Institute for
Biogeochemistry in Germany and records meteorological observation
 data for the city of Jena, Germany. The data includes
multiple meteorological indicators such as temperature, precipitation,
 wind speed, etc., measured at different locations in
Jena by ground stations. These data have been continuously
collected every 10 minutes since 2003.

<b>Landslide</b>: Describes the landslide geological disaster
monitoring data for Yangchan Village in Huangshan City,
China. The data spans from April 2022 to March 2023,
recorded on a daily basis. It belongs to a self-built benchmark
dataset.

### (2) Network Parameter Tuning
Given that the hyperparameters that have a signiffcant
impact on DAMixer network include the convolution kernel
size, activation function, loss function, optimizer, number of
training epochs, learning rate, number of nodes in the hidden
layer, and number of hidden layers, this paper uses crossvalidation
 to ffnd the optimal parameter values that maximize
the model’s generalization performance, as shown in Table I.
![landslide](imgs/t1.png)



### (3) Benchmarks Models
We train benchmark models such as <a href="https://github.com/LincanLi98/STG-Mamba" target="_blank">STG-Mamba</a>, <a href="https://github.com/XDZhelheim/STAEformer" target="_blank">STAEformer</a>, <a href="https://github.com/BUAABIGSCity/PDFormer" target="_blank">PDFormer</a>, <a href="https://github.com/chnsh/DCRNN_PyTorch" target="_blank">DCRNN</a>, <a href="https://github.com/hazdzz/STGCN" target="_blank">STGCN</a>, <a href="https://github.com/nnzhan/MTGNN" target="_blank">MTGNN</a>, <a href="https://github.com/zhengchuanpan/GMAN" target="_blank">GMAN</a>, ,<a href="https://github.com/LiuAoyu1998/STIDGCN" target="_blank">STIDGCN</a>, <a href="https://github.com/wengwenchao123/DDGCRN" target="_blank">DDGCRN</a> ,<a href="https://github.com/LeiBAI/AGCRN" target="_blank">AGCRN</a>, and <a href="https://github.com/nnzhan/Graph-WaveNet" target="_blank">Graph-WaveNet</a> based on the same sample set.

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
