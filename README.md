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
