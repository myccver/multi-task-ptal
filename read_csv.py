import pandas as pd

# 读取CSV文件
data = pd.read_csv('/home/yunchuan/HR-Pro/dataset/ActivityNet1.3/point_labels/point_gaussian.csv')
unique_classes = data['class'].unique()

# 打印不重复的类名
print(unique_classes)
# 显示数据的前几行
print(data.head())
