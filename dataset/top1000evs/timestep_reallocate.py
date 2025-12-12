import pandas as pd
from datetime import datetime, time

input_filename = '20140824_processed.csv'

# 读取CSV文件
df = pd.read_csv(input_filename)

# 将timestamp列转换为datetime格式（根据你的时间格式调整）
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S')

# 提取时间部分
df['time'] = df['timestamp'].dt.time

# 定义筛选条件：从8:00开始的100个时间步
start_time = time(8, 0)  # 8:00
end_time = time(16, 20)  # 8:00 + 100*5分钟 = 16:20

# 筛选数据
filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy()

# 删除临时的time列
filtered_df = filtered_df.drop('time', axis=1)

# 验证结果
print(f"原始数据行数: {len(df)}")
print(f"筛选后数据行数: {len(filtered_df)}")
print(f"\n筛选后的时间范围:")
print(f"开始时间: {filtered_df['timestamp'].min()}")
print(f"结束时间: {filtered_df['timestamp'].max()}")
filtered_df['timestamp'] = filtered_df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')

# 保存筛选后的数据
filtered_df.to_csv(f'reallocated/{input_filename}', index=False)
print("\n筛选后的数据已保存")