import sqlite3
import pandas as pd

# 数据库路径
db_path = r"D:\Desktop\Data\AP_1p5.db"

# 要导出的表名
table_name = "exp_10"

# 导出目标路径
csv_path = rf"D:\Desktop\Data\{table_name}.csv"

# 连接数据库
conn = sqlite3.connect(db_path)

# 读取表数据到 DataFrame
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# 导出为 CSV
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# 关闭连接
conn.close()

print(f"✅ 表 {table_name} 已成功导出为 CSV 文件：\n{csv_path}")
