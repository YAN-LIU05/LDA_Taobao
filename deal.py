import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
font = FontProperties(fname=font_path)
# 读取xlsx文件
df = pd.read_excel('/root/autodl-tmp/analysis_report_3.xlsx')

# 假设'keyword'列包含了形如['计划', '控卡', '全餐', '脂', '辣酱']的数据
keywords = df['keywords'].dropna().apply(eval).explode()  # 去除NaN值，使用eval转换字符串为列表，再展开

# 统计词频
word_counts = Counter(keywords)

# 获取出现频率最高的5个词
top_10_words = word_counts.most_common(10)

# 提取词和频率
words, counts = zip(*top_10_words)

# 可视化：绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.xlabel('关键词', fontsize=14,fontproperties=font)
plt.ylabel('频率', fontsize=14,fontproperties=font)
plt.title('出现频率最高的10个词', fontsize=16,fontproperties=font)
plt.xticks(rotation=45, ha='right',fontproperties=font)  # 旋转x轴标签，以便显示长词
plt.tight_layout()  # 调整布局以避免标签重叠
plt.savefig('report_3.png')
