import sys

# 过滤掉包含某些关键词的行
filtered_keywords = [
    "MoE Debug",
    "Gate probs",
    "Expert selection",
    "Expert usage",
    "Balance loss",
    "Importance loss",
    "Final aux loss"
]

for line in sys.stdin:
    # 检查这行是否包含任何要过滤的关键词
    if not any(keyword in line for keyword in filtered_keywords):
        sys.stdout.write(line) 