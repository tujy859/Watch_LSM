import re

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

# Add mantis paper and analysis to directory tree
readme = readme.replace(
    "│   │   ├── CMU_MOMENT_ANALYSIS.md           # CMU 开源时间序列通用基础模型\n│   │   └── LIMU_BERT_ANALYSIS.md            # IMU/加速度计空间表征基础模型",
    "│   │   ├── CMU_MOMENT_ANALYSIS.md           # CMU 开源时间序列通用基础模型\n│   │   ├── LIMU_BERT_ANALYSIS.md            # IMU/加速度计空间表征基础模型\n│   │   └── MANTIS_NOAH_ANALYSIS.md          # 华为诺亚轻量级时序分类基础模型"
)

readme = readme.replace(
    "│   ├── limu_bert.pdf                    # LIMU-BERT 论文原件",
    "│   ├── limu_bert.pdf                    # LIMU-BERT 论文原件\n│   ├── mantis_icml2026.pdf              # Mantis 论文原件 (ICML 2026)"
)

readme = readme.replace(
    "│   ├── moment/                          # CMU MOMENT 官方代码\n│   └── limu_bert/                       # LIMU-BERT 官方代码与预训练权重",
    "│   ├── moment/                          # CMU MOMENT 官方代码\n│   ├── limu_bert/                       # LIMU-BERT 官方代码与预训练权重\n│   └── mantis/                          # 华为诺亚 Mantis 官方代码 (ICML 2026)"
)

readme = readme.replace(
    "│   └── exp5_cross_modal_denoising.py    # 实验5: 跨模态自适应消除运动伪影实证",
    "│   ├── exp5_cross_modal_denoising.py    # 实验5: 跨模态自适应消除运动伪影实证\n│   └── exp6_mantis_eval.py              # 实验6: Mantis 时序分类基础模型实测 (准确率99.7%)"
)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("Updated root README.md with Mantis!")
