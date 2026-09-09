import re

html_path = "papers/presentation/index.html"
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Update slide count in header
html = html.replace("1 / 10", "1 / 11")

mantis_html_slide = """
    <!-- SLIDE 8: Huawei Mantis -->
    <div class="slide">
      <div class="slide-header">
        <div class="slide-tag">Discriminative Foundation Model (Huawei Noah ICML 2026)</div>
        <div class="slide-title">Huawei Mantis: 填补时序大模型“分类盲区”的轻量级判别基座</div>
        <div class="slide-subtitle">论文：《Mantis: Lightweight Foundation Model for Time Series Classification》</div>
      </div>
      <div class="grid-4">
        <div class="card">
          <div class="card-title bg-tag">📖 背景与痛点</div>
          <ul class="feature-list">
            <li>以往时序大模型（Chronos、TimesFM）全在卷预测（Forecasting）。</li>
            <li>手表的核心商业需求大量是<strong>分类与判别</strong>（动作识别 HAR、防误计步、心律失常/房颤筛查、跌倒预警）。</li>
          </ul>
        </div>
        <div class="card">
          <div class="card-title tech-tag">⚙️ 核心技术方案</div>
          <ul class="feature-list">
            <li><strong>Token Generator Unit (TGU)</strong>：自适应局部多尺度卷积，精准捕捉时序几何形态。</li>
            <li><strong>极轻量 ViT-1D (~8M 参数)</strong>：体积小、推理极速，专为端侧低功耗处理器设计。</li>
            <li><strong>自监督对比学习</strong>：大规模合成与真实时序预训练，专攻判别特征。</li>
          </ul>
        </div>
        <div class="card">
          <div class="card-title data-tag">📊 数据与实测战绩 (实验 6)</div>
          <ul class="feature-list">
            <li>在智能手表真实 3 轴运动加速度数据（1069 样本）上实测：</li>
            <li>CPU 单样本特征提取仅 <strong>19.80 ms</strong>！</li>
            <li>线性探测在走路、跑步、骑行三分类准确率高达 <strong>99.7%</strong>（显著超越 LIMU-BERT 的 92.1%）！</li>
          </ul>
        </div>
        <div class="card">
          <div class="card-title comm-tag">💡 商业落地启发</div>
          <ul class="feature-list">
            <li><strong>终结手部假动作误计步</strong>：输入 3 轴 ACC 提取高判别力表征，彻底过滤刷牙、打字假动作。</li>
            <li><strong>端侧心律失常极速初筛</strong>：INT8 量化部署在手表 MCU 上，毫秒级本地报警。</li>
          </ul>
        </div>
      </div>
    </div>
"""

# Insert before SLIDE 8: 跨模态解耦实证
html = html.replace("<!-- SLIDE 8: 跨模态解耦实证 -->", mantis_html_slide + "\n    <!-- SLIDE 9: 跨模态解耦实证 -->")
html = html.replace("<!-- SLIDE 9: 商业技术横向选型矩阵 -->", "<!-- SLIDE 10: 商业技术横向选型矩阵 -->")
html = html.replace("<!-- SLIDE 10: 自建 Watch-LSM 落地路线图 -->", "<!-- SLIDE 11: 自建 Watch-LSM 落地路线图 -->")

# Add table row in decision matrix
old_row = "<td><strong>DAPO LIMU-BERT</strong></td>\n          <td>三轴 ACC + 陀螺仪</td>\n          <td><strong>极佳 (微型BERT)</strong></td>\n          <td>解耦佩戴个体差异，运动特征分明</td>\n          <td>零误触权威计步、帕金森步态筛查与老年人防跌倒监测</td>\n        </tr>"
new_row = old_row + "\n        <tr>\n          <td><strong>Huawei Mantis</strong></td>\n          <td>单变量/多变量(3轴ACC)</td>\n          <td><strong>极佳 (~8M 参数, 19ms)</strong></td>\n          <td>专注分类判别，Token生成器</td>\n          <td>零误触权威计步、动作状态高精分类(实测99.7%)、端侧心律失常初筛</td>\n        </tr>"
html = html.replace(old_row, new_row)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Updated presentation/index.html with Mantis slide!")
