# Dry Bean Dataset 多分类实验

这是《机器学习与项目实践》课程期末项目。项目使用教师提供的 Dry Bean Dataset 脏数据，完成从数据分析、数据清洗、特征工程、模型训练到结果展示的完整流程。任务目标是根据干豆的形状特征预测其所属类别。

项目已经完整运行，实验结果、图表和静态展示页面保存在 `outputs/` 中。打开 `index.html` 会自动跳转到 `outputs/report.html`，可作为 GitHub Pages 展示入口。

## 一、数据说明

数据位于 `DryBeanDataset/`，由训练集、验证集和测试集三部分组成。

| 文件 | 用途 | 行数 | 列数 |
| --- | --- | ---: | ---: |
| `Dry_Bean_Dataset_Dirty_train.csv` | 训练集 | 9527 | 17 |
| `Dry_Bean_Dataset_Dirty_val.csv` | 验证集 | 1347 | 17 |
| `Dry_Bean_Dataset_Dirty_test.csv` | 测试集 | 2737 | 17 |

每条样本包含 16 个数值型形状特征和 1 个类别标签 `Class`。类别共有 7 种：

```text
BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
```

原始数据中存在一定污染，主要包括：

- 标签格式不统一，例如大小写混乱、尾随空格、`D3RMAS0N`、`H0R0Z`、`S3K3R` 等字符替换；
- 数值列中存在缺失值、`?`、带单位字符串，例如 `0.8122 cm`；
- 验证集和测试集中存在负的 `Area`；
- 训练集中存在重复样本；
- 部分连续特征存在离群点。

清洗统计如下：

| 数据集 | 原始行数 | 清洗后行数 | 原始缺失单元格 | 数值转换后缺失单元格 | 非数值污染单元格 | 修正负面积 | 标签归一化数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 9527 | 9503 | 741 | 943 | 460 | 0 | 406 |
| val | 1347 | 1347 | 103 | 125 | 60 | 11 | 61 |
| test | 2737 | 2737 | 221 | 267 | 138 | 19 | 114 |

## 二、数据处理流程

数据处理代码主要在 `src/data_preprocess.py` 中，流程如下：

1. 读取训练集、验证集和测试集。
2. 统一字段名，去除字符串两端空格。
3. 将 `?`、空字符串、带单位的数值文本转换为可处理的数值或缺失值。
4. 修正标签污染，将异常写法统一为标准类别名。
5. 修正负面积，将明显由符号错误造成的负 `Area` 转为正值。
6. 删除训练集中的重复样本。
7. 构造 4 个形状相关组合特征：
   - `MajorMinorAxisRatio`
   - `AreaConvexAreaRatio`
   - `PerimeterSqrtAreaRatio`
   - `AxisLengthDiff`
8. 使用训练集的中位数填补缺失值。
9. 使用训练集拟合 Z-Score 标准化器，并应用到训练集、验证集和测试集。
10. 将类别标签编码为模型可训练的整数标签。

新增特征均来自原始几何特征之间的比例或差值，用于补充豆粒形状的长宽关系、面积紧致程度和边界复杂程度。

## 三、使用的算法

课程中已经学习过逻辑回归、KNN、支持向量机等分类算法，因此实验选用 5 种多分类算法：3 种课堂已学算法和 2 种课堂未讲算法。

| 算法 | 来源 | 类型 | 说明 |
| --- | --- | --- | --- |
| Logistic Regression | 课堂已学 | 线性模型 | 使用 softmax 思想完成多分类，可记录 log loss |
| KNN | 课堂已学 | 距离模型 | 根据邻近样本投票分类 |
| Linear SVM | 课堂已学 | 线性模型 | 使用 hinge loss 寻找分类间隔 |
| Random Forest | 课堂未讲 | 集成模型 | 多棵决策树投票，适合处理非线性关系 |
| Gaussian Naive Bayes | 课堂未讲 | 概率模型 | 基于贝叶斯公式和高斯分布假设 |

其中 Logistic Regression 和 Linear SVM 属于线性模型。KNN、Random Forest 和 Gaussian Naive Bayes 不属于线性模型。

## 四、实验结果

五种模型均在清洗后的数据上训练，并在测试集上评估。主要结果如下：

| 模型 | 训练集准确率 | 验证集准确率 | 测试集准确率 | 宏平均 F1 | 训练-测试差异 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9079 | 0.9139 | 0.9105 | 0.9231 | -0.0026 |
| KNN | 1.0000 | 0.9220 | 0.9295 | 0.9405 | 0.0705 |
| Linear SVM | 0.9126 | 0.9139 | 0.9141 | 0.9268 | -0.0016 |
| Random Forest | 1.0000 | 0.9206 | 0.9229 | 0.9327 | 0.0771 |
| Gaussian Naive Bayes | 0.9011 | 0.9154 | 0.9065 | 0.9154 | -0.0054 |

从测试集准确率看，KNN 表现最好，Random Forest 次之。KNN 和 Random Forest 的训练集准确率为 1.0000，说明这两个模型对训练数据拟合较强，需要结合过拟合分析一起判断。Logistic Regression 和 Linear SVM 的训练集、验证集、测试集准确率比较接近，泛化表现更稳定。Gaussian Naive Bayes 准确率最低，但模型简单、训练速度快，可作为概率模型基准。

## 五、鲁棒性实验

鲁棒性实验在训练集上加入噪声，再使用干净测试集评估模型性能下降情况。这样可以观察训练数据受到污染后，各模型是否仍能保持稳定。

使用的噪声类型包括：

| 噪声类型 | 加噪对象 | 含义 |
| --- | --- | --- |
| `gaussian` | 训练特征 | 给数值特征加入高斯扰动 |
| `missing` | 训练特征 | 随机制造缺失值 |
| `label_flip` | 训练标签 | 随机翻转一部分样本标签 |

实验结果显示，Random Forest 的平均准确率下降最小，整体鲁棒性最好。Linear SVM 和 Logistic Regression 对缺失噪声更敏感。Gaussian Naive Bayes 对标签翻转有一定波动，但整体下降幅度不大。

## 六、输出文件

运行后主要输出保存在 `outputs/` 中。

| 文件或目录 | 内容 |
| --- | --- |
| `outputs/report.html` | 静态展示页面 |
| `outputs/metrics_summary.csv` | 五种算法的准确率、F1、推理速度等指标 |
| `outputs/classification_reports.csv` | 每个类别的 precision、recall、F1 |
| `outputs/data_cleaning_summary.csv` | 数据清洗统计 |
| `outputs/class_distribution.csv` | 清洗后的类别分布 |
| `outputs/loss_history.csv` | Logistic Regression 和 Linear SVM 的 loss 记录 |
| `outputs/robustness_results.csv` | 鲁棒性实验结果 |
| `outputs/figures/` | 所有实验图表 |

图表包括：

- 数据污染统计图；
- 类别分布图；
- 特征相关性热力图；
- 五种算法准确率对比图；
- 推理速度对比图；
- 过拟合差异图；
- loss 曲线；
- 鲁棒性对比图；
- 每个模型独立的指标图、类别 F1 图、混淆矩阵和鲁棒性图。

`models/` 中保存了训练后的模型文件，便于后续复用或检查。

## 七、项目结构

```text
.
├── DryBeanDataset/                  # 原始脏数据
├── index.html                       # GitHub Pages 展示入口
├── main.py                          # 主程序入口
├── regenerate_report.py             # 根据已有结果重新生成图表和报告
├── requirements.txt                 # 依赖库
├── README.md                        # 项目说明
├── src/
│   ├── constants.py                 # 常量配置
│   ├── data_preprocess.py           # 数据读取、清洗、特征工程、标签编码和标准化
│   ├── models.py                    # 五种模型定义
│   ├── evaluation.py                # 模型训练、评估、loss 记录、推理速度和模型保存
│   ├── robustness.py                # 加噪训练鲁棒性实验
│   ├── visualization.py             # 图表绘制
│   └── reporting.py                 # 静态 HTML 报告生成
├── models/                          # 已训练模型
├── outputs/                         # 实验结果、图表和展示页面
├── 机器学习与项目实践期末总结论文.md
└── 机器学习与项目实践期末总结论文.docx
```

## 八、运行方式

建议使用 Python 3.10 或更高版本。

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

完整运行：

```bash
python main.py --data-dir DryBeanDataset --output-dir outputs --model-dir models
```

跳过鲁棒性实验：

```bash
python main.py --skip-robustness
```

只运行部分模型：

```bash
python main.py --models logistic_regression,knn,linear_svm
```

调整鲁棒性实验噪声：

```bash
python main.py --noise-types gaussian missing label_flip --noise-strengths 0.03 0.07 0.12
```

根据已有 `outputs/` 重新生成展示页面：

```bash
python regenerate_report.py --output-dir outputs
```

## 九、总结

本项目完成了结构化数据多分类任务的完整实验流程。数据处理部分解决了标签污染、缺失值、非数值字符串、负面积和重复样本等问题；模型部分比较了线性模型、距离模型、集成模型和概率模型；结果展示部分生成了可直接上传到 GitHub Pages 的静态网页。

综合实验结果，KNN 在测试集准确率上最高，Random Forest 在鲁棒性上更稳定，Logistic Regression 和 Linear SVM 推理速度快且泛化稳定，Gaussian Naive Bayes 可作为简单概率基准。不同算法各有优缺点，实际选择时需要同时考虑准确率、速度、鲁棒性和过拟合情况。
