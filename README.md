# MIT-BIH 五类心拍分类

本项目实现课堂的 N、A、V、L、R 五类心拍分类。读取本项目 `data` 中的原始 WFDB 文件，使用专家标注的 R 峰位置截取心拍。目标是心拍类别识别，不包含独立的 R 峰检测器。

## 当前默认评价方式：课堂复现

默认 `--evaluation-protocol classroom`：先随机打乱心拍行，再随机排列索引，按课堂代码流程分出 **80% 训练、20% 测试**。不按患者分组、不做类别分层，也没有单独的验证集。每轮计算测试集 Accuracy，用它保存最佳模型及触发早停，Macro-F1 作为辅助指标。

同一患者的不同心拍可能出现在两侧，且测试集参与选模。因此结果应注明“随机心拍 8:2 划分，按测试集 Accuracy 选模”，不能解释为新患者上的独立测试表现。程序保存实际心拍行索引和行顺序指纹，测试入口恢复同一组心拍，不重新抽样。

固定种子为 42，自己的实验可重复；课堂未提供原始划分索引，不能保证每个测试心拍完全相同。当前方案 B 网络、44 条记录范围及类别权重仍与课堂不同，本次只对齐评价方式，不保证得到 99.28%。旧的患者独立评价保留为 `--evaluation-protocol patient`。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `data.py` | 原始文件读取、db5 小波降噪、300 点心拍截取、逐心拍标准化、缓存和患者划分 |
| `models.py` | 默认方案 B：两层 128 单元 LSTM，分类器 `128→128→64→5`，输入 `[批大小, 1, 300]` |
| `train.py` | 按指定协议训练、选模和早停，保存模型、指标及图表 |
| `evaluate.py` | 恢复模型保存的心拍或患者划分并测试 |

## 环境和数据

使用指定环境 `F:\实验文件夹\lwnet-master0\.venv`。需要 torch、wfdb、PyWavelets、numpy、scikit-learn、matplotlib；这些包在编写代码时均已确认存在。程序不依赖 seaborn、TensorFlow 或 Jupyter。Notebook 的辅助包不影响这些脚本运行。

默认数据目录根据脚本所在位置定位为 `G:\深度学习实验心拍\data`。同一记录的 `.dat`、`.hea`、`.atr` 应位于该目录。不读取 `.zip`、`.xws` 或历史标注备份。

默认排除含起搏心拍的 102、104、107、217，共使用 44 条记录，MLII 导联，360 Hz。102、104 也缺少 MLII。这个记录选择与课堂 notebook 的 43 条记录清单不同，报告中需要说明。

## 1. 先运行小规模检查

在 PowerShell 运行：

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' --smoke-test
```

每条记录最多保留 64 个心拍，只训练 1 轮，验证读取、GPU、保存与绘图流程。子集尽量保持本条记录的类别覆盖；它不是正式实验数据。程序会明确标记 `development_subset=true`。课堂模式会评价开发子集中的测试心拍。

## 2. 仅准备数据

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' --prepare-only
```

保留全部可用心拍，输出样本统计、指定协议的数据划分和预处理图。缓存位于项目 `outputs\cache`，首次处理后，后续相同配置的实验会复用缓存。缓存依据原始文件路径、大小、修改时间及预处理配置失效；它不是文件内容哈希校验。

预处理过程为：读取完整 MLII 数字信号 → db5 小波降噪 → 跳过开头 10 条和末尾 5 条标注 → 过滤至 NAVLR → 按 `[R-99:R+201]` 截取并检查边界 → 每个心拍独立进行均值方差标准化。该小波方案沿用课堂方法，不宣称已经去除全部基线漂移或全部噪声。最终五类样本数量以实际输出为准。

模型默认接收与课堂一致的 `[批大小, 1, 300]`：一个时间步、300 个波形特征。预处理缓存继续用 `[样本数, 300, 1]` 保存，数据加载时转置，波形数值和顺序不变，因此可以复用已有缓存。运行配置会分别记录缓存形状与模型输入形状。

输入布局和网络结构分别写入运行配置与 checkpoint。当前默认方案 B 使用两层 128 单元 LSTM，分类器为 `128→128→64→5`；原有单层模型和逐点输入模型继续按其保存的配置读取。新结构需要从头训练，不能直接套用旧结构的权重。

## 3. 正式训练与调优

当前默认网络为讨论后确定的方案 B：

```text
[B, 1, 300]
    → LSTM(input_size=300, hidden_size=128, num_layers=2, dropout=0.2)
    → 最后一层隐藏状态 [B, 128]
    → Linear(128, 128) → ReLU → Dropout(0.5)
    → Linear(128, 64)  → ReLU → Dropout(0.5)
    → Linear(64, 5)
```

LSTM 参数量 352,256，分类器参数量 25,093，总计 **377,349 个可训练参数**。输入仍只有一个时间步。结构扩展本身不保证提高准确率。

方案 B 的完整训练命令（保持最近实验的网络及超参数，改用课堂评价）：

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' --evaluation-protocol classroom --test-fraction 0.2 --model lstm --input-size 300 --pooling last --hidden-size 128 --num-layers 2 --recurrent-dropout 0.2 --classifier-hidden-sizes 128 64 --classifier-dropout 0.5 --loss weighted --weight-power 0.5 --lr 0.001 --weight-decay 0.0001 --batch-size 64 --epochs 30 --patience 8 --clip-grad 1.0 --seed 42 --device cuda
```

以下简短示例使用命令行默认训练参数（例如普通 CE、patience=5），不等同于上面的完整对比命令。

普通 RNN 对照（使用与 B 相同的隐藏规模和分类器）：

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' --model rnn
```

方案 B 的 LSTM：

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' --model lstm
```

LSTM 加权交叉熵对比：

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' --model lstm --loss weighted
```

调整学习率的简短示例；实际对比时建议每次只改变一个因素：

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' --model lstm --loss weighted --lr 0.0003 --hidden-size 128 --num-layers 2 --recurrent-dropout 0.2
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model` | lstm | rnn / lstm |
| `--evaluation-protocol` | classroom | classroom：心拍 8:2、测试 Accuracy 选模；patient：患者约 70:15:15、验证 Macro-F1 选模 |
| `--test-fraction` | 课堂 0.2；患者 0.15 | 测试集目标比例 |
| `--val-fraction` | 课堂 0；患者 0.15 | 课堂模式不另设验证集 |
| `--input-size` | 300 | 300 为课堂 `[B,1,300]`；1 为此前逐点输入 `[B,300,1]` |
| `--hidden-size` | 128 | 隐藏单元数 |
| `--num-layers` | 2 | 循环层数 |
| `--recurrent-dropout` | 多层 0.2，单层 0 | 层间 Dropout，单层未指定时自动设为 0 |
| `--classifier-hidden-sizes` | 128 64 | 分类器隐藏层；设为 64 可恢复旧的单隐藏层分类器 |
| `--classifier-dropout` | 0.5 | 每个分类器隐藏层后的 Dropout |
| `--pooling` | last | 课堂单步输入使用 last；mean/max/meanmax 仅用于 `--input-size 1` |
| `--loss` | ce | 普通交叉熵；weighted 为加权交叉熵 |
| `--weight-power` | 0.5 | 权重为训练集类别频数倒数的 0.5 次幂，再归一化至均值为 1 |
| `--lr` | 0.001 | AdamW 学习率 |
| `--scheduler` | none | none 为固定学习率；cosine 为按轮更新的余弦退火 |
| `--min-lr` | 0.00001 | 余弦退火最低学习率，须介于 0 与初始学习率之间 |
| `--weight-decay` | 0.0001 | 权重衰减 |
| `--batch-size` | 64 | 批大小 |
| `--epochs` | 30 | 最大轮数 |
| `--patience` | 5 | 对应协议的选模指标连续多少轮未提升后早停 |
| `--clip-grad` | 1 | 梯度范数上限 |
| `--seed` | 42 | 随机种子 |
| `--device` | auto | 自动用 GPU；也可指定 cpu / cuda |
| `--no-denoise` | 不启用 | 添加该开关可做去噪消融 |

模型输出未经 softmax 的五类 logits，直接用于交叉熵。评价时才计算 softmax 概率。训练与推断使用同一模型，参数更新前置于正确的优化器流程。

### 余弦退火学习率实验

在目前最佳配置上，只增加学习率调度：初始学习率 0.0003，`T_max=50`，最低学习率 0.00001。每轮训练结束后调用一次调度器，不使用热重启；数据划分、模型结构、损失和早停规则保持不变。

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' `
  --evaluation-protocol classroom --test-fraction 0.2 `
  --model lstm --input-size 300 --pooling last `
  --hidden-size 128 --num-layers 2 --recurrent-dropout 0.2 `
  --classifier-hidden-sizes 128 64 --classifier-dropout 0.5 `
  --loss weighted --weight-power 0.5 `
  --lr 0.0003 --scheduler cosine --min-lr 0.00001 `
  --weight-decay 0.0001 --batch-size 64 `
  --epochs 50 --patience 8 --clip-grad 1.0 --seed 42 --device cuda
```

第 1 轮使用初始学习率，第 50 次轮末调度后达到最低学习率。早停可能提前结束，因此不保证运行完整退火周期；`--epochs` 同时决定最大训练轮数和退火周期长度。保持 `--patience 8` 是为了先只比较调度方式。训练输出及 `history.csv` 的 `lr` 列记录每轮实际使用的学习率，`learning_curves.png` 包含学习率曲线，checkpoint 和总结保存调度配置及最佳轮次的学习率。重新评估模型时无需传入调度器参数。

省略 `--scheduler cosine` 或使用 `--scheduler none` 时仍为固定学习率，可继续运行原有对照命令。

### 末状态与全序列特征的对比

如需保留此前的逐点时序实验，必须显式添加 `--input-size 1`，并明确原来的分类器隐藏层 `--classifier-hidden-sizes 64`。在该模式下，可用 `--pooling meanmax` 将 300 个时间步输出特征的均值与最大值拼接；隐藏维度为 64 时，分类器输入为 128 维。该方式不是课堂的单步输入模型。历史 checkpoint 没有 `input_size` 字段时自动按原来的 `input_size=1` 读取；没有 `classifier_hidden_sizes` 时使用旧的 64 单元分类器隐藏层。新的 checkpoint 会明确保存这些结构参数。

```powershell
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\train.py' --model lstm --input-size 1 --hidden-size 64 --num-layers 1 --pooling meanmax --classifier-hidden-sizes 64 --classifier-dropout 0.3 --loss weighted --weight-power 0.5 --lr 0.001 --weight-decay 0.0001 --batch-size 64 --epochs 30 --patience 8 --clip-grad 1.0 --seed 42 --device cuda
```

后续若改变学习率，先单独对比 `--lr 0.0003`。当前方案 B 的分类器 Dropout 已为 0.5，调节它时需要明确记录新旧值。比较时保持同一评价协议和数据划分；课堂模式如实报告测试集参与选模，患者模式只根据验证集调参。

## 数据划分与最佳模型选择

课堂模式默认随机心拍训练/测试为 8:2，同一心拍行不能重复，但同一患者可同时出现在两个集合。`split_manifest.json` 和 checkpoint 都保存具体索引与行顺序指纹；后续测试严格恢复这些行。按测试集 Accuracy 选模、早停，没有单独验证集。

如需保留此前的患者独立评价，在完整命令中替换评价参数为 `--evaluation-protocol patient --val-fraction 0.15 --test-fraction 0.15`。该模式仍将 201、202 归为同一患者，患者互不重叠，按验证集 Macro-F1 选模。固定种子下，使用 8192 个候选患者划分中的类别计数来近似分层，不使用模型成绩挑划分。

两种协议各自缓存划分，结果和 checkpoint 明确保存 `evaluation_protocol`、`selection_metric`、`test_used_for_model_selection`。历史 checkpoint 无协议字段时按原患者划分读取。改变评价方式需要重新训练，不能把旧模型直接换到新的测试划分。

类别权重仅从训练集计算，逐心拍标准化不使用跨集合统计量。训练指标来自训练批次（包含 Dropout），评价指标来自 eval 模式。

## 4. 最终测试

课堂模式训练完成后已经生成最佳模型的 `test_*` 结果。若要再次测试同一模型，或评价患者模式选定的最佳模型，将下面的运行目录替换成实际路径：

```powershell
$checkpoint = 'G:\深度学习实验心拍\outputs\runs\实际运行目录\best_model.pth'
& 'F:\实验文件夹\lwnet-master0\.venv\Scripts\python.exe' 'G:\深度学习实验心拍\code\evaluate.py' --checkpoint $checkpoint
```

`evaluate.py` 自动恢复课堂模式的心拍行索引或患者模式的患者名单，不重新随机抽样。课堂模式每轮均已测试，无需添加 `--evaluate-test`；患者模式仍可在确定最终配置后加该开关，训练结束再评价保留的测试集。开发子集模型的成绩会明确标记，不能当作完整实验成绩。

## 每次实验保存的结果

每次运行在 `outputs\runs\时间-模型-评价方式-随机后缀` 下建立独立目录，不覆盖上次结果：

- `config.json`：参数、环境、实际设备、评价协议、选模指标和是否开发子集。
- `preprocessing_summary.json`：记录、类别计数、被丢弃的心拍和预处理规则。
- `split_manifest.json`：具体心拍行索引或患者名单，以及各集合样本数量。
- `best_model.pth`：最佳权重、模型结构和复现实验必需的配置。
- `history.csv`、`learning_curves.png`：课堂模式为 `train_*`/`test_*`，患者模式为 `train_*`/`val_*`。
- `heartbeat_examples.png`、`denoising_comparison.png`、`class_distribution.png`：报告可用的数据说明图。
- `test_metrics.json`、`test_classification_report.txt`、`test_confusion_matrix.png`、`test_predictions.csv`：课堂模式训练后自动生成的测试结果；预测文件包含记录号、R 峰采样位置、真值、预测和类别概率。
- 患者模式训练后保存同名 `validation_*` 文件，正式测试后生成 `test_*` 文件。

报告应使用自己实际运行的测试 Accuracy，以及各类 Precision/Recall/F1、Macro-F1、混淆矩阵。按真实类别逐行统计的正确比例即该类 Recall。课堂复现模式需要注明测试集参与选模，不能写成独立验证集选模或新患者泛化结论；患者模式与课堂模式的成绩分别报告。
