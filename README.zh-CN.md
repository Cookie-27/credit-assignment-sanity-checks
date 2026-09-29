# 信用分配检查工具

[![自动检查](https://github.com/Cookie-27/credit-assignment-sanity-checks/actions/workflows/checks.yml/badge.svg)](https://github.com/Cookie-27/credit-assignment-sanity-checks/actions/workflows/checks.yml)

[English](README.md)

用可精确计算的小实验，检查“评分 → 优势 → 更新”之间是否符合预期。
Python 3.10+、纯标准库、CPU运行；无需安装依赖、下载模型或使用API。

首版是一个可复现的研究小工具，包含两步信用计算诊断和相同决策合并示例。
其中的轨迹内归一化、平滑和保护项参考
[HCAPO发表公式](https://arxiv.org/html/2603.08754v1)。代码独立实现，后验使用精确Bayes值；
任务、模型和提示都不同于作者实验，不能称为作者代码复现。

## 直接运行

克隆仓库（也可下载并解压源码ZIP），然后运行：

```bash
git clone https://github.com/Cookie-27/credit-assignment-sanity-checks.git
cd credit-assignment-sanity-checks
python -m credit_sanity demo
```

查看对照并核验全部结果：

```bash
python -m credit_sanity demo --q0 1 --q1 1 --p 0.9
python -m credit_sanity demo --echo
python -m credit_sanity coefficient-demo
python -m unittest discover -s tests -v
python -m credit_sanity verify
python -m credit_sanity report
```

重新生成全部结果：

```bash
python -m credit_sanity sweep --output results/theory.json
python -m credit_sanity report --output results/summary.md
```

本地已用Windows/Python 3.12验证；页面顶部的CI标记链接到Windows、Ubuntu与
Python 3.10、3.12的最新检查结果。Windows若使用Python启动器，可将`python`替换为`py -3`。

不运行代码也可以先看[自动生成的结果摘要](results/summary.md)。

## 能看见什么

实验第一步二选一，第二步只能提交。最终奖励为0或1，策略只有一个可训练logit。
默认成功概率为`q=(1,.99)`，动作0概率为`.9`，组大小为8。

| 指标 | 精确值 |
|---|---:|
| 真实终局奖励梯度 | +0.000900000000 |
| 普通组标准化奖励梯度 | +0.002377806524 |
| 轨迹内评分代理＋成功保护 | +0.025626506224 |
| 再加相邻平滑 | −0.180982441841 |

该设置下，平滑代理的首步期望更新方向与真实奖励梯度相反。
完整240条件全部提供，避免只展示一个反例。

同样保留三个限制结论：

1. 两动作真实价值相同，策略漂移也可以完全不损害奖励。
2. 给出完整动作回显后，本固定网格中的反向和同值漂移全部消失。
3. 相同输入、相同动作的信用排名互换，可能不改变合并后的更新系数。

所以，本项目支持有限设置中的精确诊断，不据此宣称某方法普遍较差、真实LLM训练受损，
或已经发明新RL算法。

## 如何检查结果

主程序按四类动作/奖励的计数枚举，用多项分布计算精确期望；测试另用独立的有序序列枚举对照。
还包括目标有限差分、HCA恒等式、两步平滑恒等式和边界输入检查。
`verify`会重新计算全部240条件，逐项比较已提交结果。

当某动作完全无法获得正奖励时，逆比率HCA对照会明确返回`null`，不静默跳过未定义项。
原始理论的概率支持条件及本例适用范围见[研究笔记](docs/research_note.md)。

[复现说明](REPRODUCIBILITY.md) · [研究笔记](docs/research_note.md) ·
[贡献指南](CONTRIBUTING.md) · [引用](REFERENCES.md) · [MIT许可证](LICENSE)
