# 设计约定 — AI Toolkit 前端拍板（2026-08-26）

> **状态**：产品已拍板；配合引擎 [#284](https://github.com/wochenlong/lora-scripts-next/issues/284)、Klein [#299](https://github.com/wochenlong/lora-scripts-next/issues/299)、映射表 [#300](https://github.com/wochenlong/lora-scripts-next/issues/300)。  
> **后端先行**；本文锁定前端心智与首版范围，实施可后置。

---

## 1. 入口：同模型多引擎（对齐 Anima）

与现有 Anima（Kohya / Fast）相同：**同一模型族，可选不同引擎**。

| 模型 | 既有引擎（保留） | 新增 |
|------|------------------|------|
| Klein | — | **AI Toolkit**（首发） |
| Anima | Kohya / Fast | **AI Toolkit** 平行路径 |
| SDXL | Kohya | **AI Toolkit** 平行路径 |
| Krea2 | Musubi | **AI Toolkit** 平行路径 |

选择器仍是「模型 → 引擎 → 目标」；不因 Toolkit 另开第二套首页卡片语义。

---

## 2. 首版范围

四模型 **全做**（Klein / Anima / SDXL / Krea2），不缩成仅 Klein。

---

## 3. 训练长度：只显示 steps（明确不做 epoch）

- AI Toolkit 上游以 **`train.steps`** 为训练长度；**无 epoch 一等配置**（社区与 `TrainConfig` 均按 steps；时长用 steps + `num_repeats` 等间接控制）。
- Next 工作台（AI Toolkit 路径）：**只展示 / 只提交 steps**。
- **明确不做**：不为 Toolkit 假造 epoch 控件、不做 epoch→steps 换算、不在映射表里把 Kohya 的 `max_train_epochs` 对齐到 Toolkit。
- 与 Kohya / Musubi 路径的 epoch UI 仅存在于**各自引擎草稿**，互不冒充、互不串改。

---

## 4. 底模填写

- 首版 **不支持 HF 在线下载**。
- 使用 **本机仓库 / 目录路径**（Diffusers 布局或 Toolkit 要求的本地路径）；filepicker / 手填路径即可。
- 模型管理器、远程 ID 下载另议，不挡 Toolkit 前端首版。

---

## 5. 草稿隔离

草稿 key：**模型 × 引擎**（及既有 target 维若需要）**相互独立**。

换引擎不携带另一引擎草稿；同模型不同引擎各存一份。与 Anima 多引擎草稿策略对齐并写死为「独立」。

---

## 6. 右侧预览

继续 **TOML 预览**（用户心智不变）。

- 展示层可按映射表用统一字段名生成可读 TOML。
- Adapter 提交时再写成 Toolkit 原生 YAML / job config；**不**把右侧预览改成 YAML 主导。

---

## 7. 引擎专有角标

原则见 [`training-form-section-order.md`](./training-form-section-order.md) §2.3。细化：

- 控件旁 **小圆点**，或 **小问号**；点击 / hover 说明「此为当前引擎专用参数」。
- 不另开「AI Toolkit 专区」；角标不进 TOC。

---

## 8. 参考图路径（编辑）

见 [`image-edit-dataset-contract.md`](./image-edit-dataset-contract.md)：

- `control_data_dirs[]` **可增删**，不写死路数 UI。
- 上限：**最多 5** 个参考图目录。
- Klein 编辑路径控件与 **Toolkit 前端同批落地**（[#252](https://github.com/wochenlong/lora-scripts-next/issues/252) 已改口径）。

---

## 9. Klein 变体选择

- **先按体量**：4B / 9B。
- 进入后再选类型（base / 蒸馏等）；**默认 base**。
- base 与蒸馏均支持（以后端 Toolkit 可训为准）。

---

## 10. 相关

- 分区排序 [`training-form-section-order.md`](./training-form-section-order.md)
- 编辑数据集 [`image-edit-dataset-contract.md`](./image-edit-dataset-contract.md)
- 统一词映射 [#300](https://github.com/wochenlong/lora-scripts-next/issues/300)
- 引擎管理壳 [`training-engine-management.md`](./training-engine-management.md)
