# 今晚三小时：最短交付路线

已经完成：真实数据模型训练与评估、4项API测试、报告初稿、5页PPT、流程图、保存/重载模型。

你需要做的只剩这些：

1. 打开 `docs/P2_Draft_Report.docx`，看是否理解主要数字（新版 0.5907 RMSLE，平均误差 6.18 分钟）。
2. 打开 `docs/P2_Stakeholder_Draft.pptx`，修改任何你不喜欢的措辞；这一版是5页、约5分钟。
3. Windows VS Code 从项目目录打开终端，按英文 README 命令运行 `python src/check_saved_model.py` 和 `python -m pytest -q tests/test_api.py`。
4. 要展示 FastAPI：运行 `uvicorn app:app --reload`，浏览器打开 http://127.0.0.1:8000/docs；点 POST /predict → Try it out → 输入 `{"distance_km":5,"pickup_hour":14}` → Execute。
5. 核实 `.gitignore` 已排除 `data/train.zip`，再上传 GitHub；请同学独立运行并签 P2 验证表。
6. Canvas 的 P2 截止时间仍需核实；在真正提交前务必确认教授最新说明。

注意：完整重新训练需要从 Kaggle 合规获得的 `data/train.zip`。已经保存的模型可直接用于 API 演示，但不能替代同学独立复现。
