"""生成静态 HTML 项目报告。"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import ModelSpec


def _df_to_html(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    if df.empty:
        return "<p>无记录。</p>"
    return df.to_html(
        index=False,
        border=0,
        classes="data-table",
        float_format=lambda value: float_format.format(value),
    )


def _relative_image(path: Path, report_path: Path) -> str:
    return escape(path.relative_to(report_path.parent).as_posix())


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _format_speed(value: float) -> str:
    return f"{value:.6f}"


def _overfitting_comment(gap: float) -> str:
    if gap >= 0.05:
        return "训练集精度明显高于测试集精度，存在一定过拟合现象。"
    if gap <= -0.02:
        return "测试集精度略高于训练集精度，未观察到过拟合，可能与数据划分或清洗后分布有关。"
    return "训练集和测试集精度差异较小，泛化表现较稳定。"


def _robustness_comment(model_robustness: pd.DataFrame) -> str:
    if model_robustness.empty:
        return "本次运行跳过了鲁棒性实验，因此没有该算法的加噪训练结果。"
    worst = model_robustness.sort_values("accuracy_drop", ascending=False).iloc[0]
    mean_drop = model_robustness["accuracy_drop"].mean()
    return (
        f"平均精度下降为 {_format_metric(mean_drop)}；最明显下降出现在 "
        f"{worst['noise_type']} 噪声、强度 {worst['strength']:.2f}，"
        f"下降 {_format_metric(worst['accuracy_drop'])}。"
    )


def _render_image_grid(paths: Iterable[Path], report_path: Path) -> str:
    figures = []
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            continue
        figures.append(
            f'<figure><img src="{_relative_image(path, report_path)}" '
            f'alt="{escape(path.stem)}"><figcaption>{escape(path.stem)}</figcaption></figure>'
        )
    if not figures:
        return "<p>没有可展示的图表。</p>"
    return '<div class="figure-grid">' + "".join(figures) + "</div>"


def _render_model_sections(
    report_path: Path,
    metrics: pd.DataFrame,
    robustness: pd.DataFrame,
    classification_reports: pd.DataFrame,
    model_specs: Iterable[ModelSpec],
    individual_plot_paths: dict[str, list[Path]],
) -> str:
    sections = []
    metric_lookup = metrics.set_index("model")

    for spec in model_specs:
        if spec.name not in metric_lookup.index:
            continue
        row = metric_lookup.loc[spec.name]
        model_robustness = robustness[robustness["model"] == spec.name]
        model_report = classification_reports[
            classification_reports["model"] == spec.name
        ].copy()
        if not model_report.empty:
            model_report = model_report.drop(columns=["model"])

        loss_note = (
            "该算法使用迭代优化，本报告已单独给出训练集和验证集 loss 曲线。"
            if spec.training_strategy in {"sgd_log_loss", "sgd_hinge"}
            else "该算法不是按 epoch 迭代记录 loss 的训练方式，因此不单独绘制 loss 曲线。"
        )

        metric_cards = f"""
        <div class="metric-cards">
          <div><strong>{_format_metric(row['train_accuracy'])}</strong><span>训练集准确率</span></div>
          <div><strong>{_format_metric(row['val_accuracy'])}</strong><span>验证集准确率</span></div>
          <div><strong>{_format_metric(row['test_accuracy'])}</strong><span>测试集准确率</span></div>
          <div><strong>{_format_metric(row['test_macro_f1'])}</strong><span>宏平均 F1</span></div>
          <div><strong>{_format_metric(row['train_test_accuracy_gap'])}</strong><span>过拟合差异</span></div>
          <div><strong>{_format_speed(row['inference_ms_per_sample'])}</strong><span>毫秒/样本</span></div>
        </div>
        """

        sections.append(
            f"""
      <article class="model-block" id="{escape(spec.name)}">
        <h3>{escape(spec.display_name)} <span>{escape(spec.course_source)}</span></h3>
        <p>{escape(spec.description)}</p>
        {metric_cards}
        <p><strong>过拟合分析：</strong>{_overfitting_comment(float(row['train_test_accuracy_gap']))}</p>
        <p><strong>推理速度：</strong>测试集平均预测耗时 {_format_speed(row['inference_seconds_mean'])} 秒，
        折合单样本 {_format_speed(row['inference_ms_per_sample'])} ms。</p>
        <p><strong>鲁棒性分析：</strong>{escape(_robustness_comment(model_robustness))}</p>
        <p><strong>Loss 曲线说明：</strong>{loss_note}</p>
        <h4>分类报告</h4>
        {_df_to_html(model_report)}
        <h4>单算法图表</h4>
        {_render_image_grid(individual_plot_paths.get(spec.name, []), report_path)}
      </article>
            """
        )

    return "\n".join(sections)


def generate_static_report(
    output_dir: Path,
    cleaning_stats: pd.DataFrame,
    class_distribution: pd.DataFrame,
    metrics: pd.DataFrame,
    robustness: pd.DataFrame,
    classification_reports: pd.DataFrame,
    model_specs: Iterable[ModelSpec],
    comparison_plot_paths: Iterable[Path],
    individual_plot_paths: dict[str, list[Path]] | None = None,
) -> Path:
    """生成包含总览和单算法分析的 outputs/report.html。"""

    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"
    individual_plot_paths = individual_plot_paths or {}
    model_specs = list(model_specs)

    sorted_metrics = metrics.sort_values("test_accuracy", ascending=False)
    model_rows = pd.DataFrame(
        [
            {
                "模型": spec.display_name,
                "来源": spec.course_source,
                "说明": spec.description,
            }
            for spec in model_specs
        ]
    )

    overview_images = _render_image_grid(comparison_plot_paths, report_path)
    model_sections = _render_model_sections(
        report_path=report_path,
        metrics=metrics,
        robustness=robustness,
        classification_reports=classification_reports,
        model_specs=model_specs,
        individual_plot_paths=individual_plot_paths,
    )

    if sorted_metrics.empty:
        best_summary = "暂无模型结果。"
    else:
        best_row = sorted_metrics.iloc[0]
        best_summary = (
            f"当前测试集精度最高的是 {best_row['display_name']}，"
            f"测试集准确率 = {_format_metric(best_row['test_accuracy'])}，"
            f"宏平均 F1 = {_format_metric(best_row['test_macro_f1'])}。"
        )

    robustness_summary = (
        robustness.groupby("display_name")["accuracy_drop"]
        .mean()
        .reset_index()
        .rename(columns={"display_name": "模型", "accuracy_drop": "平均准确率下降"})
        .sort_values("平均准确率下降")
        if not robustness.empty
        else pd.DataFrame()
    )

    model_names_present = set(metrics["model"]) if "model" in metrics.columns else set()
    individual_index = "".join(
        f'<a href="#{escape(spec.name)}">{escape(spec.display_name)}</a>'
        for spec in model_specs
        if spec.name in model_names_present
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dry Bean Dataset 多分类实验报告</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.6;
      color: #1f2933;
      background: #f7f8fb;
    }}
    header {{
      background: #16324f;
      color: white;
      padding: 32px 9vw;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }}
    section, .model-block {{
      background: white;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 20px;
    }}
    h1, h2, h3 {{
      margin-top: 0;
    }}
    h3 span {{
      color: #607083;
      font-size: 15px;
      margin-left: 8px;
      font-weight: 500;
    }}
    h4 {{
      margin-bottom: 8px;
    }}
    .index-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }}
    .index-links a {{
      color: #16324f;
      background: #edf2f7;
      border-radius: 5px;
      padding: 5px 9px;
      text-decoration: none;
      font-size: 14px;
    }}
    .metric-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 16px 0;
    }}
    .metric-cards div {{
      border: 1px solid #dbe3ec;
      border-radius: 6px;
      padding: 10px;
      background: #f8fafc;
    }}
    .metric-cards strong {{
      display: block;
      font-size: 22px;
      color: #16324f;
    }}
    .metric-cards span {{
      display: block;
      color: #526170;
      font-size: 13px;
    }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid #e6eaf0;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: #f0f4f8;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    figure {{
      margin: 0;
    }}
    figcaption {{
      margin-top: 5px;
      color: #526170;
      font-size: 13px;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #d7dee8;
      border-radius: 6px;
      background: white;
    }}
    code {{
      background: #edf2f7;
      padding: 2px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Dry Bean Dataset 多分类机器学习项目</h1>
    <p>包含数据分析、数据清洗、特征工程、五种算法实验、速度/鲁棒性/过拟合对比，以及每个算法的独立分析。</p>
  </header>
  <main>
    <section>
      <h2>项目结论总览</h2>
      <p>{escape(best_summary)}</p>
      <div class="index-links">{individual_index}</div>
    </section>
    <section>
      <h2>数据说明与污染情况</h2>
      <p>数据目标是根据豆类图像提取出的形状特征预测七个干豆品种。原始数据已经由教师划分为训练集、验证集和测试集。</p>
      {_df_to_html(cleaning_stats)}
    </section>
    <section>
      <h2>类别分布</h2>
      {_df_to_html(class_distribution)}
    </section>
    <section>
      <h2>数据处理流程</h2>
      <ol>
        <li>统一列名并解析数值特征，将 <code>?</code>、缺失值和带单位的字符串转成可处理的数值或缺失值。</li>
        <li>修正标签大小写、尾随空格和字符替换污染，例如 <code>D3RMAS0N</code>、<code>H0R0Z</code>。</li>
        <li>对物理上不合理的数值做规则处理，例如负面积转为正面积，非法比例值设为缺失。</li>
        <li>删除训练集重复样本；用训练集的中位数填补缺失值，再用 Z-Score 标准化。</li>
        <li>增加少量具有形状含义的组合特征，例如长短轴比、面积/凸包面积比、周长/面积平方根比。</li>
      </ol>
    </section>
    <section>
      <h2>算法设计</h2>
      {_df_to_html(model_rows)}
    </section>
    <section>
      <h2>五种算法总览对比</h2>
      {_df_to_html(sorted_metrics)}
      <h3>鲁棒性均值排序</h3>
      {_df_to_html(robustness_summary)}
      <h3>总览图表</h3>
      {overview_images}
    </section>
    <section>
      <h2>单算法详细分析</h2>
      <p>下面按算法分别列出指标解释、分类报告和该算法独立图表，便于在论文或 GitHub 页面中逐项展示。</p>
    </section>
    {model_sections}
  </main>
</body>
</html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path
