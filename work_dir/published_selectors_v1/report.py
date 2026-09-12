"""Generate tables only after both methods finish; never select on test."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    summaries = {m: json.loads((HERE / m / 'summary.json').read_text()) for m in ['mvselect', 'mflstm_a2c']}
    lines = ['# MVSelect / mfLSTM-A2C 与 v5：同一 weighted-fusion 协议', '',
             '这是两个方法迁移到 ETRI 的适配实验，不是原论文数据集的复现精度。详细差异见 [README](README.md)。', '',
             'VPOCLIP 冻结；2,900 个四视角 seen 训练样本；3 个训练 seed；30 个相同评测 seed。'
             '两种方法都只按验证集选 checkpoint/seed，不按 unseen 选模型。', '',
             '**重要审计**：历史 v5 的 pseudo-unseen 验证类别也在训练类别列表内，'
             '因此这里不是类别隔离验证。真实 unseen 仍为 5-way，仅用于测试。', '',
             '表中为验证集选中的模型；± 表示 3 个训练 seed 的测试准确率标准差。'
             '差值 CI 为配对样本 bootstrap（先平均 30 个评测 seed，5,000 次重采样）。'
             '它不包含受试者聚类、超参数选择和多重检验的不确定性，不能只凭微小正差宣称稳定领先。', '']
    for name, d in summaries.items():
        lines += [f"- {name}: 验证选择 seed {d['selection']['seed']}，epoch {d['selection']['epoch']}。"]
    for group in ['true_unseen_5way', 'seen']:
        for moves in [1, 2]:
            for fusion in ['rule_cls', 'rule_geo', 'rule_combined', 'lightweight_gating', 'equal_mean']:
                lines += ['', f'## {group} / 移动 {moves} 次 / {fusion}', '',
                          '| 起点 | Random | v5 锚点 | 方法 | 验证所选 Top1 | 3训练seed均值±SD | 相对v5 (pp) [95% CI] | 相对Random (pp) [95% CI] |',
                          '|---|---:|---:|---|---:|---:|---:|---:|']
                for protocol in ['fixed0','fixed1','fixed2','fixed3','random_start']:
                    for name, d in summaries.items():
                        rows = []
                        for seed, entries in d['by_train_seed'].items():
                            r = next(x for x in entries if (x['group'],x['protocol'],x['moves'],x['fusion']) == (group,protocol,moves,fusion))
                            rows.append(r['top1'])
                            if seed == str(d['selection']['seed']): chosen = r
                        def delta(k):
                            a,b = chosen[k+'_ci95']
                            return f"{chosen[k]*100:+.2f} [{a*100:+.2f}, {b*100:+.2f}]"
                        lines.append(f"| {protocol} | {chosen['random_top1']*100:.2f}% | {chosen['v5_top1']*100:.2f}% | {name} | {chosen['top1']*100:.2f}% | {np.mean(rows)*100:.2f}±{np.std(rows,ddof=1)*100:.2f}% | {delta('delta_v5')} | {delta('delta_random')} |")
    lines += ['', '## 可复查文件', '', '- 每个方法 `summary.json` 含全部训练种子的结果。',
              '- `seed_*/evaluation/` 含逐样本路径、预测、标签和 30 个评测种子。',
              '- `seed_*/audit.json`、`train.jsonl`、`best.pt`、`last.pt` 保留训练证据。',
              '- 原始 v5 的文件没有重写；Random 与 v5 逐样本输出直接来自锚点。', '']
    (HERE / 'Comparison_Report.md').write_text('\n'.join(lines))
    print('REPORT_COMPLETE', flush=True)


if __name__ == '__main__': main()
