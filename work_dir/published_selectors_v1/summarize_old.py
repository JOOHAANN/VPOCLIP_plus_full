import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'mflstm_a2c_old_split'


def main():
    d = json.loads((MODEL / 'summary.json').read_text())
    selected = str(d['selection']['seed'])
    lines = [
        '# mfLSTM-A2C 旧划分复现结果', '',
        '数据：旧 50/5 划分；训练池 2,734 个完整四视角 seen episode；测试集 414 个真实 unseen 5-way episode。',
        'mfLSTM 使用 10 轮缺失观测预训练 + 40 轮 A2C；训练种子 20260909/10/11；评测种子 20260920–20260949。',
        'VPOCLIP 冻结，最终统一使用 weighted fusion。模型选择只看旧验证集 fixed0 的 equal-fusion 增益；三个种子均为 0，因此按种子顺序选择 20260909。', '',
        '这里的 CI 是 5,000 次 paired episode bootstrap，先对 30 个评测种子取平均；不把 30 个评测种子当成 30 个独立测试集。', '',
        '## 真实 unseen 5-way：验证选择种子 20260909', '',
        '| 起点 | Fusion | mfLSTM | Random | mfLSTM-Random | 95% CI(pp) | v5 anchor | mfLSTM-v5(pp) |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for protocol in ['fixed0','fixed1','fixed2','fixed3','random_start']:
        for fusion in ['equal_mean','lightweight_gating','rule_cls','rule_geo','rule_combined']:
            r = next(x for x in d['by_train_seed'][selected] if x['group']=='true_unseen_5way' and x['protocol']==protocol and x['fusion']==fusion)
            ci = r['delta_random_ci95']
            lines.append(f"| {protocol} | {fusion} | {r['top1']*100:.2f}% | {r['random_top1']*100:.2f}% | {r['delta_random']*100:+.2f} | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | {r['anchor_top1']*100:.2f}% | {r['delta_anchor']*100:+.2f} |")
    lines += ['', '## 真实 unseen：3 个训练种子均值（随机起点）', '', '| Fusion | mfLSTM | Random | v5 anchor |', '|---|---:|---:|---:|']
    for fusion in ['equal_mean','lightweight_gating','rule_cls','rule_geo','rule_combined']:
        rows = [x for values in d['by_train_seed'].values() for x in values if x['group']=='true_unseen_5way' and x['protocol']=='random_start' and x['fusion']==fusion]
        lines.append(f"| {fusion} | {np.mean([x['top1'] for x in rows])*100:.2f}% ± {np.std([x['top1'] for x in rows],ddof=1)*100:.2f} | {np.mean([x['random_top1'] for x in rows])*100:.2f}% | {np.mean([x['anchor_top1'] for x in rows])*100:.2f}% |")
    lines += ['', '## 结论', '',
              '1. 旧划分随机起点上，mfLSTM 的 equal-mean 三种子均值为 52.11%，随机为 52.02%，优势约 +0.09pp，不能称为稳定提升。',
              '2. RuleCls 三种子均值为 51.58%，随机为 51.14%，约 +0.44pp；但这是多个融合规则之一，必须结合上表 CI 解读。',
              '3. 验证选择的 20260909 在 equal-mean 随机起点为 51.06%，低于随机 52.02%；说明种子差异明显，不能只看三种子均值或单个 checkpoint。',
              '4. 这次结果证明 mfLSTM 能在旧协议上完整运行，但尚未证明它稳定优于你的 v5 或随机策略。', '',
              '## 文件', '',
              '- 训练/测试汇总：`mflstm_a2c_old_split/summary.json`',
              '- 逐种子 checkpoint 与日志：`mflstm_a2c_old_split/seed_*/`',
              '- 真正旧 v5 weighted 锚点：`v5_old_weighted_anchor/`',
              '- 旧划分训练日志：`../../logs/published_selectors_v1/mflstm_a2c_old_split_v5_anchor_final.log`',
    ]
    (ROOT / 'OldSplit_MF_LSTM_Report.md').write_text('\n'.join(lines))
    print('OLD_REPORT_COMPLETE', flush=True)


if __name__ == '__main__': main()
