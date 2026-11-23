"""Rebuild the OCPE method diagram from explicit, reviewed labels (CPU only)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.font_manager import FontProperties


OUT = Path(__file__).resolve().parent
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
INK = "#203449"
BLUE = "#e9f2fb"
GREEN = "#e7f4ed"
GRAY = "#f2f4f7"


def box(ax, x, y, w, h, label, fill=BLUE, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.012,rounding_size=0.02",
                              facecolor=fill, edgecolor="#a2b4c7", linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontproperties=FONT, fontsize=size, color=INK, linespacing=1.65)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=13, linewidth=1.25, color="#59748b"))


def panel(ax, title, proposed=False):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.015, 0.015), 0.97, 0.97,
                              boxstyle="round,pad=0.005,rounding_size=0.025",
                              facecolor="white", edgecolor="#b5c1cb",
                              linestyle="--" if proposed else "-", linewidth=1.2))
    ax.text(.05, .935, title, fontproperties=FONT, fontsize=13,
            color="#667589" if proposed else "#17634e", va="center")


def fork(ax, left, right, bottom, left_fill=BLUE):
    box(ax, .29, .755, .42, .105, "真实可见历史＋原始掩码", GRAY)
    box(ax, .06, .435, .40, .19, left, left_fill, size=11)
    box(ax, .54, .435, .40, .19, right, GREEN, size=11)
    arrow(ax, (.42, .75), (.26, .635))
    arrow(ax, (.58, .75), (.74, .635))
    box(ax, .29, .255, .42, .09, "投影后相加")
    arrow(ax, (.26, .425), (.40, .35))
    arrow(ax, (.74, .425), (.60, .35))
    box(ax, .13, .065, .74, .105, bottom, GRAY, size=11)
    arrow(ax, (.5, .245), (.5, .18))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 10.3))
    fig.patch.set_facecolor("#f7f9fb")
    fig.suptitle("OCPE：已实现的 S4 结构与候选强主干扩展", fontproperties=FONT,
                 fontsize=20, color=INK, y=.982)

    ax = axes[0, 0]
    panel(ax, "A  纯后验输入版｜已实现")
    for y, label, fill in [(.755, "真实可见历史＋原始掩码", GRAY),
                           (.525, "OCPE：逐前缀后验均值与对数方差", GREEN),
                           (.295, "特征投影 → 普通 S4", BLUE),
                           (.065, "预测头 → 未来点预测", GRAY)]:
        box(ax, .10, y, .80, .125, label, fill, size=12)
    for top in [.755, .525, .295]:
        arrow(ax, (.5, top - .012), (.5, top - .09))

    ax = axes[0, 1]
    panel(ax, "B  历史分支融合版 F3｜已实现")
    fork(ax, "线性插值＋原始掩码\n历史投影", "OCPE 后验编码\n独立投影",
         "普通 S4 → 预测头 → 未来点预测")

    ax = axes[1, 0]
    panel(ax, "C  OCPE＋S4M｜候选，尚未验证", proposed=True)
    fork(ax, "衰减插补\nATPM 原型检索与融合", "OCPE 逐前缀编码\n独立投影",
         "原有 MDS-S4＋掩码流 → 预测")

    ax = axes[1, 1]
    panel(ax, "D  OCPE＋CTF｜候选，尚未验证", proposed=True)
    fork(ax, "原有 patch/token 编码\n最终通道表示", "OCPE 窗口末端摘要\n通道相关投影",
         "原有通道预测头 → 未来点预测")

    fig.text(.5, .017, "OCPE 始终只把真实观测作为证据；方差属于潜变量，不是未来预测区间。",
             ha="center", fontproperties=FONT, fontsize=12, color="#52677c")
    fig.subplots_adjust(left=.025, right=.975, top=.93, bottom=.055, hspace=.10, wspace=.065)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"architecture.{ext}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
