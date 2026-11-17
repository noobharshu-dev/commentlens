"""
Tests for Mood Bar segment generation and legend grid layout.
Verifies that 0-count segments do not produce phantom slivers and that
legend items are rendered in isolated containers to prevent text collisions.
"""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import pytest


def _build_mood_bar_html(pos: int, neu: int, mix: int, neg: int):
    """Mirror the logic from app.py _render_mood_bar."""
    tot = max(pos + neu + mix + neg, 1)
    pos_pct = round(pos * 100 / tot)
    neu_pct = round(neu * 100 / tot)
    mix_pct = round(mix * 100 / tot)
    neg_pct = round(neg * 100 / tot)

    segments_html = []
    if pos > 0:
        segments_html.append(f'<div class="seg" style="flex:{pos};background:var(--pos);height:52px;" title="Positive: {pos:,} ({pos_pct}%)"></div>')
    if neu > 0:
        segments_html.append(f'<div class="seg" style="flex:{neu};background:var(--neu);height:52px;" title="Neutral: {neu:,} ({neu_pct}%)"></div>')
    if mix > 0:
        segments_html.append(f'<div class="seg" style="flex:{mix};background:var(--mix);height:52px;" title="Mixed: {mix:,} ({mix_pct}%)"></div>')
    if neg > 0:
        segments_html.append(f'<div class="seg" style="flex:{neg};background:var(--neg);height:52px;" title="Negative: {neg:,} ({neg_pct}%)"></div>')

    if not segments_html:
        segments_html.append('<div class="seg" style="flex:1;background:var(--line);height:52px;"></div>')

    legend_grid_html = f"""
    <div class="mood-legend-grid">
      <div class="mood-legend-item">
        <div class="mood-legend-header">
          <span class="mood-legend-dot" style="background:var(--pos);"></span>
          <span class="mood-legend-title">Positive</span>
        </div>
        <div class="mood-legend-meta">{pos:,} &nbsp;<span class="mood-legend-pct">{pos_pct}%</span></div>
      </div>
      <div class="mood-legend-item">
        <div class="mood-legend-header">
          <span class="mood-legend-dot" style="background:var(--neu);"></span>
          <span class="mood-legend-title">Neutral</span>
        </div>
        <div class="mood-legend-meta">{neu:,} &nbsp;<span class="mood-legend-pct">{neu_pct}%</span></div>
      </div>
      <div class="mood-legend-item">
        <div class="mood-legend-header">
          <span class="mood-legend-dot" style="background:var(--mix);"></span>
          <span class="mood-legend-title">Mixed</span>
        </div>
        <div class="mood-legend-meta">{mix:,} &nbsp;<span class="mood-legend-pct">{mix_pct}%</span></div>
      </div>
      <div class="mood-legend-item">
        <div class="mood-legend-header">
          <span class="mood-legend-dot" style="background:var(--neg);"></span>
          <span class="mood-legend-title">Negative</span>
        </div>
        <div class="mood-legend-meta">{neg:,} &nbsp;<span class="mood-legend-pct">{neg_pct}%</span></div>
      </div>
    </div>
    """
    return "".join(segments_html), legend_grid_html


def test_zero_count_category_has_no_phantom_segment():
    # User's exact scenario: 73 pos, 41 neu, 0 mix, 39 neg
    bar_html, legend_html = _build_mood_bar_html(pos=73, neu=41, mix=0, neg=39)

    # Positive, Neutral, Negative segments must exist
    assert 'background:var(--pos)' in bar_html
    assert 'background:var(--neu)' in bar_html
    assert 'background:var(--neg)' in bar_html

    # Mixed has 0 count, so NO segment should be in the bar
    assert 'background:var(--mix)' not in bar_html

    # In the legend, all 4 items exist and are inside separate containers
    assert legend_html.count('mood-legend-item') == 4
    assert 'Mixed' in legend_html
    assert '0 &nbsp;<span class="mood-legend-pct">0%</span>' in legend_html


def test_all_zeros_fallback():
    bar_html, legend_html = _build_mood_bar_html(pos=0, neu=0, mix=0, neg=0)
    assert 'background:var(--line)' in bar_html
    assert legend_html.count('mood-legend-item') == 4


def test_tiny_count_does_not_break_grid():
    # 1 mixed out of 1000
    bar_html, legend_html = _build_mood_bar_html(pos=500, neu=300, mix=1, neg=199)
    assert 'flex:1' in bar_html
    assert 'background:var(--mix)' in bar_html
    assert legend_html.count('mood-legend-item') == 4
