from __future__ import annotations

import queue
import sys
import time
import os
from collections import deque
from pathlib import Path
from typing import Any


JOINT_NAMES = [
    "j1",
    "j2",
    "j3",
    "j4",
    "j5",
    "j6",
    "gripper",
]

TOPICS = [
    ("cmd_vla_30hz", "VLA Cmd", (213, 94, 0)),
    ("cmd_high_follow_200hz", "HighFollow Cmd", (0, 102, 204)),
    ("state_200hz", "State", (0, 140, 70)),
]


class RingSeries:
    def __init__(self, window_sec: float):
        self.window_sec = float(window_sec)
        self.data: dict[str, list[deque[tuple[float, float]]]] = {
            topic: [deque() for _ in range(14)] for topic, _, _ in TOPICS
        }
        self.t0: float | None = None

    def append(self, item: dict[str, Any]) -> None:
        topic = str(item.get("topic", ""))
        if topic not in self.data:
            return
        values = item.get("values")
        if not isinstance(values, list) or len(values) < 14:
            return
        t = float(item.get("monotonic_sec", time.monotonic()))
        if self.t0 is None:
            self.t0 = t
        x = t - self.t0
        cutoff = x - self.window_sec
        for idx in range(14):
            series = self.data[topic][idx]
            series.append((x, float(values[idx])))
            while series and series[0][0] < cutoff:
                series.popleft()

    def xy(self, topic: str, index: int) -> tuple[list[float], list[float]]:
        series = self.data[topic][index]
        if not series:
            return [], []
        xs, ys = zip(*series)
        return list(xs), list(ys)


def run_viewer(telemetry_queue, window_sec: float = 10.0, default_arms=("left",), default_topics=None) -> None:
    try:
        _prefer_pyqt_plugin_path()
        from PyQt5 import QtCore
        from PyQt5 import QtWidgets
        import pyqtgraph as pg
    except Exception as exc:
        print(f"[realtime-plot] PyQt5/pyqtgraph import failed: {exc}", file=sys.stderr)
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(
        """
        QWidget { background: white; color: #111111; }
        QCheckBox { background: white; color: #111111; padding: 2px 6px; }
        QScrollArea { background: white; border: none; }
        """
    )
    pg.setConfigOptions(antialias=False)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    default_topics = set(default_topics or ("cmd_high_follow_200hz", "state_200hz"))
    series = RingSeries(window_sec=window_sec)

    window = QtWidgets.QWidget()
    window.setWindowTitle("Kai0 SDK Realtime Joint Curves")
    layout = QtWidgets.QVBoxLayout(window)

    toolbar = QtWidgets.QHBoxLayout()
    left_box = QtWidgets.QCheckBox("Left Arm")
    right_box = QtWidgets.QCheckBox("Right Arm")
    pause_box = QtWidgets.QCheckBox("Pause")
    left_box.setChecked("left" in set(default_arms))
    right_box.setChecked("right" in set(default_arms))
    toolbar.addWidget(left_box)
    toolbar.addWidget(right_box)
    toolbar.addSpacing(16)

    topic_boxes = {}
    for topic, label, _ in TOPICS:
        box = QtWidgets.QCheckBox(label)
        box.setChecked(topic in default_topics)
        topic_boxes[topic] = box
        toolbar.addWidget(box)
    toolbar.addSpacing(16)
    toolbar.addWidget(pause_box)
    toolbar.addStretch(1)
    layout.addLayout(toolbar)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    content = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(content)
    scroll.setWidget(content)
    layout.addWidget(scroll)

    plots: dict[tuple[str, int], Any] = {}
    curves: dict[tuple[str, int, str], Any] = {}

    def rebuild_plots() -> None:
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        plots.clear()
        curves.clear()
        arms = []
        if left_box.isChecked():
            arms.append(("left", 0))
        if right_box.isChecked():
            arms.append(("right", 7))
        row = 0
        for arm_name, offset in arms:
            for joint_i, joint_name in enumerate(JOINT_NAMES):
                idx = offset + joint_i
                plot = pg.PlotWidget(title=f"{arm_name}_{joint_name}")
                plot.setBackground("w")
                plot.showGrid(x=True, y=True, alpha=0.18)
                plot.getAxis("left").setPen(pg.mkPen("#222222"))
                plot.getAxis("left").setTextPen(pg.mkPen("#222222"))
                plot.getAxis("bottom").setPen(pg.mkPen("#222222"))
                plot.getAxis("bottom").setTextPen(pg.mkPen("#222222"))
                plot.addLegend(offset=(8, 8))
                plot.setMinimumHeight(150)
                plots[(arm_name, idx)] = plot
                for topic, label, color in TOPICS:
                    pen = pg.mkPen(color=color, width=1.5)
                    curves[(arm_name, idx, topic)] = plot.plot([], [], pen=pen, name=label)
                grid.addWidget(plot, row, 0)
                row += 1

    def drain_queue() -> None:
        for _ in range(2000):
            try:
                item = telemetry_queue.get_nowait()
            except queue.Empty:
                return
            except Exception:
                return
            series.append(item)

    def refresh() -> None:
        drain_queue()
        if pause_box.isChecked():
            return
        active_topics = {topic for topic, box in topic_boxes.items() if box.isChecked()}
        for (arm_name, idx), plot in list(plots.items()):
            for topic, _, _ in TOPICS:
                curve = curves[(arm_name, idx, topic)]
                if topic not in active_topics:
                    curve.setData([], [])
                    continue
                xs, ys = series.xy(topic, idx)
                curve.setData(xs, ys)
            plot.enableAutoRange(axis="y", enable=True)

    left_box.stateChanged.connect(rebuild_plots)
    right_box.stateChanged.connect(rebuild_plots)
    rebuild_plots()

    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(50)

    window.resize(1150, 900)
    window.show()
    app.exec_()


if __name__ == "__main__":
    print("realtime_curve_viewer.py is started by agilex_inference_openpi.py when telemetry_plot.enabled=true")


def _prefer_pyqt_plugin_path() -> None:
    try:
        import PyQt5
    except Exception:
        return
    plugin_path = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
    if plugin_path.is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_path / "platforms")
        os.environ.pop("QT_PLUGIN_PATH", None)
