"""Megane main window: node palette, graph view, params, preview, menus.

Cooking runs on the Qt thread pool so the UI never blocks; results carry a
generation counter so a stale cook can't overwrite a newer one. Parameter and
graph edits trigger a debounced auto-cook, but only when every node in the
preview chain is realtime-capable -- heavy ("bake") chains cook on demand only
(F5 / Cook button), matching the spec's pseudo-live model.
"""
from __future__ import annotations

import os
import traceback

from ..core import backend
from ..core.node import Node, registered_nodes
from .bridge import EngineBridge
from .ng_compat import QtCore, QtGui, QtWidgets
from .param_panel import ParamPanel
from .preview_panel import PreviewPanel

_AUTO_COOK_DEBOUNCE_MS = 300


class _CookSignals(QtCore.QObject):
    done = QtCore.Signal(int, str, object)   # generation, node_id, outputs
    error = QtCore.Signal(int, str, str)     # generation, node_id, traceback


class _CookTask(QtCore.QRunnable):
    def __init__(self, bridge: EngineBridge, node_id: str, generation: int,
                 signals: _CookSignals) -> None:
        super().__init__()
        self.bridge = bridge
        self.node_id = node_id
        self.generation = generation
        self.signals = signals

    def run(self) -> None:  # executed on a pool thread
        try:
            outputs = self.bridge.cook(self.node_id)
            self.signals.done.emit(self.generation, self.node_id, outputs)
        except Exception:  # noqa: BLE001 - report any node failure to the UI
            self.signals.error.emit(self.generation, self.node_id,
                                    traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Megane 眼鏡 — image → sound lens")
        self.resize(1500, 900)

        self.bridge = EngineBridge(parent=self)
        self._selected: Node | None = None
        self._project_path: str | None = None
        self._generation = 0
        self._cook_signals = _CookSignals()
        self._cook_signals.done.connect(self._on_cook_done)
        self._cook_signals.error.connect(self._on_cook_error)

        self.setCentralWidget(self.bridge.ng.widget)
        self._build_palette_dock()
        self._build_param_dock()
        self._build_preview_dock()
        self._build_menus()

        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(_AUTO_COOK_DEBOUNCE_MS)
        self._auto_timer.timeout.connect(self._auto_cook_fire)

        self.bridge.selectionChanged.connect(self._on_selection)
        self.bridge.paramChanged.connect(lambda _eid: self._schedule_auto_cook())
        self.bridge.graphChanged.connect(self._schedule_auto_cook)
        self.bridge.message.connect(lambda m: self.statusBar().showMessage(m, 6000))

        self._engine_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self._engine_label)
        self._refresh_engine_label()

    # -- docks -------------------------------------------------------------
    def _build_palette_dock(self) -> None:
        self.palette_list = QtWidgets.QListWidget()
        for name in sorted(registered_nodes()):
            self.palette_list.addItem(name)
        self.palette_list.setToolTip("double-click to add a node")
        self.palette_list.itemDoubleClicked.connect(
            lambda item: self.bridge.add_node(item.text()))
        dock = QtWidgets.QDockWidget("Nodes", self)
        dock.setWidget(self.palette_list)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def _build_param_dock(self) -> None:
        self.param_panel = ParamPanel()
        self.param_panel.valueEdited.connect(self._on_param_edited)
        dock = QtWidgets.QDockWidget("Parameters", self)
        dock.setWidget(self.param_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _build_preview_dock(self) -> None:
        self.preview = PreviewPanel()
        self.preview.cookRequested.connect(self.cook_now)
        dock = QtWidgets.QDockWidget("Preview", self)
        dock.setWidget(self.preview)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    # -- menus --------------------------------------------------------------
    def _build_menus(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        m_file.addAction("New", QtGui.QKeySequence.New, self._file_new)
        m_file.addAction("Open…", QtGui.QKeySequence.Open, self._file_open)
        m_file.addAction("Save", QtGui.QKeySequence.Save, self._file_save)
        m_file.addAction("Save As…", QtGui.QKeySequence.SaveAs, self._file_save_as)
        m_file.addSeparator()
        m_file.addAction("Quit", QtGui.QKeySequence.Quit, self.close)

        m_engine = self.menuBar().addMenu("&Engine")
        cook_act = m_engine.addAction("Cook", self.cook_now)
        cook_act.setShortcut(QtGui.QKeySequence("F5"))
        self.auto_cook_act = m_engine.addAction("Auto-cook")
        self.auto_cook_act.setCheckable(True)
        self.auto_cook_act.setChecked(True)
        m_engine.addSeparator()

        prec_menu = m_engine.addMenu("Precision")
        self._prec_group = QtGui.QActionGroup(self)
        for prec in ("fp16", "fp32", "fp64"):
            act = prec_menu.addAction(prec)
            act.setCheckable(True)
            act.setChecked(prec == self.bridge.engine.settings["precision"])
            act.triggered.connect(lambda _c, p=prec: self._set_precision(p))
            self._prec_group.addAction(act)

        self.gpu_act = m_engine.addAction("Use GPU (CUDA)")
        self.gpu_act.setCheckable(True)
        self.gpu_act.setEnabled(backend.gpu_available())
        if not backend.gpu_available():
            self.gpu_act.setToolTip("CuPy not installed / no CUDA device")
        self.gpu_act.toggled.connect(self._set_gpu)
        m_engine.addAction("Output Directory…", self._set_output_dir)

        m_help = self.menuBar().addMenu("&Help")
        m_help.addAction("About Megane", self._about)

    # -- engine settings -------------------------------------------------------
    def _refresh_engine_label(self) -> None:
        s = self.bridge.engine.settings
        dev = "GPU (CuPy)" if s.get("use_gpu") and backend.gpu_available() else "CPU (NumPy)"
        self._engine_label.setText(f"lens: {dev}   focus: {s.get('precision')}")
        self._engine_label.setToolTip("compute backend and float precision")

    def _set_precision(self, prec: str) -> None:
        self.bridge.engine.settings["precision"] = prec
        self._refresh_engine_label()
        self._schedule_auto_cook()

    def _set_gpu(self, enabled: bool) -> None:
        self.bridge.engine.settings["use_gpu"] = bool(enabled)
        self._refresh_engine_label()
        self._schedule_auto_cook()

    def _set_output_dir(self) -> None:
        sel = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory")
        if sel:
            self.bridge.engine.settings["output_dir"] = sel
            self.statusBar().showMessage(f"output directory: {sel}", 4000)

    # -- selection / params -------------------------------------------------------
    def _on_selection(self, node: Node | None) -> None:
        self._selected = node
        self.param_panel.show_node(node)

    def _on_param_edited(self, name: str, value: object) -> None:
        if self._selected is not None:
            self.bridge.set_param(self._selected.id, name, value)

    # -- cooking ---------------------------------------------------------------
    def _schedule_auto_cook(self) -> None:
        if self.auto_cook_act.isChecked():
            self._auto_timer.start()

    def _auto_cook_fire(self) -> None:
        target = self.bridge.preview_target(self._selected)
        if target is None:
            return
        if not self.bridge.chain_is_realtime(target):
            self.statusBar().showMessage(
                "heavy node in chain — press F5 to cook (bake)", 4000)
            return
        self._launch_cook(target)

    def cook_now(self) -> None:
        target = self.bridge.preview_target(self._selected)
        if target is None:
            self.statusBar().showMessage("nothing to cook — add nodes first", 4000)
            return
        self._launch_cook(target)

    def _launch_cook(self, target: str) -> None:
        self._generation += 1
        self.preview.status.setText("cooking…")
        task = _CookTask(self.bridge, target, self._generation, self._cook_signals)
        QtCore.QThreadPool.globalInstance().start(task)

    def _on_cook_done(self, generation: int, node_id: str, outputs: object) -> None:
        if generation != self._generation:
            return  # a newer cook superseded this one
        node = self.bridge.engine.nodes.get(node_id)
        label = f"{node.type_name} ({node_id})" if node else node_id
        self.preview.show_outputs(outputs, label)

    def _on_cook_error(self, generation: int, node_id: str, text: str) -> None:
        if generation != self._generation:
            return
        self.preview.show_error(text)
        self.statusBar().showMessage("cook failed", 4000)

    # -- file ops -----------------------------------------------------------------
    def _file_new(self) -> None:
        self.bridge.new()
        self._project_path = None
        self.param_panel.show_node(None)
        self.setWindowTitle("Megane 眼鏡 — image → sound lens")

    def _file_open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", "", "Megane projects (*.megane *.json)")
        if path:
            self.load_project(path)

    def load_project(self, path: str) -> None:
        try:
            self.bridge.load(path)
        except Exception as exc:  # noqa: BLE001 - surface bad files, don't crash
            QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._project_path = path
        self.param_panel.show_node(None)
        self._sync_settings_ui()
        self.setWindowTitle(f"Megane 眼鏡 — {os.path.basename(path)}")
        self.statusBar().showMessage(f"opened {path}", 4000)

    def _sync_settings_ui(self) -> None:
        prec = self.bridge.engine.settings.get("precision", "fp32")
        for act in self._prec_group.actions():
            act.setChecked(act.text() == prec)
        self.gpu_act.setChecked(bool(self.bridge.engine.settings.get("use_gpu")))
        self._refresh_engine_label()

    def _file_save(self) -> None:
        if self._project_path:
            self.bridge.save(self._project_path)
            self.statusBar().showMessage(f"saved {self._project_path}", 4000)
        else:
            self._file_save_as()

    def _file_save_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project", "untitled.megane", "Megane projects (*.megane)")
        if path:
            self._project_path = path
            self.bridge.save(path)
            self.setWindowTitle(f"Megane 眼鏡 — {os.path.basename(path)}")
            self.statusBar().showMessage(f"saved {path}", 4000)

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, "About Megane",
            "<b>Megane 眼鏡</b> — a lens for hearing images.<br>"
            "Faithful image→audio translation, node by node.<br><br>"
            "An Idolmancy tool.")
