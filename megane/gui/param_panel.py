"""Parameter panel: auto-generated editors from a node's ``Param`` specs.

Widget mapping
--------------
* choices == [True, False]      -> QCheckBox
* other fixed choices           -> QComboBox (original Python values kept)
* int default                   -> QSpinBox
* float default                 -> QDoubleSpinBox
* list default (e.g. notes)     -> QLineEdit, comma-separated numbers
* str named path/directory      -> QLineEdit + Browse button
* any other str                 -> QLineEdit

Every edit emits ``valueEdited(param_name, value)``; the main window forwards
it to the bridge (full user control, per the spec -- no hidden clamping).
"""
from __future__ import annotations

from typing import Any

from ..core.node import Node, Param
from .ng_compat import QtCore, QtWidgets


def _parse_number_list(text: str, fallback: list) -> list:
    items = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    out: list = []
    for it in items:
        try:
            out.append(int(it))
        except ValueError:
            try:
                out.append(float(it))
            except ValueError:
                return fallback
    return out or fallback


class ParamPanel(QtWidgets.QScrollArea):
    valueEdited = QtCore.Signal(str, object)  # (param_name, new_value)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(260)
        self._node: Node | None = None
        self._container = QtWidgets.QWidget()
        self.setWidget(self._container)
        self._form = QtWidgets.QFormLayout(self._container)
        self._form.setLabelAlignment(QtCore.Qt.AlignRight)
        self._title = QtWidgets.QLabel("no node selected")
        self._title.setStyleSheet("font-weight: bold;")
        self._form.addRow(self._title)

    # -- public ------------------------------------------------------------
    def show_node(self, node: Node | None) -> None:
        self._node = node
        # clear all rows below the title
        while self._form.rowCount() > 1:
            self._form.removeRow(1)
        if node is None:
            self._title.setText("no node selected")
            return
        self._title.setText(f"{node.type_name}  ({node.id})")
        if node.params:
            self._form.addRow("preset", self._preset_row(node))
        for spec in node.params:
            widget = self._make_widget(spec, node.values[spec.name])
            if spec.help:
                widget.setToolTip(spec.help)
            self._form.addRow(spec.name, widget)

    # -- presets -------------------------------------------------------------
    def _preset_row(self, node: Node) -> QtWidgets.QWidget:
        from ..core import presets

        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        combo = QtWidgets.QComboBox()
        combo.addItem("—")
        available = presets.list_presets(node.type_name)
        for name in sorted(available):
            combo.addItem(name)
        save_btn = QtWidgets.QToolButton()
        save_btn.setText("save…")
        save_btn.setToolTip("Save this node's current parameters as a preset")
        lay.addWidget(combo, 1)
        lay.addWidget(save_btn)

        def apply(index: int) -> None:
            name = combo.itemText(index)
            values = available.get(name)
            if not values or self._node is None:
                return
            for key, value in values.items():
                if key in self._node.values:
                    self._emit(key, value)
            # rebuild after the edits land so the widgets show the new values
            QtCore.QTimer.singleShot(0, lambda: self.show_node(self._node))

        def save() -> None:
            if self._node is None:
                return
            name, ok = QtWidgets.QInputDialog.getText(
                self, "Save preset", f"Preset name for {self._node.type_name}:")
            if ok and name.strip():
                presets.save_user_preset(self._node.type_name, name.strip(),
                                         dict(self._node.values))
                self.show_node(self._node)  # refresh the combo

        combo.activated.connect(apply)
        save_btn.clicked.connect(save)
        return box

    # -- widget factory -------------------------------------------------------
    def _emit(self, name: str, value: Any) -> None:
        if self._node is not None:
            self.valueEdited.emit(name, value)

    def _make_widget(self, spec: Param, current: Any) -> QtWidgets.QWidget:
        if spec.choices == [True, False]:
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(current))
            w.toggled.connect(lambda v, n=spec.name: self._emit(n, bool(v)))
            return w
        if spec.choices:
            w = QtWidgets.QComboBox()
            for choice in spec.choices:
                w.addItem(str(choice), choice)
            idx = w.findData(current)
            w.setCurrentIndex(idx if idx >= 0 else 0)
            w.currentIndexChanged.connect(
                lambda i, n=spec.name, cb=w: self._emit(n, cb.itemData(i)))
            return w
        if isinstance(spec.default, bool):
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(current))
            w.toggled.connect(lambda v, n=spec.name: self._emit(n, bool(v)))
            return w
        if isinstance(spec.default, int):
            w = QtWidgets.QSpinBox()
            w.setRange(-2_147_483_647, 2_147_483_647)
            w.setValue(int(current))
            w.valueChanged.connect(lambda v, n=spec.name: self._emit(n, int(v)))
            return w
        if isinstance(spec.default, float):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            w.setDecimals(4)
            w.setSingleStep(0.1)
            w.setValue(float(current))
            w.valueChanged.connect(lambda v, n=spec.name: self._emit(n, float(v)))
            return w
        if isinstance(spec.default, list):
            w = QtWidgets.QLineEdit(", ".join(str(v) for v in current))
            w.editingFinished.connect(
                lambda n=spec.name, le=w: self._emit(
                    n, _parse_number_list(le.text(), list(spec.default))))
            return w
        if spec.name in ("path", "directory"):
            return self._path_widget(spec, str(current))
        w = QtWidgets.QLineEdit(str(current))
        w.editingFinished.connect(
            lambda n=spec.name, le=w: self._emit(n, le.text()))
        return w

    def _path_widget(self, spec: Param, current: str) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit(current)
        btn = QtWidgets.QToolButton()
        btn.setText("…")
        lay.addWidget(edit, 1)
        lay.addWidget(btn)

        def browse() -> None:
            if spec.name == "directory":
                sel = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose directory")
            else:
                sel, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose file")
            if sel:
                edit.setText(sel)
                self._emit(spec.name, sel)

        btn.clicked.connect(browse)
        edit.editingFinished.connect(
            lambda n=spec.name, le=edit: self._emit(n, le.text()))
        return box
