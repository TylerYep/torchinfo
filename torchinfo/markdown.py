"""Optional eager execution recording and dependency-free Markdown rendering."""

from __future__ import annotations

import json
import tempfile
import textwrap
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from .enums import ColumnSettings
from .formatting import HEADER_TITLES

if TYPE_CHECKING:
    import os

    from .layer_info import LayerInfo
    from .model_statistics import ModelStatistics


def tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


@dataclass
class Node:
    name: str
    kind: str
    groups: tuple[int, ...] = ()
    edge_shapes: dict[int, list[str]] = field(default_factory=dict)


class GraphRecorder(TorchDispatchMode):
    """Record one eager run; hooks collapse module internals at the requested depth."""

    def __init__(self, depth: int) -> None:
        cast("Callable[[], None]", super().__init__)()
        self.depth = depth
        self.nodes: list[Node] = []
        self.rows: dict[int, int] = {}
        self.groups: list[tuple[str, str]] = []
        self.group_path: tuple[int, ...] = ()
        self.group_stack: list[tuple[int, ...]] = []
        self.row_names: dict[int, str] = {}
        self.producers: dict[int, tuple[weakref.ReferenceType[torch.Tensor], int]] = {}
        self.aliases: dict[int, tuple[weakref.ReferenceType[Any], int]] = {}
        self.state: set[int] = set()
        self.stack: list[int | None] = []
        self.active: int | None = None
        self.versions: list[list[tuple[torch.Tensor, int]]] = []

    def add(self, name: str, kind: str, inputs: Any) -> int:
        node = Node(name, kind, groups=self.group_path)
        for tensor in tensors(inputs):
            producer = self.producer(tensor)
            if producer is not None:
                node.edge_shapes.setdefault(producer, []).append(
                    str(list(tensor.shape))
                )
        self.nodes.append(node)
        return len(self.nodes) - 1

    def producer(self, tensor: torch.Tensor) -> int | None:
        if id(tensor) in self.state:
            return None
        storage = tensor.untyped_storage()
        alias = self.aliases.get(storage._cdata)
        entry = self.producers.get(id(tensor))
        candidates = []
        if alias is not None and alias[0]() is storage:
            candidates.append(alias[1])
        if entry is not None and entry[0]() is tensor:
            candidates.append(entry[1])
        if candidates:
            return max(candidates)
        raise RuntimeError(
            "Markdown capture encountered an untracked tensor; "
            "pass it as an input or register it as model state."
        )

    def assign(self, outputs: Any, node: int, *, mutation: bool = False) -> None:
        for tensor in tensors(outputs):
            self.producers[id(tensor)] = (weakref.ref(tensor), node)
            storage = tensor.untyped_storage()
            if mutation:
                self.aliases[storage._cdata] = (weakref.ref(storage), node)

    def __torch_dispatch__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        kwargs = kwargs or {}
        if self.active is not None:
            return func(*args, **kwargs)
        node = self.add(str(func), "operation", (args, kwargs))
        output = func(*args, **kwargs)
        mutation = any(
            arg.alias_info is not None and arg.alias_info.is_write
            for arg in func._schema.arguments
        )
        self.assign(output, node, mutation=mutation)
        return output

    @contextmanager
    def capture(
        self, model: nn.Module, lookup: dict[int, LayerInfo], inputs: Any, kwargs: Any
    ) -> Iterator[None]:
        if any(
            isinstance(m, torch.jit.ScriptModule) or hasattr(m, "_orig_mod")
            for m in model.modules()
        ):
            raise ValueError("Markdown capture requires an eager, uncompiled model.")
        self.state = {id(t) for t in (*model.parameters(), *model.buffers())}
        handles = []

        def before(module: nn.Module, args: Any, kw: Any) -> None:
            self.stack.append(self.active)
            self.versions.append([(t, t._version) for t in tensors((args, kw))])
            info = lookup[id(module)]
            self.group_stack.append(self.group_path)
            self.row_names[id(info)] = ".".join(
                [self.groups[g][0] for g in self.group_path[1:]] + [info.var_name]
            )
            if self.active is None and (
                (
                    info.is_leaf_layer
                    and (
                        info.depth > 0
                        or any(
                            cls is not nn.Module
                            and cls.__module__.startswith("torch.nn")
                            for cls in type(module).__mro__
                        )
                    )
                )
                or info.depth >= self.depth
            ):
                self.active = self.add(
                    f"{info.var_name} ({info.class_name})", "module", (args, kw)
                )
                self.rows[id(info)] = self.active
            elif self.active is None:
                group = len(self.groups)
                self.groups.append((info.var_name, info.class_name))
                self.group_path = (*self.group_path, group)

        def after(module: nn.Module, args: Any, kw: Any, output: Any) -> None:
            del module, args, kw
            self.group_path = self.group_stack.pop()
            previous = self.stack.pop()
            versions = self.versions.pop()
            if self.active is not None and previous is None:
                for tensor, version in versions:
                    if tensor._version != version:
                        self.assign(tensor, self.active, mutation=True)
                self.assign(output, self.active)
            self.active = previous

        try:
            for index, tensor in enumerate(tensors((inputs, kwargs))):
                node = len(self.nodes)
                self.nodes.append(Node(f"Input {index + 1}", "input"))
                self.assign(tensor, node)
            for module in model.modules():
                handles.append(
                    module.register_forward_pre_hook(before, with_kwargs=True)
                )
                handles.append(module.register_forward_hook(after, with_kwargs=True))
            with self:
                yield
        finally:
            for handle in handles:
                handle.remove()
            self.producers.clear()
            self.aliases.clear()
            self.state.clear()
            self.stack.clear()
            self.group_stack.clear()
            self.group_path = ()
            self.versions.clear()
            self.active = None

    def finish(self, output: Any) -> None:
        for index, tensor in enumerate(tensors(output)):
            self.add(f"Output {index + 1}", "output", tensor)


def escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("`", "&#96;")
        .replace("\n", " ")
        .replace("\r", " ")
    )


# Classic Mermaid syntax works in viewers that predate the newer shape API.
BOX_SHAPES = {
    "hexagon": ('{{"', '"}}'),
    "stadium": ('(["', '"])'),
    "rectangle": ('["', '"]'),
    "rounded": ('("', '")'),
    "subroutine": ('[["', '"]]'),
    "parallelogram": ('[/"', '"/]'),
    "trapezoid": ('[/"', '"\\]'),
    "cylinder": ('[("', '")]'),
}


def load_layer_styles() -> dict[str, Any]:
    """Read the shipped type-to-style registry, including from installed wheels."""
    return cast(
        "dict[str, Any]",
        json.loads(
            files("torchinfo").joinpath("layer_styles.json").read_text(encoding="utf-8")
        ),
    )


def layer_style(module: nn.Module, registry: dict[str, Any]) -> str:
    """Use the actual Python type and its bases, independent of display names."""
    for cls in type(module).__mro__:
        if cls.__name__ in registry["types"]:
            return str(registry["types"][cls.__name__])
    return str(registry["default"])


def render_markdown(
    stats: ModelStatistics,
    graph: GraphRecorder,
    columns: tuple[ColumnSettings, ...] | None,
) -> str:
    explicit_columns = columns is not None
    columns = (
        columns
        if columns is not None
        else (
            ColumnSettings.INPUT_SIZE,
            ColumnSettings.OUTPUT_SIZE,
            ColumnSettings.NUM_PARAMS,
            ColumnSettings.MULT_ADDS,
            ColumnSettings.TRAINABLE,
        )
    )
    registry = load_layer_styles()
    styles = registry["styles"]
    config = {
        "theme": "base",
        "htmlLabels": False,
        "flowchart": {
            "htmlLabels": False,
            "padding": 24,
            "rankSpacing": 70,
            "subGraphTitleMargin": {"top": 12, "bottom": 24},
        },
        "themeVariables": {
            "fontFamily": "Arial",
            "fontSize": "14px",
            "lineColor": "#000000",
            "textColor": "#000000",
            "primaryTextColor": "#000000",
            "titleColor": "#000000",
            "edgeLabelBackground": "#ffffff",
        },
        "themeCSS": (
            ".flowchart-link {stroke:#000000!important;stroke-width:2.5px!important;}"
            "marker path {fill:#000000!important;stroke:#000000!important;}"
            ".cluster-label text,.cluster-label span,.cluster-label tspan {"
            "fill:#000000!important;"
            "color:#000000!important;font-weight:700!important;}"
            ".edgeLabel text {fill:#000000!important;}"
        ),
    }
    lines = [
        "# Model summary",
        "",
        (
            "This graph shows tensor data flow for the supplied input "
            "and executed path only."
        ),
        "",
        "```mermaid",
        "%%{init: " + json.dumps(config) + "}%%",
        "flowchart TD",
    ]
    if not graph.groups:
        lines.extend(['    subgraph canvas[" "]', "    direction TB"])
    info_by_node = {
        node_index: info
        for info in stats.summary_list
        if (node_index := graph.rows.get(id(info))) is not None
    }

    def metrics(info: LayerInfo) -> list[str]:
        if not explicit_columns:
            return []
        values = stats.formatting.row_values(
            info, info.depth == stats.formatting.max_depth, stats.total_params
        )
        return [
            f"{HEADER_TITLES[col]}: {values[col]}"
            for col in columns
            if col not in {ColumnSettings.INPUT_SIZE, ColumnSettings.OUTPUT_SIZE}
        ]

    used_styles: set[str] = set()
    group_depths: dict[int, int] = {}
    opened: tuple[int, ...] = ()
    for index, node in enumerate(graph.nodes):
        node_groups = node.groups
        if graph.groups and node.kind in {"input", "output"}:
            node_groups = (0,)
        common = 0
        while (
            common < min(len(opened), len(node_groups))
            and opened[common] == node_groups[common]
        ):
            common += 1
        lines.extend("    end" for _ in opened[common:])
        for level, group in enumerate(node_groups[common:], start=common):
            group_depths[group] = level
            name, class_name = graph.groups[group]
            group_label = escape(f"{name} ({class_name})")
            lines.append(f'    subgraph g{group}["{group_label}"]')
            lines.append("    direction TB")
        opened = node_groups
        style_name = (
            layer_style(info_by_node[index].module, registry)
            if index in info_by_node
            else node.kind
        )
        used_styles.add(style_name)
        left, right = BOX_SHAPES[styles[style_name]["shape"]]
        label_lines = [node.name]
        if index in info_by_node:
            label_lines.extend(metrics(info_by_node[index]))
        label = "<br/>".join(
            escape(part)
            for line in label_lines
            for part in textwrap.wrap(line, width=38)
        )
        lines.append(f"    n{index}{left}{label}{right}:::{style_name}")
    lines.extend("    end" for _ in opened)
    for index, node in enumerate(graph.nodes):
        for source in sorted(node.edge_shapes):
            if source != index:
                edge_label = "<br/>".join(
                    escape(shape) for shape in node.edge_shapes[source]
                )
                lines.append(f'    n{source} -->|"{edge_label}"| n{index}')
    if not graph.groups:
        lines.append("    end")
        lines.append("    style canvas fill:#ffffff,stroke:none")
    palette = ("#64748b", "#60a5fa", "#a78bfa", "#5eead4")
    for group, level in group_depths.items():
        stroke = palette[level % len(palette)]
        fill = "#ffffff" if group == 0 else "none"
        lines.append(
            f"    style g{group} fill:{fill},stroke:{stroke},"
            "stroke-width:2px,color:#000000"
        )
    lines.append(
        "    linkStyle default stroke:#000000,stroke-width:2.5px,color:#000000"
    )
    for name in sorted(used_styles):
        style = styles[name]
        lines.append(
            f"    classDef {name} fill:{style['fill']},stroke:{style['stroke']},"
            "stroke-width:2px,color:#000000"
        )
    lines.extend(
        [
            "```",
            "",
            (
                "Arrow labels show tensor shapes. Black arrows indicate data flow. "
                "Colored subgraph borders indicate nesting depth; "
                "headings identify the module name and type. "
                "Layer shapes and colors follow the bundled layer_styles.json registry."
            ),
            "",
            "## Layer statistics",
            "",
            "| Layer | " + " | ".join(HEADER_TITLES[c] for c in columns) + " |",
            "| --- | " + " | ".join("---" for _ in columns) + " |",
        ]
    )
    fmt = stats.formatting
    for info in stats.summary_list:
        if info.depth > fmt.max_depth or (
            fmt.hide_recursive_layers and info.is_recursive
        ):
            continue
        values = fmt.row_values(info, info.depth == fmt.max_depth, stats.total_params)
        cells = [
            (
                f"{graph.row_names[id(info)]} ({info.class_name})"
                if id(info) in graph.row_names
                else info.get_layer_name(True, True)
            ),
            *(values[c] for c in columns),
        ]
        lines.append("| " + " | ".join(escape(cell) for cell in cells) + " |")
    lines.extend(
        [
            "",
            (
                "Functional-operation MACs: — (not estimated). Totals retain "
                "torchinfo's existing module-based MAC coverage."
            ),
            "",
            "## Totals",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ]
    )
    # Reuse the exact existing total formatting and unit conventions.
    for line in repr(stats).splitlines():
        if line.startswith(
            (
                "Total params",
                "Trainable params",
                "Non-trainable params",
                "Total mult-adds",
                "Input size",
                "Forward/backward pass size",
                "Params size",
                "Estimated Total Size",
            )
        ):
            key, value = line.split(":", 1)
            lines.append(f"| {key} | {value.strip()} |")
    return "\n".join(lines) + "\n"


def write_markdown(
    path: str | os.PathLike[str],
    stats: ModelStatistics,
    graph: GraphRecorder,
    columns: tuple[ColumnSettings, ...] | None,
) -> None:
    content = render_markdown(stats, graph, columns)
    destination = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(content)
        Path(temporary).replace(destination)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
