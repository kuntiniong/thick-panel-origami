import importlib.util
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure stdout/stderr use UTF-8 on Windows when output is piped or when the
# system locale does not support CJK characters (e.g. cp1252 terminals).
# This is a no-op on platforms that already use UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.framework import (
    AlgorithmSpec,
    ThickPanelDesignFramework,
    get_algorithm_spec,
    register_algorithm,
    resolve_n_processes,
)


def _optimization_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _project_root() -> str:
    return os.path.dirname(_optimization_dir())


_CONFIG_HEADER = re.compile(
    r"(?m)^# (?:Optimization algorithm|Which optimizer to run)"
)
_ALGORITHM_LINE = re.compile(r"(?m)(?=^algorithm:\s)")
_YAML_DOC_SEPARATOR = re.compile(r"(?m)^---\s*$")


def _resolve_config_file() -> str:
    base_dir = _optimization_dir()
    config_path = os.path.join(base_dir, "config.yml")
    example_path = os.path.join(base_dir, "config.example.yml")

    if os.path.exists(config_path):
        return config_path
    if os.path.exists(example_path):
        print("config.yml not found. Falling back to config.example.yml")
        return example_path
    raise FileNotFoundError(
        "No optimization config found. Create optimization/config.yml "
        "from optimization/config.example.yml"
    )


def _split_config_chunks(text: str) -> List[str]:
    """Split pasted config blocks that repeat the same top-level YAML keys."""
    for pattern in (_CONFIG_HEADER, _ALGORITHM_LINE):
        positions = [match.start() for match in pattern.finditer(text)]
        if len(positions) > 1:
            chunks: List[str] = []
            for index, start in enumerate(positions):
                end = positions[index + 1] if index + 1 < len(positions) else len(text)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks
    return [text]


def _parse_config_text(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return [{}]

    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    if len(docs) > 1:
        return docs

    if _YAML_DOC_SEPARATOR.search(text):
        parts = [part.strip() for part in _YAML_DOC_SEPARATOR.split(text) if part.strip()]
        if len(parts) > 1:
            return [yaml.safe_load(part) or {} for part in parts]

    chunks = _split_config_chunks(text)
    if len(chunks) > 1:
        return [yaml.safe_load(chunk) or {} for chunk in chunks]

    if docs:
        return docs
    return [yaml.safe_load(text) or {}]


def load_configs(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load one or more optimization configs from YAML."""
    config_file_used = config_path or _resolve_config_file()
    with open(config_file_used, "r", encoding="utf-8") as f:
        return _parse_config_text(f.read())


def load_config() -> Dict[str, Any]:
    """Load the first optimization config from optimization/config.yml."""
    configs = load_configs()
    return configs[0] if configs else {}


def resolve_input_path(input_value: str) -> str:
    """Resolve descriptionData name or path to an existing JSON file."""
    project_root = _project_root()

    if os.path.isabs(input_value):
        json_path = input_value
    elif input_value.endswith(".json"):
        json_path = (
            input_value
            if os.path.exists(input_value)
            else os.path.join(project_root, input_value)
        )
    elif os.path.exists(input_value):
        json_path = input_value
    else:
        json_path = os.path.join(project_root, "descriptionData", f"{input_value}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input JSON not found: {json_path}")

    return json_path


def _framework_kwargs(config: Dict[str, Any], json_path: str) -> Dict[str, Any]:
    framework_cfg = config.get("framework", {})
    batch_size = framework_cfg.get("batch_size", 25)
    population_size = framework_cfg.get("population_size", batch_size)

    n_processes = resolve_n_processes(framework_cfg.get("n_processes"))

    initial_offsets_cfg = framework_cfg.get("initial_offsets")
    initial_offsets = (
        [float(value) for value in initial_offsets_cfg] if initial_offsets_cfg else None
    )

    kwargs = {
        "json_path": json_path,
        "batch_size": batch_size,
        "population_size": population_size,
        "n_processes": n_processes,
        "min_thickness": framework_cfg.get("min_thickness", 2.0),
        "discrete_step": framework_cfg.get("discrete_step", 0.4),
        "max_offset": framework_cfg.get("max_offset", 50.0),
        "use_gui": framework_cfg.get("use_gui", False),
        "symm_mode": framework_cfg.get("symm_mode", True),
    }
    if initial_offsets is not None:
        kwargs["initial_offsets"] = initial_offsets
    return kwargs


def _load_cma_framework_class():
    cma_path = os.path.join(_optimization_dir(), "algorithms", "cma-es.py")
    spec = importlib.util.spec_from_file_location("optimization_cma_es", cma_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ThickPanelCMAFramework


def _load_bo_framework_class():
    from optimization.algorithms.bo import ThickPanelBOFramework

    return ThickPanelBOFramework


def _load_cma_margin_framework_class():
    cma_margin_path = os.path.join(_optimization_dir(), "algorithms", "cma-es-margin.py")
    spec = importlib.util.spec_from_file_location("optimization_cma_es_margin", cma_margin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ThickPanelCMAMarginFramework


def _load_cma_elitist_margin_framework_class():
    path = os.path.join(_optimization_dir(), "algorithms", "cma-es-elitist-margin.py")
    spec = importlib.util.spec_from_file_location("optimization_cma_es_elitist_margin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ThickPanelCMAElitistMarginFramework


def _load_manual_framework_class():
    from optimization.manual import ThickPanelManualFramework

    return ThickPanelManualFramework


def _load_de_framework_class():
    from optimization.algorithms.de import ThickPanelDEFramework

    return ThickPanelDEFramework


def _common_optimize_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    opt_cfg = config.get("optimization", {})
    framework_cfg = config.get("framework", {})
    batch_size = framework_cfg.get("batch_size", 25)
    population_size = framework_cfg.get("population_size", batch_size)

    return {
        "population_size": population_size,
        "generations": opt_cfg.get("generations", 50),
        "verbose": opt_cfg.get("verbose", True),
    }


def _cma_es_optimize_kwargs(config: Dict[str, Any], common: Dict[str, Any]) -> Dict[str, Any]:
    cma_cfg = config.get("cma_es", {})
    return {**common, "sigma_init": cma_cfg.get("sigma_init", 5.0)}


def _cma_es_margin_optimize_kwargs(config: Dict[str, Any], common: Dict[str, Any]) -> Dict[str, Any]:
    cma_cfg = config.get("cma_es_margin", config.get("cma_es", {}))
    kwargs = {**common, "sigma_init": cma_cfg.get("sigma_init", 5.0)}
    if "margin" in cma_cfg:
        kwargs["margin"] = cma_cfg["margin"]
    return kwargs


def _cma_es_elitist_margin_optimize_kwargs(config: Dict[str, Any], common: Dict[str, Any]) -> Dict[str, Any]:
    cma_cfg = config.get("cma_es_elitist_margin", config.get("cma_es_margin", config.get("cma_es", {})))
    kwargs = {**common, "sigma_init": cma_cfg.get("sigma_init", 10.0)}
    if "margin" in cma_cfg:
        kwargs["margin"] = cma_cfg["margin"]
    if "enc_m" in cma_cfg:
        kwargs["enc_m"] = cma_cfg["enc_m"]
    return kwargs


def _bo_optimize_kwargs(config: Dict[str, Any], common: Dict[str, Any]) -> Dict[str, Any]:
    bo_cfg = config.get("bo", {})
    n_initial_points = bo_cfg.get("n_initial_points")
    if n_initial_points is None:
        n_initial_points = common["population_size"]

    return {
        **common,
        "n_initial_points": n_initial_points,
        "base_estimator": bo_cfg.get("base_estimator", "GP"),
        "acq_func": bo_cfg.get("acq_func", "EI"),
        "random_state": bo_cfg.get("random_state", config.get("random_seed", 42)),
    }


def _de_optimize_kwargs(config: Dict[str, Any], common: Dict[str, Any]) -> Dict[str, Any]:
    de_cfg = config.get("de", {})
    return {
        **common,
        "mutation_factor": de_cfg.get("mutation_factor", 0.8),
        "crossover_prob": de_cfg.get("crossover_prob", 0.9),
        "strategy": de_cfg.get("strategy", "rand1bin"),
        "random_state": de_cfg.get("random_state", config.get("random_seed", 42)),
    }


def _manual_optimize_kwargs(config: Dict[str, Any], _common: Dict[str, Any]) -> Dict[str, Any]:
    manual_cfg = config.get("manual", {})
    if "offsets" not in manual_cfg or not manual_cfg["offsets"]:
        raise ValueError("manual.offsets is required when algorithm: manual")

    return {
        "offsets": manual_cfg["offsets"],
        "verbose": config.get("optimization", {}).get("verbose", True),
    }


def _format_param_token(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if value == int(value):
            return str(int(value))
        return str(value).replace(".", "p")
    return str(value).replace(" ", "")


def _build_result_prefix(
    spec: AlgorithmSpec,
    optimize_kwargs: Dict[str, Any],
    framework_kwargs: Dict[str, Any],
) -> str:
    """Encode algorithm + framework settings in the physResult folder prefix."""
    parts = [spec.result_prefix]

    parts.extend(
        [
            f"pop{_format_param_token(framework_kwargs['population_size'])}",
            f"min{_format_param_token(framework_kwargs['min_thickness'])}",
            f"ds{_format_param_token(framework_kwargs['discrete_step'])}",
            f"mo{_format_param_token(framework_kwargs['max_offset'])}",
        ]
    )
    if "generations" in optimize_kwargs:
        parts.append(f"gen{_format_param_token(optimize_kwargs['generations'])}")

    if spec.key == "bo":
        parts.extend(
            [
                f"ninit{_format_param_token(optimize_kwargs['n_initial_points'])}",
                _format_param_token(optimize_kwargs["base_estimator"]),
                _format_param_token(optimize_kwargs["acq_func"]),
                f"rs{_format_param_token(optimize_kwargs['random_state'])}",
            ]
        )
    elif spec.key in ("cma_es", "cma_es_margin", "cma_es_elitist_margin"):
        parts.append(f"sigma{_format_param_token(optimize_kwargs['sigma_init'])}")
        if optimize_kwargs.get("margin") is not None:
            parts.append(f"margin{_format_param_token(optimize_kwargs['margin'])}")
        if spec.key == "cma_es_elitist_margin" and "enc_m" in optimize_kwargs:
            parts.append(
                f"encm{_format_param_token(1 if optimize_kwargs['enc_m'] else 0)}"
            )
    elif spec.key == "de":
        parts.extend(
            [
                f"f{_format_param_token(optimize_kwargs['mutation_factor'])}",
                f"cr{_format_param_token(optimize_kwargs['crossover_prob'])}",
                _format_param_token(optimize_kwargs["strategy"]),
            ]
        )

    return "-".join(parts)


def register_builtin_algorithms() -> None:
    register_algorithm(
        AlgorithmSpec(
            key="cma_es",
            aliases=("cma-es", "cma_es", "cmaes"),
            result_prefix="cma_es",
            config_key="cma_es",
            framework_loader=_load_cma_framework_class,
            optimize_kwargs_builder=_cma_es_optimize_kwargs,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            key="bo",
            aliases=("bo", "bayesian", "bayes"),
            result_prefix="bo",
            config_key="bo",
            framework_loader=_load_bo_framework_class,
            optimize_kwargs_builder=_bo_optimize_kwargs,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            key="cma_es_margin",
            aliases=("cma-es-margin", "cma_es_margin", "cmaeswm"),
            result_prefix="cma_es_margin",
            config_key="cma_es_margin",
            framework_loader=_load_cma_margin_framework_class,
            optimize_kwargs_builder=_cma_es_margin_optimize_kwargs,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            key="cma_es_elitist_margin",
            aliases=("cma-es-elitist-margin", "cma_es_elitist_margin", "cmaeswm_elitist"),
            result_prefix="cma_es_elitist_margin",
            config_key="cma_es_elitist_margin",
            framework_loader=_load_cma_elitist_margin_framework_class,
            optimize_kwargs_builder=_cma_es_elitist_margin_optimize_kwargs,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            key="manual",
            aliases=("manual",),
            result_prefix="manual",
            config_key="manual",
            framework_loader=_load_manual_framework_class,
            optimize_kwargs_builder=_manual_optimize_kwargs,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            key="de",
            aliases=("de", "differential-evolution", "differential_evolution"),
            result_prefix="de",
            config_key="de",
            framework_loader=_load_de_framework_class,
            optimize_kwargs_builder=_de_optimize_kwargs,
        )
    )


register_builtin_algorithms()


def create_framework(
    config: Dict[str, Any], json_path: str
) -> Tuple[ThickPanelDesignFramework, AlgorithmSpec, Dict[str, Any]]:
    spec = get_algorithm_spec(config.get("algorithm", "cma-es"))
    framework_kwargs = _framework_kwargs(config, json_path)
    optimize_kwargs = spec.optimize_kwargs_builder(config, _common_optimize_kwargs(config))
    framework_cls = spec.framework_loader()
    kwargs = dict(framework_kwargs)
    kwargs["algorithm_key"] = spec.key
    kwargs["result_prefix"] = _build_result_prefix(
        spec, optimize_kwargs, framework_kwargs
    )
    return framework_cls(**kwargs), spec, optimize_kwargs


def run_from_config(
    config: Optional[Dict[str, Any]] = None,
    default_algorithm: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    if config is None:
        config = load_config()

    config = dict(config)
    if default_algorithm is not None:
        config["algorithm"] = default_algorithm

    json_path = resolve_input_path(config["input"])
    framework, _spec, optimize_kwargs = create_framework(config, json_path)
    return framework.optimize(**optimize_kwargs)


def main(default_algorithm: Optional[str] = None) -> None:
    configs = load_configs()
    total_runs = len(configs)

    for run_index, config in enumerate(configs, start=1):
        config = dict(config)
        if default_algorithm is not None:
            config["algorithm"] = default_algorithm

        if total_runs > 1:
            print("=" * 60)
            print(f"Config run {run_index} of {total_runs}")
            print("=" * 60)

        print("=" * 60)
        print("厚板折纸高度偏移量优化 / Thick Panel Height Offset Optimization")
        print("=" * 60)
        print(f"Algorithm: {config.get('algorithm', 'cma-es')}")
        print(f"Input: {config.get('input')}")

        np.random.seed(config.get("random_seed", 42))
        run_from_config(config)


if __name__ == "__main__":
    main()