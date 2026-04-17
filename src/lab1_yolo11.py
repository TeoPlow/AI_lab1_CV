from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO
import yaml


def _results_dict_to_plain(results: Any) -> dict[str, float]:
    plain: dict[str, float] = {}
    for key, value in results.results_dict.items():
        try:
            plain[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return plain


def _resolve_data_yaml(data_yaml: str) -> str:
    data_path = Path(data_yaml).resolve()
    with data_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        return str(data_path)

    yaml_dir = data_path.parent
    cfg_path = cfg.get("path")

    if cfg_path is None:
        cfg["path"] = str(yaml_dir)
    else:
        base = Path(str(cfg_path))
        if not base.is_absolute():
            cwd_candidate = (Path.cwd() / base).resolve()
            if cwd_candidate.exists():
                cfg["path"] = str(cwd_candidate)
            else:
                cfg["path"] = str((yaml_dir / base).resolve())

    resolved_dir = Path("runs") / ".cache"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / f"resolved_{data_path.stem}.yaml"
    resolved_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return str(resolved_path)


def train(args: argparse.Namespace) -> None:
    resolved_data = _resolve_data_yaml(args.data)
    model = YOLO(args.model)
    train_results = model.train(
        data=resolved_data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        pretrained=True,
        verbose=True,
    )

    save_dir = getattr(train_results, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Не удалось определить папку запуска после обучения.")

    run_dir = Path(save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Лучший чекпоинт не найден: {best_weights}")

    best_model = YOLO(str(best_weights))
    test_results = best_model.val(
        data=resolved_data,
        split=args.test_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        verbose=True,
    )

    summary = {
        "train_run_dir": str(run_dir),
        "best_weights": str(best_weights),
        "test_split": args.test_split,
        "metrics": _results_dict_to_plain(test_results),
    }

    summary_path = run_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Обучение завершено.")
    print(f"Папка запуска: {run_dir}")
    print(f"Файл метрик JSON: {summary_path}")
    for key, value in summary["metrics"].items():
        print(f"{key}: {value:.6f}")


def evaluate(args: argparse.Namespace) -> None:
    resolved_data = _resolve_data_yaml(args.data)
    model = YOLO(args.weights)
    results = model.val(
        data=resolved_data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        verbose=True,
    )

    metrics = _results_dict_to_plain(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Метрики сохранены в: {output_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


def predict(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    print("Предсказание завершено.")
    print(f"Папка с результатами: {Path(args.project) / args.name}")


def experiments(args: argparse.Namespace) -> None:
    models = ["yolo11n.pt", "yolo11s.pt"] if args.models == "n_s" else ["yolo11n.pt"]

    for model_name in models:
        run_name = f"{Path(model_name).stem}_e{args.epochs}"
        local_args = argparse.Namespace(
            model=model_name,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=run_name,
            patience=args.patience,
            seed=args.seed,
            test_split=args.test_split,
        )
        print(f"Запуск эксперимента: {model_name}")
        train(local_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Пайплайн ЛР1 (CV) на Ultralytics YOLOv11 для датасета Hard Hat Workers"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--data", type=str, default="dataset/data.local.yaml")
    shared.add_argument("--imgsz", type=int, default=640)
    shared.add_argument("--batch", type=int, default=16)
    shared.add_argument("--device", type=str, default="cpu")
    shared.add_argument("--workers", type=int, default=4)

    p_train = subparsers.add_parser("train", parents=[shared])
    p_train.add_argument("--model", type=str, default="yolo11n.pt")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--project", type=str, default="runs/train")
    p_train.add_argument("--name", type=str, default="helmet_detector")
    p_train.add_argument("--patience", type=int, default=20)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--test-split", type=str, default="test")
    p_train.set_defaults(func=train)

    p_eval = subparsers.add_parser("eval", parents=[shared])
    p_eval.add_argument("--weights", type=str, required=True)
    p_eval.add_argument("--split", type=str, default="test")
    p_eval.add_argument("--output", type=str, default="runs/eval/metrics_test.json")
    p_eval.set_defaults(func=evaluate)

    p_predict = subparsers.add_parser("predict")
    p_predict.add_argument("--weights", type=str, required=True)
    p_predict.add_argument("--source", type=str, default="dataset/test/images")
    p_predict.add_argument("--conf", type=float, default=0.25)
    p_predict.add_argument("--iou", type=float, default=0.7)
    p_predict.add_argument("--imgsz", type=int, default=640)
    p_predict.add_argument("--device", type=str, default="cpu")
    p_predict.add_argument("--save-txt", action="store_true")
    p_predict.add_argument("--project", type=str, default="runs/predict")
    p_predict.add_argument("--name", type=str, default="helmet_detector")
    p_predict.set_defaults(func=predict)

    p_exp = subparsers.add_parser("experiments", parents=[shared])
    p_exp.add_argument("--models", choices=["n_only", "n_s"], default="n_s")
    p_exp.add_argument("--epochs", type=int, default=20)
    p_exp.add_argument("--project", type=str, default="runs/experiments")
    p_exp.add_argument("--patience", type=int, default=10)
    p_exp.add_argument("--seed", type=int, default=42)
    p_exp.add_argument("--test-split", type=str, default="test")
    p_exp.set_defaults(func=experiments)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
