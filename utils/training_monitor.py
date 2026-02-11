"""
학습 진행도 고급 모니터링 및 시각화 (Rich + Seaborn)

이 모듈은 Fashion JSON Encoder 학습 과정을 Rich와 Seaborn을 사용하여
실시간으로 모니터링하고 아름답게 시각화하는 기능을 제공합니다.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from tqdm import tqdm

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")

# 한글 폰트 설정
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Rich console
console = Console()


class TrainingMonitor:
    """학습 진행도 고급 모니터링 클래스 (Rich + Seaborn 기반)"""

    def __init__(self, results_dir: str = "results", update_chart_every: int = 5):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.update_chart_every = update_chart_every

        # 학습 상태 추적
        self.training_state = {
            "stage": "Stage 1",
            "current_epoch": 0,
            "total_epochs": 0,
            "start_time": None,
            "is_training": False,
            "epoch_start_time": None,
            "epoch_durations": [],
        }

        # 메트릭 저장
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "top1_accuracy": [],
            "top5_accuracy": [],
            "mrr": [],
            "positive_similarity": [],
            "negative_similarity": [],
            "l2_norm": [],
            "epochs": [],
        }

        # Best 메트릭 추적
        self.best_metrics = {
            "epoch": 0,
            "val_loss": float("inf"),
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "mrr": 0.0,
        }

        # Rich Progress
        self.progress = None
        self.task_id = None
        self.epoch_pbar = None

    def start_training(self, stage: str, total_epochs: int):
        """학습 시작"""
        self.training_state.update(
            {
                "stage": stage,
                "current_epoch": 0,
                "total_epochs": total_epochs,
                "start_time": datetime.now(),
                "is_training": True,
                "epoch_durations": [],
            }
        )

        # Rich Panel로 시작 메시지
        console.print(
            Panel.fit(
                f"[bold cyan]{stage}[/bold cyan] 학습 시작\n"
                f"[yellow]총 에포크:[/yellow] {total_epochs}\n"
                f"[green]시작 시간:[/green] {self.training_state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}",
                title="🚀 Training Started",
                border_style="cyan",
            )
        )

        # tqdm 진행 바 초기화 (fallback)
        self.epoch_pbar = tqdm(
            total=total_epochs,
            desc=f"{stage}",
            unit="epoch",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    def update_epoch(self, epoch: int, metrics: Dict[str, float]):
        """에포크 업데이트"""
        # 에포크 시작 시간 기록
        if self.training_state["epoch_start_time"] is None:
            self.training_state["epoch_start_time"] = time.time()

        # 에포크 소요 시간 계산
        epoch_duration = time.time() - self.training_state["epoch_start_time"]
        self.training_state["epoch_durations"].append(epoch_duration)
        self.training_state["current_epoch"] = epoch

        # 다음 에포크 시작 시간 리셋
        self.training_state["epoch_start_time"] = time.time()

        # 메트릭 히스토리 업데이트
        self.metrics_history["epochs"].append(epoch)
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        # Best 메트릭 업데이트
        is_best = False
        if "val_loss" in metrics and metrics["val_loss"] < self.best_metrics["val_loss"]:
            self.best_metrics["val_loss"] = metrics["val_loss"]
            self.best_metrics["epoch"] = epoch
            is_best = True

        if "top5_accuracy" in metrics and metrics["top5_accuracy"] > self.best_metrics["top5_accuracy"]:
            self.best_metrics["top5_accuracy"] = metrics["top5_accuracy"]
            if not is_best:
                self.best_metrics["epoch"] = epoch
                is_best = True

        if "top1_accuracy" in metrics:
            self.best_metrics["top1_accuracy"] = max(
                self.best_metrics["top1_accuracy"], metrics["top1_accuracy"]
            )

        if "mrr" in metrics:
            self.best_metrics["mrr"] = max(self.best_metrics["mrr"], metrics["mrr"])

        # Rich 테이블로 메트릭 표시
        self._display_metrics_table(epoch, metrics, is_best)

        # tqdm 진행 바 업데이트
        if self.epoch_pbar:
            postfix_dict = {}
            if "val_loss" in metrics:
                postfix_dict["val_loss"] = f"{metrics['val_loss']:.4f}"
            if "top5_accuracy" in metrics:
                postfix_dict["top5"] = f"{metrics['top5_accuracy']:.2%}"
            self.epoch_pbar.set_postfix(postfix_dict)
            self.epoch_pbar.update(1)

        # 주기적으로 차트 업데이트 (best 모델이거나 update_chart_every 주기)
        if is_best or epoch % self.update_chart_every == 0:
            self.create_enhanced_charts()

    def _display_metrics_table(self, epoch: int, metrics: Dict[str, float], is_best: bool):
        """Rich 테이블로 메트릭 표시"""
        table = Table(title=f"Epoch {epoch} Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Current", justify="right", style="yellow", width=12)
        table.add_column("Best", justify="right", style="green", width=12)
        table.add_column("Status", justify="center", width=8)

        # ETA 계산
        eta_str = self._calculate_eta()

        # Val Loss
        if "val_loss" in metrics:
            status = "🟢 NEW" if is_best and metrics["val_loss"] == self.best_metrics["val_loss"] else "⚪"
            table.add_row(
                "Val Loss",
                f"{metrics['val_loss']:.4f}",
                f"{self.best_metrics['val_loss']:.4f}",
                status,
            )

        # Top-1 Accuracy
        if "top1_accuracy" in metrics:
            status = "🟢" if metrics["top1_accuracy"] >= self.best_metrics["top1_accuracy"] else "⚪"
            table.add_row(
                "Top-1 Accuracy",
                f"{metrics['top1_accuracy']:.2%}",
                f"{self.best_metrics['top1_accuracy']:.2%}",
                status,
            )

        # Top-5 Accuracy
        if "top5_accuracy" in metrics:
            status = "🟢" if metrics["top5_accuracy"] >= self.best_metrics["top5_accuracy"] else "⚪"
            table.add_row(
                "Top-5 Accuracy",
                f"{metrics['top5_accuracy']:.2%}",
                f"{self.best_metrics['top5_accuracy']:.2%}",
                status,
            )

        # MRR
        if "mrr" in metrics:
            status = "🟢" if metrics["mrr"] >= self.best_metrics["mrr"] else "⚪"
            table.add_row(
                "MRR",
                f"{metrics['mrr']:.4f}",
                f"{self.best_metrics['mrr']:.4f}",
                status,
            )

        table.add_row("", "", "", "")
        table.add_row("ETA", eta_str, f"Best: Epoch {self.best_metrics['epoch']}", "⏰")

        console.print(table)

    def _calculate_eta(self) -> str:
        """예상 완료 시간 계산"""
        if not self.training_state["epoch_durations"]:
            return "계산 중..."

        avg_epoch_time = sum(self.training_state["epoch_durations"]) / len(
            self.training_state["epoch_durations"]
        )
        remaining_epochs = (
            self.training_state["total_epochs"] - self.training_state["current_epoch"]
        )
        eta_seconds = avg_epoch_time * remaining_epochs

        eta_timedelta = timedelta(seconds=int(eta_seconds))
        eta_time = datetime.now() + eta_timedelta

        return f"{str(eta_timedelta).split('.')[0]} (완료: {eta_time.strftime('%H:%M:%S')})"

    def save_checkpoint(self, checkpoint_path: str):
        """체크포인트 저장 알림"""
        console.print(f"💾 [green]체크포인트 저장:[/green] {checkpoint_path}")

    def finish_training(self):
        """학습 완료"""
        self.training_state["is_training"] = False
        end_time = datetime.now()
        duration = end_time - self.training_state["start_time"]

        # tqdm 진행 바 종료
        if self.epoch_pbar:
            self.epoch_pbar.close()

        # 최종 결과 표시
        console.print(
            Panel.fit(
                f"[bold green]{self.training_state['stage']}[/bold green] 학습 완료!\n"
                f"[yellow]총 소요 시간:[/yellow] {duration}\n"
                f"[cyan]Best Epoch:[/cyan] {self.best_metrics['epoch']}\n"
                f"[magenta]Best Top-5:[/magenta] {self.best_metrics['top5_accuracy']:.2%}",
                title="✅ Training Completed",
                border_style="green",
            )
        )

        # 최종 결과 저장 및 시각화
        self.save_training_summary()
        self.create_enhanced_charts()

    def create_enhanced_charts(self):
        """Seaborn 스타일의 향상된 차트 생성"""
        if not self.metrics_history["epochs"]:
            return

        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Fashion JSON Encoder - {self.training_state["stage"]} (Best: Epoch {self.best_metrics["epoch"]})',
            fontsize=16,
            fontweight="bold",
        )

        epochs = self.metrics_history["epochs"]
        best_epoch = self.best_metrics["epoch"]

        # 1. 손실 곡선
        if self.metrics_history["train_loss"] or self.metrics_history["val_loss"]:
            ax = axes[0, 0]
            if self.metrics_history["train_loss"]:
                sns.lineplot(
                    x=epochs,
                    y=self.metrics_history["train_loss"],
                    ax=ax,
                    label="Train Loss",
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                )
            if self.metrics_history["val_loss"]:
                sns.lineplot(
                    x=epochs,
                    y=self.metrics_history["val_loss"],
                    ax=ax,
                    label="Val Loss",
                    linewidth=2.5,
                    marker="s",
                    markersize=4,
                )
            # Best epoch 표시
            if best_epoch in epochs:
                ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label="Best")
            ax.set_title("Loss Curve", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # 2. Top-K 정확도
        if self.metrics_history["top1_accuracy"] or self.metrics_history["top5_accuracy"]:
            ax = axes[0, 1]
            if self.metrics_history["top1_accuracy"]:
                sns.lineplot(
                    x=epochs,
                    y=self.metrics_history["top1_accuracy"],
                    ax=ax,
                    label="Top-1",
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                )
            if self.metrics_history["top5_accuracy"]:
                sns.lineplot(
                    x=epochs,
                    y=self.metrics_history["top5_accuracy"],
                    ax=ax,
                    label="Top-5",
                    linewidth=2.5,
                    marker="s",
                    markersize=4,
                )
            # Best epoch 표시
            if best_epoch in epochs:
                ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label="Best")
            ax.set_title("Top-K Accuracy", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # 3. MRR
        if self.metrics_history["mrr"]:
            ax = axes[1, 0]
            sns.lineplot(
                x=epochs,
                y=self.metrics_history["mrr"],
                ax=ax,
                label="MRR",
                linewidth=2.5,
                marker="D",
                markersize=4,
                color="purple",
            )
            # Best epoch 표시
            if best_epoch in epochs:
                ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label="Best")
            ax.set_title("Mean Reciprocal Rank", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("MRR", fontsize=10)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # 4. 유사도
        if self.metrics_history["positive_similarity"] or self.metrics_history["negative_similarity"]:
            ax = axes[1, 1]
            if self.metrics_history["positive_similarity"]:
                sns.lineplot(
                    x=epochs,
                    y=self.metrics_history["positive_similarity"],
                    ax=ax,
                    label="Positive",
                    linewidth=2.5,
                    marker="^",
                    markersize=4,
                )
            if self.metrics_history["negative_similarity"]:
                sns.lineplot(
                    x=epochs,
                    y=self.metrics_history["negative_similarity"],
                    ax=ax,
                    label="Negative",
                    linewidth=2.5,
                    marker="v",
                    markersize=4,
                )
            # Best epoch 표시
            if best_epoch in epochs:
                ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label="Best")
            ax.set_title("Similarity", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Similarity", fontsize=10)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = self.results_dir / "training_charts.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close()

        console.print(f"📈 [cyan]학습 차트 저장:[/cyan] {chart_path}")

    def save_training_summary(self):
        """학습 요약 저장"""
        summary = {
            "training_state": self.training_state.copy(),
            "metrics_history": self.metrics_history.copy(),
            "best_metrics": self.best_metrics.copy(),
            "timestamp": datetime.now().isoformat(),
        }

        # datetime 객체를 문자열로 변환
        if summary["training_state"]["start_time"]:
            summary["training_state"]["start_time"] = summary["training_state"][
                "start_time"
            ].isoformat()

        summary_path = self.results_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        console.print(f"📄 [cyan]학습 요약 저장:[/cyan] {summary_path}")


class DashboardDataProvider:
    """대시보드용 데이터 제공 클래스"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드용 데이터 반환"""
        summary_path = self.results_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            summary = {}

        kpi_data = self._extract_kpi_data(summary)
        chart_data = self._extract_chart_data(summary)
        search_data = self._extract_search_data()

        return {
            "kpi": kpi_data,
            "charts": chart_data,
            "search_results": search_data,
            "timestamp": datetime.now().isoformat(),
        }

    def _extract_kpi_data(self, summary: Dict) -> Dict[str, Any]:
        """KPI 카드 데이터 추출"""
        metrics = summary.get("metrics_history", {})
        state = summary.get("training_state", {})
        best = summary.get("best_metrics", {})

        return {
            "current_epoch": state.get("current_epoch", 0),
            "total_epochs": state.get("total_epochs", 0),
            "stage": state.get("stage", "N/A"),
            "best_epoch": best.get("epoch", 0),
            "top1_accuracy": (
                metrics.get("top1_accuracy", [])[-1] if metrics.get("top1_accuracy") else 0
            ),
            "top5_accuracy": (
                metrics.get("top5_accuracy", [])[-1] if metrics.get("top5_accuracy") else 0
            ),
            "best_top5_accuracy": best.get("top5_accuracy", 0),
            "mrr": metrics.get("mrr", [])[-1] if metrics.get("mrr") else 0,
            "best_mrr": best.get("mrr", 0),
            "val_loss": metrics.get("val_loss", [])[-1] if metrics.get("val_loss") else 0,
            "best_val_loss": best.get("val_loss", float("inf")),
        }

    def _extract_chart_data(self, summary: Dict) -> Dict[str, Any]:
        """차트 데이터 추출"""
        metrics = summary.get("metrics_history", {})

        return {
            "loss_curve": {
                "epochs": metrics.get("epochs", []),
                "train_loss": metrics.get("train_loss", []),
                "val_loss": metrics.get("val_loss", []),
            },
            "accuracy_curve": {
                "epochs": metrics.get("epochs", []),
                "top1_accuracy": metrics.get("top1_accuracy", []),
                "top5_accuracy": metrics.get("top5_accuracy", []),
            },
            "similarity_curve": {
                "epochs": metrics.get("epochs", []),
                "positive_similarity": metrics.get("positive_similarity", []),
                "negative_similarity": metrics.get("negative_similarity", []),
            },
        }

    def _extract_search_data(self) -> Dict[str, Any]:
        """검색 결과 데이터 추출"""
        search_dir = self.results_dir / "similarity_search"
        if not search_dir.exists():
            return {}

        summary_file = search_dir / "similarity_search_summary.md"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_text = f.read()
        else:
            summary_text = ""

        image_files = list(search_dir.glob("*.png"))

        return {
            "summary": summary_text,
            "result_images": [str(img.name) for img in image_files],
            "total_queries": len(image_files),
        }
