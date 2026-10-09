import os
import json
import time
from datetime import datetime
from accelerate import Accelerator
import torch
from .logger import ModelLogger


class EnhancedModelLogger(ModelLogger):
    """增强的模型日志记录器，支持记录loss、学习率等训练指标"""

    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x,
                 log_interval=10):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.log_interval = log_interval  # 每隔多少步记录一次
        self.metrics_log_file = os.path.join(output_path, "training_metrics.jsonl")
        self.summary_log_file = os.path.join(output_path, "training_summary.txt")

        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)

        # 初始化指标存储
        self.current_epoch = 0
        self.global_step = 0
        self.epoch_losses = []
        self.start_time = time.time()

        # 写入日志头部
        with open(self.summary_log_file, 'w') as f:
            f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log_metrics(self, accelerator: Accelerator, metrics: dict):
        """记录训练指标到文件"""
        if not accelerator.is_main_process:
            return

        # 添加时间戳和全局步数
        metrics_with_meta = {
            'timestamp': datetime.now().isoformat(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            **metrics
        }

        # 写入JSONL格式的日志文件（每行一个JSON对象）
        with open(self.metrics_log_file, 'a') as f:
            f.write(json.dumps(metrics_with_meta) + '\n')

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module,
                    loss: float = None, learning_rate: float = None, save_steps=None):
        """在每个训练步结束时调用"""
        self.num_steps += 1
        self.global_step += 1

        # 记录loss
        if loss is not None:
            self.epoch_losses.append(float(loss))

            # 每隔log_interval步记录一次指标
            if self.global_step % self.log_interval == 0:
                metrics = {
                    'loss': float(loss),
                    'avg_loss': sum(self.epoch_losses[-self.log_interval:]) / min(len(self.epoch_losses), self.log_interval)
                }

                if learning_rate is not None:
                    metrics['learning_rate'] = float(learning_rate)

                # 计算吞吐量
                elapsed_time = time.time() - self.start_time
                samples_per_sec = self.global_step / elapsed_time if elapsed_time > 0 else 0
                metrics['samples_per_sec'] = samples_per_sec

                self.log_metrics(accelerator, metrics)

                # 打印到控制台
                if accelerator.is_main_process:
                    log_str = f"[Epoch {self.current_epoch} Step {self.global_step}] "
                    log_str += f"Loss: {loss:.6f} | Avg Loss: {metrics['avg_loss']:.6f}"
                    if learning_rate is not None:
                        log_str += f" | LR: {learning_rate:.2e}"
                    log_str += f" | Speed: {samples_per_sec:.2f} samples/s"
                    print(log_str)

        # 保存checkpoint
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_start(self, epoch_id: int):
        """在epoch开始时调用"""
        self.current_epoch = epoch_id
        self.epoch_losses = []

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        """在epoch结束时调用"""
        # 计算epoch统计信息
        if len(self.epoch_losses) > 0 and accelerator.is_main_process:
            avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
            min_loss = min(self.epoch_losses)
            max_loss = max(self.epoch_losses)

            epoch_summary = {
                'epoch': epoch_id,
                'avg_loss': avg_loss,
                'min_loss': min_loss,
                'max_loss': max_loss,
                'num_steps': len(self.epoch_losses)
            }

            # 记录到metrics文件
            self.log_metrics(accelerator, {'epoch_summary': epoch_summary})

            # 写入summary文件
            with open(self.summary_log_file, 'a') as f:
                f.write(f"\nEpoch {epoch_id} 完成:\n")
                f.write(f"  平均Loss: {avg_loss:.6f}\n")
                f.write(f"  最小Loss: {min_loss:.6f}\n")
                f.write(f"  最大Loss: {max_loss:.6f}\n")
                f.write(f"  训练步数: {len(self.epoch_losses)}\n")
                f.write("-" * 80 + "\n")

            print(f"\n{'='*80}")
            print(f"Epoch {epoch_id} 完成 - 平均Loss: {avg_loss:.6f}, 最小Loss: {min_loss:.6f}, 最大Loss: {max_loss:.6f}")
            print(f"{'='*80}\n")

        # 调用父类方法保存模型
        super().on_epoch_end(accelerator, model, epoch_id)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        """在训练结束时调用"""
        if accelerator.is_main_process:
            total_time = time.time() - self.start_time

            with open(self.summary_log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总训练时长: {total_time/3600:.2f} 小时\n")
                f.write(f"总训练步数: {self.global_step}\n")
                f.write(f"{'='*80}\n")

            print(f"\n训练完成！总时长: {total_time/3600:.2f} 小时, 总步数: {self.global_step}")

        # 调用父类方法
        super().on_training_end(accelerator, model, save_steps)
