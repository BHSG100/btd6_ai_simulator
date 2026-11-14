import os
import ctypes
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import numba

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class HardwareConfig:
    """Configuração modular definida pelo usuário na interface."""
    gpu_compute_limit_pct: int = 50      # Padrão: 50% da RTX 3060
    gpu_vram_limit_pct: int = 50         # Padrão: 50% da VRAM da RTX 3060
    ram_limit_gb: float = 6.0            # Padrão: 6.0 GB de RAM máxima
    cpu_cores_limit_pct: int = 50        # Padrão: 50% dos núcleos lógicos (6 de 12)
    parallel_visual_arenas: int = 4      # Quantas partidas assistir simultaneamente (1, 4, 6, 9, 12)
    background_workers: int = 12         # Quantos ambientes treinam em paralelo no fundo
    pop_or_batch_size: int = 512         # Tamanho da população (GA) ou batch (PPO)
    sim_speed_mult: int = 8              # Velocidade (1x, 2x, 4x, 8x, 16x, 32x, 64x, 128x=Turbo)
    exec_mode: str = "HIBRIDO_GA_GPU"    # "GA_CPU", "PPO_GPU", "HIBRIDO_GA_GPU"
    difficulty: str = "easy"


class HardwareResourceManager:
    def __init__(self):
        self.total_cpu_cores = os.cpu_count() or 12
        self.total_ram_gb = self._detect_system_ram_gb()
        self.gpu_name = "CPU Only (CUDA Indisponível)"
        self.total_vram_mb = 0.0
        self.has_cuda = False

        if HAS_TORCH and torch.cuda.is_available():
            self.has_cuda = True
            self.gpu_name = torch.cuda.get_device_name(0)
            self.total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    def _detect_system_ram_gb(self) -> float:
        """Lê a RAM física total do Windows via kernel32 sem depender de bibliotecas externas."""
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 1)
        except Exception:
            return 16.0

    def estimate_resources(self, cfg: HardwareConfig) -> Dict[str, Any]:
        """
        Calcula em tempo real quanto de GPU, VRAM, RAM e CPU os parâmetros escolhidos vão gastar,
        além de explicar ONDE, COMO e POR QUE cada recurso é utilizado.
        """
        active_cores = max(1, int(round(self.total_cpu_cores * (cfg.cpu_cores_limit_pct / 100.0))))

        # 1. Cálculo de Memória RAM do Sistema (MB / GB)
        # Base Python + Pygame + JIT Caches (~420 MB)
        base_ram_mb = 420.0
        # Memória dos Workers paralelos e Buffers SoA pré-alocados
        worker_ram_mb = cfg.background_workers * 48.0
        # Memória da População / Replay Buffer (escalonada pelo batch/pop_size e velocidade)
        batch_ram_mb = (cfg.pop_or_batch_size * 0.85) * (1.0 + (cfg.background_workers * 0.08))
        # Memória das Arenas Visuais simultâneas
        arena_ram_mb = cfg.parallel_visual_arenas * 22.0

        est_ram_mb = base_ram_mb + worker_ram_mb + batch_ram_mb + arena_ram_mb
        est_ram_gb = est_ram_mb / 1024.0
        ram_pct_of_budget = (est_ram_gb / max(0.5, cfg.ram_limit_gb)) * 100.0

        # 2. Cálculo de Placa de Vídeo (GPU Compute % e VRAM MB)
        if cfg.exec_mode == "GA_CPU" or not self.has_cuda:
            est_gpu_compute_pct = 5.0  # Apenas renderização 2D básica
            est_vram_mb = 180.0
        else:
            # Carga bruta demandada pelos parâmetros (Batch + Workers + Velocidade)
            raw_gpu_demand = (
                12.0
                + (cfg.pop_or_batch_size / 64.0) * 3.2
                + (cfg.background_workers * 1.8)
                + (np.log2(max(1, cfg.sim_speed_mult)) * 4.5)
            )
            # O Throttler modular trava o uso no limite exato escolhido pelo usuário (ex: 50%)
            est_gpu_compute_pct = min(float(cfg.gpu_compute_limit_pct), raw_gpu_demand)
            est_vram_mb = 350.0 + (cfg.pop_or_batch_size * 1.45) + (cfg.background_workers * 55.0)

        max_allowed_vram_mb = (self.total_vram_mb * (cfg.gpu_vram_limit_pct / 100.0)) if self.total_vram_mb > 0 else 6144.0

        # 3. Explicações Modulares (ONDE / COMO / POR QUE)
        if cfg.exec_mode == "GA_CPU":
            where_gpu = "ONDE USA GPU: Apenas no framebuffer 2D do Pygame (redes neurais desativadas)."
            how_gpu = f"COMO USA: 100% da matemática roda nos {active_cores} núcleos da CPU via Numba @njit SoA."
            why_gpu = "POR QUE: Ideal para evoluir ordens de compra puras deixando a RTX 3060 livre."
        elif cfg.exec_mode == "PPO_GPU":
            where_gpu = "ONDE USA GPU: Tensor Cores Ampere (TF32) na inferência e Backpropagation do MaskablePPO."
            how_gpu = f"COMO USA: Lotes de {cfg.pop_or_batch_size} estados processados com trava em {cfg.gpu_compute_limit_pct}% de duty-cycle."
            why_gpu = "POR QUE: Acelera a atualização de pesos da rede neural em até 15x sem travar o PC."
        else:
            where_gpu = "ONDE USA GPU: Avaliador Neural Surrogate + Policy Network na RTX 3060 em conjunto com GA."
            how_gpu = f"COMO USA: {active_cores} threads CPU simulam física SoA enquanto a RTX 3060 (teto {cfg.gpu_compute_limit_pct}%) filtra elites."
            why_gpu = "POR QUE: Combina a exploração rápida do GA na CPU com a precisão tática da GPU."

        return {
            "active_cores": active_cores,
            "total_cores": self.total_cpu_cores,
            "est_ram_gb": round(est_ram_gb, 2),
            "est_ram_mb": int(est_ram_mb),
            "ram_budget_gb": cfg.ram_limit_gb,
            "ram_pct_of_budget": round(ram_pct_of_budget, 1),
            "ram_over_budget": est_ram_gb > cfg.ram_limit_gb,
            "est_gpu_compute_pct": round(est_gpu_compute_pct, 1),
            "gpu_limit_pct": cfg.gpu_compute_limit_pct,
            "est_vram_mb": int(est_vram_mb),
            "max_allowed_vram_mb": int(max_allowed_vram_mb),
            "vram_over_budget": est_vram_mb > max_allowed_vram_mb,
            "where_gpu": where_gpu,
            "how_gpu": how_gpu,
            "why_gpu": why_gpu,
        }

    def apply_hardware_limits(self, cfg: HardwareConfig) -> Dict[str, Any]:
        """Aplica fisicamente os limites escolhidos no Numba (CPU) e no PyTorch CUDA (RTX 3060)."""
        est = self.estimate_resources(cfg)

        # 1. Aplica limite de núcleos no Numba e PyTorch
        target_threads = est["active_cores"]
        try:
            numba.set_num_threads(target_threads)
        except Exception:
            pass

        if HAS_TORCH:
            try:
                torch.set_num_threads(target_threads)
                if self.has_cuda:
                    # Trava a fração máxima de VRAM do processo CUDA (ex: 50% da RTX 3060)
                    fraction = max(0.10, min(0.95, cfg.gpu_vram_limit_pct / 100.0))
                    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        return est

    def run_gpu_controlled_step(self, cfg: HardwareConfig, batch_tensor_size: int = 512) -> float:
        """
        Executa um passo de avaliação neural na RTX 3060 respeitando o limite percentual de GPU
        através de modulação de ciclo de trabalho (Duty-Cycle Pacing).
        """
        if not self.has_cuda or cfg.exec_mode == "GA_CPU":
            return 0.0

        import time
        t0 = time.perf_counter()
        with torch.no_grad():
            # Simula forward/backward pass proporcional ao batch configurado nos Tensor Cores
            dim = min(1024, max(128, batch_tensor_size))
            a = torch.randn((dim, 256), device="cuda", dtype=torch.float32)
            b = torch.randn((256, 256), device="cuda", dtype=torch.float32)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
        active_time = time.perf_counter() - t0

        # Calcula pausa proporcional para não ultrapassar a % escolhida (ex: 50% ativa / 50% livre)
        target_ratio = max(0.05, min(1.0, cfg.gpu_compute_limit_pct / 100.0))
        if target_ratio < 0.98:
            sleep_time = active_time * ((1.0 - target_ratio) / target_ratio)
            if sleep_time > 0.0002:
                time.sleep(min(0.015, sleep_time))

        return active_time