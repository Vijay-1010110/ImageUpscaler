import psutil
import torch


def get_system_metrics():
    """Get CPU, RAM, and GPU usage metrics."""
    metrics = {}

    # CPU usage (percentage across all cores)
    metrics["System/CPU_percent"] = psutil.cpu_percent(interval=None)

    # RAM usage
    ram = psutil.virtual_memory()
    metrics["System/RAM_used_GB"] = ram.used / (1024 ** 3)
    metrics["System/RAM_percent"] = ram.percent

    # GPU metrics (if CUDA available)
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()

        # GPU memory
        allocated = torch.cuda.memory_allocated(gpu_idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(gpu_idx) / (1024 ** 3)
        total = torch.cuda.get_device_properties(gpu_idx).total_mem / (1024 ** 3)

        metrics["GPU/Memory_allocated_GB"] = allocated
        metrics["GPU/Memory_reserved_GB"] = reserved
        metrics["GPU/Memory_total_GB"] = total
        metrics["GPU/Memory_percent"] = (allocated / total) * 100

        # GPU utilization (if pynvml available)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["GPU/Utilization_percent"] = util.gpu
            metrics["GPU/Temperature_C"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            pass  # pynvml not installed — skip GPU utilization

    return metrics
