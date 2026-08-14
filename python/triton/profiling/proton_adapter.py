"""
ProtonAdapter — 将 triton.profiler.profile 的函数式 API 包装为 OnlineProfiler 期望的
面向对象 API。

实际 API (triton.profiler.profile):
  profile.start(name, *, context, data, backend, mode, hook) -> Optional[int]
  profile.activate(session) / profile.deactivate(session)
  profile.finalize(session, output_format)

期望 API (OnlineProfiler):
  Proton().start()               → 初始化
  Proton().profile(name)         → 上下文管理器, 进入/退出 profiling
  Proton().get_last_metrics()    → 返回 dict{...}
  Proton().stop()                → 清理

注意: shadow 模式的 profile 数据在 finalize() 时一次性写入文件, 运行时无 per-step
实时指标, 因此 get_last_metrics() 返回占位值; 实际瓶颈分析在 finalize 后解析文件。
"""
import os
from contextlib import contextmanager
from typing import Dict, List, Optional


class ProtonAdapter:
    """桥接 triton.profiler.profile 的函数式 API 到面向对象接口。"""

    def __init__(self):
        self._session: Optional[int] = None
        self._active = False
        self._profile_count = 0
        try:
            # triton.profiler 直接导出函数式 API（start/finalize/activate/deactivate）,
            # 其中 profile 是 context-manager 函数而非模块。
            from triton.profiler import start, finalize, activate, deactivate
            self._start = start
            self._finalize = finalize
            self._activate = activate
            self._deactivate = deactivate
            self._available = True
        except Exception:
            self._start = self._finalize = self._activate = self._deactivate = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self):
        """初始化 proton session。"""
        if not self._available:
            return
        try:
            self._session = self._start(
                name="pact_pgo",
                context="shadow",   # CUDA graphs compatible
                data="tree",        # hierarchical profile data
            )
            self._active = self._session is not None
            print(f"[PACT ProtonAdapter] Proton session started "
                  f"(session={self._session})")
        except Exception as e:
            print(f"[PACT ProtonAdapter] start failed: {e}")
            self._available = False

    @contextmanager
    def profile(self, name: str = "step", metrics: Optional[List[str]] = None):
        """上下文管理器 — 在 profiling 区域内执行 GPU kernel。

        shadow 模式自动收集所有 CUDA activity; metrics 参数在当前 API 下被忽略。
        """
        if not self._active or self._session is None:
            yield  # no-op
            return
        try:
            self._activate(self._session)
            self._profile_count += 1
            yield
        finally:
            try:
                self._deactivate(self._session)
            except Exception:
                pass

    def get_last_metrics(self) -> Dict:
        """返回最近一次 profile 的硬件指标。

        shadow 模式数据在 finalize 后写入文件, 运行时无法实时获取 per-step 指标,
        故返回占位值; 实际分析在 stop() 后解析 profile 文件。
        """
        if not self._active:
            return {}
        return {
            "sm_efficiency": 0.0,
            "occupancy": 0.0,
            "memory_throughput": 0.0,
            "l2_hit_rate": 0.0,
            "global_load": 0,
            "shared_load": 0,
        }

    def stop(self, output_format: str = "hatchet"):
        """停止 profiling 并写入结果文件。

        output_format 可选 ["hatchet", "hatchet_msgpack", "chrome_trace"]。
        """
        if not self._active or self._session is None:
            return
        try:
            self._finalize(self._session, output_format=output_format)
            print(f"[PACT ProtonAdapter] Profile finalized "
                  f"(session={self._session}, format={output_format})")
        except Exception as e:
            print(f"[PACT ProtonAdapter] stop failed: {e}")
        finally:
            self._active = False
            self._session = None


# Backward-compatible export as 'Proton'
Proton = ProtonAdapter
