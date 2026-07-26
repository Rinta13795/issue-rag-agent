"""Agent 包入口：只导出 run_agent。"""

__all__ = ["run_agent"]


def __getattr__(name: str):
    """输入导出名，按需懒加载 run_agent，避免 import src.agent 时提前初始化图依赖。"""
    if name == "run_agent":
        from src.agent.graph import run_agent

        return run_agent
    raise AttributeError(name)
