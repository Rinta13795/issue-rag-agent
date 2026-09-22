"""Demo 预设样例：提供 3 个适合面试现场演示的真实测试案例。"""

from src.demo.models import ExampleItem

PRESET_EXAMPLES = [
    ExampleItem(
        id="sample_duplicate",
        label="样例 1 · 重复 Issue (Duplicate)",
        tag="DUPLICATE",
        title='"Connect to an LDAP address book" on account setup opens CardDAV setup dialog',
        description=(
            'When clicking "Connect to an LDAP address book" during account setup, '
            'Thunderbird opens the "New CardDAV Addressbook" setup dialog instead of the LDAP address book dialog.'
        ),
        expected_decision="duplicate",
        note="Thunderbird 真实历史 Bug，命中已知重复 Issue thunderbird:1727737 / 1729050",
    ),
    ExampleItem(
        id="sample_similar",
        label="样例 2 · 相似历史 Issue (Similar / Duplicate)",
        tag="SIMILAR",
        title='Firefox View tracks browsing pages even when history preference is set to "Never remember history"',
        description=(
            'Even though I have "History" turned off (set to Never remember History), '
            'Firefox View continues to track and display webpages and recently closed tabs.'
        ),
        expected_decision="similar / duplicate",
        note="Firefox View 历史记录隐私问题，命中 firefox:1839896 / 1808085",
    ),
    ExampleItem(
        id="sample_new",
        label="样例 3 · 独立全新 Issue (New)",
        tag="NEW",
        title="Add WebGPU compute pipeline offscreen canvas rendering support in WebWorker on macOS M3",
        description=(
            "Feature request: Enable TransferControlToOffscreen on WebWorker context with WebGPU compute pipelines "
            "and texture array storage on Apple Silicon M3 architecture."
        ),
        expected_decision="new",
        note="GitBugs 语料库外的新特性与问题，无高分匹配候选，判定为全新 Issue",
    ),
]


def get_preset_examples() -> list[ExampleItem]:
    return PRESET_EXAMPLES
