"""切分模块：导入子模块以触发 @register 注册"""
from . import recursive  # noqa: F401
# 阶段2将新增: from . import semantic, parent_child
