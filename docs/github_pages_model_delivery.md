# GitHub Pages 与 Top-N 模型发布设计

## 目标

- 每次 `main` push 自动测试并部署最新 Web/Bot 到 GitHub Pages。
- 网页提供已晋级的 Top-N 模型供选择和真人对照。
- 模型二进制及导出 JSON 永不进入任何 Git commit、branch、tag 或 Git LFS 历史。
- Top-N 更新采用原子发布；旧资产可物理删除，不无限增长。

## 存储边界

1. Git 只保存源码、测试、JSON schema和不含权重的示例manifest。
2. 固定 GitHub Release `web-models` 保存当前 Top-N 发布资产。它不是 Git object history。
3. 本地晋级脚本根据固定评估结果选择Top-N，导出浏览器格式，计算SHA-256，先追加内容寻址资产和不可变generation manifest；不得先删除旧资产。
4. GitHub Actions从release下载指定generation中经过manifest白名单和SHA-256校验的模型，复制到临时 `_site/models/`，再用`actions/upload-pages-artifact`部署。Pages artifact不提交到Git；整个站点一次部署，访问者不会看到半新半旧状态。
5. Pages部署成功后才清理Release：保留“当前成功代＋上一成功代”，因此资产数量上界约为`2N + 2 manifests`；部署失败不清理，可直接回退上一manifest。
6. 禁止使用`gh-pages`持久分支、Git LFS或将模型commit后再删除；这些方案都会留下历史或额外对象管理负担。

## 自动部署

`.github/workflows/pages.yml`：

- 触发：`push: main`、`workflow_dispatch`、模型晋级后的`repository_dispatch`。
- 权限：`contents: read`、`pages: write`、`id-token: write`。
- 并发：`pages`组，取消旧部署。
- 门禁：Node/Web测试、manifest schema、资产数量/单文件/总大小、SHA-256、模型加载smoke、secret scan。
- 构建：只复制`web/`到`_site/`，从固定release下载manifest所列Top-N资产。
- 部署：`configure-pages`→`upload-pages-artifact`→`deploy-pages`。
- 失败时不覆盖上一次成功站点。

## Top-N 晋级

`scripts/publish_web_models.py`：

- 输入必须是不可变checkpoint绝对路径、SHA-256、固定seed评估摘要和选择规则。
- 默认N=3，可配置上限5。
- 使用预声明综合分数；同时保留不同能力标签，避免只按单一matched kill率选型。
- 导出到临时目录并浏览器加载smoke；资产名含内容SHA前缀，禁止覆盖同名不同内容。
- 先创建临时release/上传全部资产和`models.json`，远端回读校验后，再更新固定`web-models` release。
- 固定release只保留当前manifest引用的N个模型、manifest及必要签名；删除不再引用资产。
- 发布失败保持上一版完整可用。
- 发布成功触发Pages workflow重部署，无需为模型变更创建Git commit。

## Web行为

- 启动时读取同源`models/models.json`；显示名称、candidate、cycle、评估标签、checkpoint hash、模型文件hash和大小。
- 规则Bot列表来自当前Git push中的Bot registry，所以Pages始终展示最新代码策略。
- 模型按需加载，不在首页一次下载全部Top-N。
- 下载后先校验SHA-256再实例化；加载失败回退规则Bot并显示明确错误。
- 保留本地文件上传入口，便于测试未晋级模型；本地模型不会上传。

## 强制门禁

- `.gitignore`覆盖`web/models/`、`dist/`和发布临时目录。
- CI扫描所有Git跟踪对象路径与新增blob大小，禁止模型扩展名及大权重JSON。
- 文档明确：`git rm`不能消除历史；若误提交模型，在首次公开/协作前立即用`git filter-repo`清历史并强推，同时撤销泄露资产。
