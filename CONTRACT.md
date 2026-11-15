# 分页归档隔离复核

## 已提供的材料与入口
data/incident/pack.json 是本批完整事故包：初始记录、18个可信事务和8个原始归档页均在其中。JSON Pointer 用于引用材料位置，如 /transactions/0/writes/1、/pages/0/rows/0。
source.py 提供 load(directory)、snapshot(pack,commits,at)、replay(pack,schedule)，仅读取和重放给定时序。evidence.py 提供 replay_kept(pack,witness)，仅核验已选隔离方案，不判断其最优性。
data/main 保留旧回归材料；data/tiny 是正常原子移位及短页样例；data/conflict 的两页分别可解释但联合矛盾；data/audit_conflict 的可信事务本身无法满足严格前驱关系；data/joint 的三页两两可解释而整体矛盾。
保留 source.py、evidence.py、checks.py、evidence_checks.py 和全部原始材料及原断言。requirements.txt 提供 SciPy；本题用 Python 与 SciPy 分析，不开发通用采集产品。

基础回归：python3 evidence_checks.py normal
已有演示：python3 evidence.py data/tiny data/tiny/witness.json
该演示返回 kept=["p1","p2"]、isolated=[]，p2 正文为 [["a",2,40,11,false]]。data/tiny/schedule.json 为原重放器的同一时序格式。

## 输入与源端语义
INPUT_DIR/pack.json 含 target、initial、transactions、pages。target 为旧重放器保留的非负整数，本轮不核定其版本或漏采；目标仅为最少整页隔离。
每行是 [id,rev,rank,value,deleted]：非空字符串ID、正整数版本、整数排序值和业务值、布尔墓碑。initial 是时刻0的有效行，ID不重复。
每个事务含 id、window=[lo,hi]、after=[前驱事务ID]、writes=[行...]。每事务恰提交一次，提交时刻为闭区间内整数；前驱严格早于后继。同ID的更正通过 after 有序，rev 逐次加1；事务内多行原子生效，完整替换旧行。deleted=true 撤回，后续版本可复活；首次写入前的ID不存在。事务审计均可信，即使隔离全部页面，所有事务及其时窗、依赖仍须满足。
每页含 id、window、offset、limit、rows，读取时刻为该闭区间内整数。所有时间共用一个相对刻度，提交与读取同刻时先提交；无因果关系的同刻事务写不同ID，效果可交换。材料数组的顺序不添加时间约束。
页面从读取时刻未删除的完整记录集按 (rank,id) 升序排列，跳过 offset 行、取至多 limit 行；ID用Python字符串顺序打破同rank平局。rows 的完整字段、行序都必须一致，短页同时约束记录总数。value不参与排序，offset不随其他页修正。
只允许整页隔离，不允许删行、改字段或隔离事务。隔离页的正文、时窗和读取事件不再约束解释；其余页面必须由同一组提交及读取时刻共同解释。不能用逐页或两两可解释替代整批可解释。

## 分析入口与产物
python3 analysis.py run INPUT_DIR OUTPUT_DIR
读取材料并重算，写 OUTPUT_DIR/result.json 和 OUTPUT_DIR/REPORT.md；本批交付目录为 output。
可信事务可行时，result.json 的格式为：
{"status":"ok","minimum":非负整数,"example":见证,"pages":[页面结果...]}
minimum 是最少隔离页数。example 为一份达到该数目的完整见证。
pages 按页面ID排序，每页结果含 id、state、keep、isolate、source：
- state 为 keep、isolate、either，分别表示全部最优解释都保留、全部都隔离、最优解释中两种情况均存在。
- keep 和 isolate 各为该方向的最优见证，不可能的方向填 null；不能把一份任意最优方案当成逐页必然结论。
- source 是该页在输入中的位置，如 /pages/0。
见证格式为 {"isolated":[页面ID...],"schedule":{"commits":{"事务ID":整数时刻...},"reads":{"保留页ID":整数时刻...}}}。isolated 无重复、按ID排序；commits 包含全部事务且仅包含这些事务；reads 包含全部保留页且仅包含这些页。每份见证恰隔离 minimum 页，须经 evidence.replay_kept 调用原 replay 核验。不同结论可以复用同一份完整见证，不能拼接局部时序。
若可信事务本身不可行，输出 {"status":"conflict","minimum":null,"example":null,"pages":[]}，报告追溯冲突的事务、时窗和依赖，不输出部分页面分类；不要求最小冲突集。
报告简述隔离结论和人工复查含义，引用原始位置，解释联合时间、原子更正和排序截取如何产生冲突及替代解释，说明为何更少隔离不可行，记录实际核验结果；不枚举全部最优方案，不修改归档或修复爬虫。

python3 analysis.py verify INPUT_DIR OUTPUT_DIR
从原始材料重新求最优值及逐页分类，核对已交 result.json 和报告关键结论，逐项重放全部见证。重新检查更少隔离不可行，以及每页在同一最优值下两个方向的可行性；不能只核验有见证的方向。
执行原基础回归；对 data/tiny、data/conflict、data/audit_conflict、data/joint 独立穷举整数提交、读取时刻及隔离集合，对照最优值和逐页分类，不复用主分析器的建模逻辑。小例还须核验同刻先提交、原子排序移位、撤回复活和短页。核验失败须非零退出并说明差异，不能只检查文件存在或信任缓存结果。
python3 analysis.py demo INPUT_DIR
重新分析并展示本批最少隔离数、各页分类及一份经原重放器核验的方案；可信事务冲突时展示冲突证据。

## 运行清单
已有 .pointwise-runtime.json 可运行基础回归与样例演示。交付时保留 version=1、kind=cli 和 setup 的argv数组列表，将 verify 改接 ["python3","analysis.py","verify","data/incident","output"]，demo 改接 ["python3","analysis.py","demo","data/incident"]。run、verify、demo 均从材料重算，不按页面ID硬编码答案。
