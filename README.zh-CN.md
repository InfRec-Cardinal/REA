<h1 align="center">REA</h1>

<p align="center">
  <strong>A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2610.42053"><img src="https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square" alt="Paper PDF"></a>
  <a href="https://arxiv.org/abs/2610.42053"><img src="https://img.shields.io/badge/arXiv-2610.42053-B31B1B?style=flat-square&amp;logo=arxiv&amp;logoColor=white" alt="arXiv 2610.42053"></a>
  <a href="https://huggingface.co/papers/2610.42053"><img src="https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E?style=flat-square&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face Paper"></a>
  <a href="https://github.com/InfRec-Cardinal/REA"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&amp;logo=github&amp;logoColor=white" alt="GitHub"></a>
  <a href="https://mikeeeeeeeeeeeeee.github.io/REA"><img src="https://img.shields.io/badge/Project%20Page-2088A2?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" alt="Project Page"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-7042C1?style=flat-square&amp;logo=bookstack&amp;logoColor=white" alt="Citation BibTeX"></a>
  <a href="https://github.com/InfRec-Cardinal/REA/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/InfRec-Cardinal/REA/ci.yml?branch=main&amp;style=flat-square&amp;logo=githubactions&amp;logoColor=white&amp;label=CI" alt="CI"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

> 🚧 **持续更新中。** REA 正在积极开发，会不断接入更多 benchmark 平台（如 [RoboDojo](docs/benchmarks/robodojo.md)）和更多真机，进展见[支持的平台](#platforms)和[更新日志](CHANGELOG.md)。欢迎贡献新的平台和机器人适配。

REA 是一个具身智能体（embodied agent）框架：它在多个快速策略之间路由，而不是把大模型放进控制回路。框架分为三层：

- **System 0：能力（capability）**，以控制频率输出动作，包括冻结的代码程序、VLA 策略、LLM 工具调用控制器，以及运动控制和全身控制器。
- **System 1：轻量路由器**。route memory 选出在开发集 episode 上表现最好的能力先运行；judge gate 只在某个能力返回时检查任务是否完成；升级阶梯（escalation ladder）在同一份预算内决定接下来运行什么。
- **System 2：大脑**。离线时编写或挑选能力；在线时只在边界处做判断或给出建议。正在运行的能力从不等待它。

同一套接口连接 benchmark 平台（LIBERO / LIBERO-Pro、BEHAVIOR-1K、RoboTwin、RoboDojo、ARC-AGI-3、Gymnasium）、真机（ALOHA、LeRobot SO-100/101 与 Koch、OpenRAL 和 RPent 支持的机器人、Unitree G1、RoboDojo-RealEval 的双臂机器人，以及你自己的驱动）和各类方法（LLM agent、VLA、code-as-policy、控制器）。REA 既可以驱动平台的环境，也可以作为策略，被自带控制循环的平台调用。每个 episode 都用同一种格式记录决策、动作、耗时和用量。

## 一分钟上手

安装 [Node.js](https://nodejs.org/) 22.19 或更高版本：

```bash
git clone https://github.com/InfRec-Cardinal/REA.git
cd REA
npm run demo            # 接口演示：一个网格和一个两关节夹具，不需要模型或硬件
npm run demo:routed     # 一个 routed episode：route memory、边界 judge、升级阶梯、结束时一次性评分
npm run demo:async      # System 2 在后台思考时，System 1 持续路由
npm run rea -- list     # 列出环境、策略和预设
```

这些演示是确定性的，不需要 Python、密钥或硬件；它们展示的是接口，不代表机器人性能。

## 一次配置，任选后端

配置文件写明环境、策略（或 systems）、episode 和预算：

```json
{
  "version": 1,
  "environment": { "type": "gymnasium", "options": { "environmentId": "CartPole-v1" } },
  "policy": { "type": "chat" },
  "episodes": [{ "id": "cartpole-0", "seed": 0, "instruction": "Balance the pole for as long as possible." }],
  "budget": { "maxSteps": 200, "maxDurationMs": 900000, "decisionTimeoutMs": 60000 }
}
```

需要模型的策略可以使用任何兼容 OpenAI Chat Completions 的接口。把服务商信息写进 `.env`（参见 [`env.example`](env.example)）：

```dotenv
REA_MODEL=your-model
REA_BASE_URL=http://localhost:8000/v1
REA_API_KEY=your-api-key
```

```bash
npm run rea -- init gymnasium            # 生成 rea.config.json
python3 -m venv runtime/gym-venv && runtime/gym-venv/bin/python -m pip install -r requirements-gym.txt
export GYM_PYTHON="$PWD/runtime/gym-venv/bin/python"
npm run rea -- doctor rea.config.json    # 检查配置和本地依赖，不连接任何设备或服务
npm run rea -- run rea.config.json --steps 10 --output outputs/first-run
```

| 预设 | 连接的对象 | 文档 |
|---|---|---|
| `demo` | 合成的网格和关节夹具 | 无需额外准备 |
| `gymnasium` | 已注册的 Gymnasium 任务及其原生动作空间 | [框架参考](docs/framework.md#gymnasium-and-additional-robots) |
| `libero` | LIBERO / LIBERO-Pro 任务套件 | [LIBERO](docs/benchmarks/libero.md) |
| `arc-agi3` | 官方 ARC-AGI-3 SDK、游戏和计分卡 | [ARC-AGI-3](docs/benchmarks/arc-agi3.md) |
| `aloha` | ACT 双臂仿真；Classic ROS1 ALOHA 真机 | [机器人](docs/robots/README.md)、[ALOHA 真机](docs/robots/aloha-real.md) |
| `lerobot` | SO-100/101、双臂 SO、Koch，以及兼容 Robot 接口的工厂 | [机器人](docs/robots/README.md#lerobot-so-100so-101-biso-and-koch) |
| `openral` | 由 manifest 描述的单臂、双臂、移动平台和人形机器人 | [OpenRAL 与 RPent](docs/robots/openral-rpent.md) |
| `rpent` | LIBERO、RoboCasa 和 RoboTwin 环境服务 | [OpenRAL 与 RPent](docs/robots/openral-rpent.md) |
| `g1` | Unitree G1 的相机、安全门控和执行器 | [Unitree G1](docs/robots/unitree-g1.md) |

BEHAVIOR-1K、RoboTwin 和 RoboDojo 有专门的运行器或适配器：[BEHAVIOR](docs/benchmarks/behavior.md)、[RoboTwin](docs/benchmarks/robotwin.md)、[RoboDojo](docs/benchmarks/robodojo.md)；全部平台见[支持的平台](#platforms)。真机预设默认处于预览模式（`execute: false`）。真机上的任务是否成功需要评估器判定，命令被接收不等于任务成功。

## Routed 与异步系统

```js
systems: {
  mode: "routed",
  system0: [
    { id: "fix", kind: "program", policy: { type: "module", options: { module: "./fix-program.mjs" } } },
    { id: "vla", kind: "vla", policy: { type: "lingbot-vla", options: { url: "ws://127.0.0.1:8006" } } },
  ],
  system1: { type: "route-memory", options: { table: "./route-table.json", recoverOn: "judge" } },
  system2: { type: "judge", options: { model: "your-vision-model", threshold: 0.6 } },
  maxStepsPerRun: 50,
}
```

`tools/route-table.mjs` 用每个能力各一次开发集评测的结果生成 route table。详见 [routed 系统](docs/routed-systems.md)；若要逐个动作路由，同时让 System 2 在后台思考，见[异步系统](docs/async-systems.md)。

<a id="platforms"></a>

## 支持的平台

新的平台和机器人会持续加入，每次新增都记录在[更新日志](CHANGELOG.md)中。**已支持**：已包含适配，并在该平台的接口上测试过。**新接入**：已包含适配并完成离线测试，平台上的验证仍在进行。

| Benchmark 平台 | 接入方式 | 状态 |
|---|---|---|
| [LIBERO / LIBERO-Pro](docs/benchmarks/libero.md) | `libero` 环境；可恢复执行的冻结程序后端 | 已支持 |
| [BEHAVIOR-1K](docs/benchmarks/behavior.md)（R1Pro） | 程序后端；在线 agent 的 episode 运行器 | 已支持 |
| [RoboTwin 2.0](docs/benchmarks/robotwin.md) | VLA / 代码候选策略，配合成功率路由器；RPent 与 OpenRAL 场景 | 已支持 |
| [ARC-AGI-3](docs/benchmarks/arc-agi3.md) | 官方 SDK、游戏和计分卡 | 已支持 |
| [Gymnasium](docs/framework.md#gymnasium-and-additional-robots) | 任意已注册环境及其原生动作空间 | 已支持 |
| [RoboDojo](docs/benchmarks/robodojo.md)（仿真与 RealEval） | 通过 `rea serve`，作为 XPolicyLab 策略接入 | 新接入 |
| 其他 XPolicyLab 平台（RoboTwin、RMBench） | 同一个 XPolicyLab 适配器 | 新接入 |
| 你的平台 | 写一个[环境](docs/extending/environments.md)（由 REA 掌控循环），或使用 [`rea serve`](docs/extending/policy-server.md)（由平台掌控循环） | |

| 真机 | 接入方式 | 状态 |
|---|---|---|
| ALOHA：ACT 仿真、Classic ROS1 真机 | `aloha` | 已支持 |
| LeRobot：SO-100/101、双臂 SO、Koch、兼容 Robot 接口的工厂 | `lerobot` | 已支持 |
| OpenRAL 机器人：Franka、UR5e/UR10e、Sawyer、WidowX、OpenArm、人形机器人等 | `openral` | 已支持 |
| Unitree G1（服务、安全门控、cerebellum） | `g1` | 已支持 |
| RoboDojo-RealEval：ARX X5、Piper、Piper X | XPolicyLab 适配器 | 新接入 |
| 你的机器人 | 写一个 [Python 驱动类](docs/extending/robots.md)（`python-robot`）或 JavaScript 驱动，或从你的机器人软件栈调用 `rea serve` | |

## 接入你自己的组件

| 想接入 | 需要实现 | 文档 |
|---|---|---|
| LLM agent、VLA、代码策略或控制器 | `decide(input) → {action}` | [System 0 策略](docs/extending/policies.md) |
| 路由器、judge 或大脑 | `route(input)`、`judge(input)` 或 `think(input)` | [路由器与大脑](docs/extending/systems.md) |
| Benchmark 或仿真器 | `reset`、`step`（routed episode 还需要 `score`） | [环境](docs/extending/environments.md) |
| 执行程序的仿真器 | JSON-lines 程序协议 | [程序后端](docs/extending/program-backends.md) |
| 真机 | Python 驱动类（`python-robot`）或 JavaScript 驱动 | [真机](docs/extending/robots.md) |
| 自带控制循环的平台（XPolicyLab / RoboDojo、比赛评测器、机器人软件栈） | `rea serve` 加一个小适配器；已附 Python 客户端 | [策略服务](docs/extending/policy-server.md) |

以上组件都可以在配置中用 `type: "module"` 加载，或注册到 `createRegistry()`；无需修改 `src/` 下的任何文件。

```js
import { createEmbodiedAgent } from "rea-agent";   // 在本仓库内使用：./src/framework/index.mjs

const agent = await createEmbodiedAgent("./rea.config.json");
try {
  const episode = await agent.run({ seed: 3 });
  const evaluation = await agent.evaluate({ outputDir: "outputs/eval" });
} finally {
  await agent.close();
}
```

## 文档

详细文档目前为英文。

- [架构](docs/architecture.md)：三层系统、两种运行时、各类接口和仓库结构
- [框架参考](docs/framework.md)：配置、CLI、环境与策略接口、输出格式
- [Routed 系统](docs/routed-systems.md) · [异步系统](docs/async-systems.md)
- 扩展：[策略](docs/extending/policies.md) · [路由器与大脑](docs/extending/systems.md) · [环境](docs/extending/environments.md) · [程序后端](docs/extending/program-backends.md) · [真机](docs/extending/robots.md) · [策略服务](docs/extending/policy-server.md)
- Benchmark：[LIBERO](docs/benchmarks/libero.md) · [BEHAVIOR](docs/benchmarks/behavior.md) · [RoboTwin](docs/benchmarks/robotwin.md) · [RoboDojo](docs/benchmarks/robodojo.md) · [ARC-AGI-3](docs/benchmarks/arc-agi3.md)
- 机器人：[ALOHA、LeRobot](docs/robots/README.md) · [ALOHA 真机](docs/robots/aloha-real.md) · [OpenRAL 与 RPent](docs/robots/openral-rpent.md) · [Unitree G1](docs/robots/unitree-g1.md) · [G1 cerebellum](agent-vla-async/README.md) · [LingBot-VLA](docs/policies/lingbot-vla.md)
- [测试](docs/testing.md) · [贡献指南](CONTRIBUTING.md) · [更新日志](CHANGELOG.md) · [第三方声明](NOTICE.md)

## 未包含的内容

模型权重、VLA 服务、仿真器及其资产（LIBERO、BEHAVIOR-1K / OmniGibson、RoboTwin）、ASPIRE、数据集、机器人标定和运行记录都需要另行安装或准备，各文档会说明对应集成需要什么。来自第三方项目的 benchmark 任务提示词和 API 文档不在本仓库中重新分发。

<a id="citation"></a>

## 引用

```bibtex
@article{zhou2026rea,
  title   = {{REA}: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control},
  author  = {Zhou, Pengfei and Ma, Yixing and Zhang, Zhengfeiyang and Chen, Haoyuan and Wan, Zhenglin and Wang, Hexin and Zhao, Wangbo and You, Yang},
  journal = {arXiv preprint arXiv:2610.42053},
  year    = {2026}
}
```

## 许可证

MIT 许可证，见 [LICENSE](LICENSE)。第三方组件及其署名见 [NOTICE.md](NOTICE.md)。
