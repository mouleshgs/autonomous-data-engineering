import { useEffect, useState } from "react";
import "./pipeline.css";
import axios from "axios";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Bot,
  CheckCircle2,
  Database,
  Download,
  FileUp,
  Gauge,
  LayoutDashboard,
  LoaderCircle,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  XCircle,
} from "lucide-react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8001";
const api = axios.create({ baseURL: `${API_BASE_URL}/api` });
type Dataset = {
  id: string;
  name: string;
  source_type: string;
  rows: number;
  columns: number;
  status: string;
  profile?: any;
  quality_before?: any;
  quality_after?: any;
  preview?: any[];
};
type Run = {
  id: string;
  status: string;
  before: any;
  after: any;
  plan: any[];
  stages: any[];
  model_type?: string;
  target_column?: string;
  model_benchmark?: any;
};
const stages = [
  "Source Analyzer",
  "Profiler Agent",
  "Planner Agent",
  "Ingestion",
  "Cleaning",
  "Transformation",
  "Validation",
  "Storage",
  "Model Benchmark",
];

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<Dataset | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [logs, setLogs] = useState<any[]>([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [activeStageIndex, setActiveStageIndex] = useState(0);
  const [view, setView] = useState("overview");
  const [modelType, setModelType] = useState<string>("generic");
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);

  const refresh = async () => {
    const response = await api.get("/datasets");
    setDatasets(response.data);
    if (!selected && response.data[0]) setSelected(response.data[0]);
    const logResponse = await api.get("/agents/logs");
    setLogs(logResponse.data);
  };
  useEffect(() => {
    refresh().catch(() => undefined);
  }, []);

  useEffect(() => {
    if (selected) {
      api
        .get(`/datasets/${selected.id}/columns`)
        .then((res) => {
          const cols: string[] = res.data.columns || [];
          setDatasetColumns(cols);
          if (cols.length > 0) {
            setTargetColumn(cols[cols.length - 1]);
          }
        })
        .catch(() => undefined);
    }
  }, [selected?.id]);

  useEffect(() => {
    if (!pipelineRunning) return;
    const timer = window.setInterval(() => {
      setActiveStageIndex((index) => Math.min(index + 1, stages.length - 1));
    }, 650);
    return () => window.clearInterval(timer);
  }, [pipelineRunning]);
  const upload = async (file: File) => {
    setBusy(true);
    const body = new FormData();
    body.append("file", file);
    const response = await api.post("/datasets/upload", body);
    setSelected(response.data);
    await refresh();
    setBusy(false);
  };
  const execute = async () => {
    if (!selected) return;
    setBusy(true);
    setPipelineRunning(true);
    setActiveStageIndex(0);
    setRun(null);

    try {
      const response = await api.post(`/pipelines/${selected.id}/run`, {
        model_type: modelType,
        target_column: modelType !== "generic" ? targetColumn : undefined,
      });
      const startedRun = response.data;
      setRun(startedRun);

      const pollRun = async () => {
        try {
          const latest = await api.get(`/pipelines/${startedRun.id}`);
          setRun(latest.data);

          if (latest.data.status === "SUCCESS" || latest.data.status === "FAILED") {
            setPipelineRunning(false);
            setBusy(false);
            await refresh();
            setSelected((await api.get(`/datasets/${selected.id}`)).data);
            return;
          }

          window.setTimeout(() => {
            void pollRun();
          }, 1200);
        } catch {
          setPipelineRunning(false);
          setBusy(false);
        }
      };

      await pollRun();
    } catch {
      setPipelineRunning(false);
      setBusy(false);
    }
  };
  const triggerDownload = () => {
    if (!selected) return;
    const link = document.createElement("a");
    link.href = `${API_BASE_URL}/api/datasets/${selected.id}/download`;
    link.download = `${selected.name.replace(/\.[^/.]+$/, "")}_processed.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  const ask = async () => {
    if (!question) return;
    setBusy(true);
    const response = await api.post("/analytics/query", { question });
    setAnswer(response.data);
    setBusy(false);
  };
  const score =
    selected?.quality_after?.overall ?? selected?.quality_before?.overall ?? 0;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={18} />
          </div>
          <div>
            <strong>ORBITAL</strong>
            <span>data intelligence</span>
          </div>
        </div>
        <nav>
          {[
            ["overview", "Overview", LayoutDashboard],
            ["datasets", "Datasets", Database],
            ["pipeline", "Pipeline Monitor", Activity],
            ["benchmark", "Benchmark", Gauge],
            ["logs", "Agent Decisions", Bot],
            ["analytics", "Analytics", BarChart3],
          ].map(([key, label, Icon]: any) => (
            <button
              className={view === key ? "active" : ""}
              onClick={() => setView(key)}
              key={key}
            >
              <Icon size={17} />
              {label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <ShieldCheck size={16} />
          <span>
            Deterministic tools
            <br />
            <b>LLM-ready architecture</b>
          </span>
        </div>
      </aside>
      <main className="main">
        <header>
          <div>
            <p className="eyebrow">AUTONOMOUS DATA ENGINEERING</p>
            <h1>
              {view === "overview"
                ? "Control center"
                : view === "pipeline"
                  ? "Pipeline monitor"
                  : view === "logs"
                    ? "Agent decision log"
                    : view === "analytics"
                      ? "Ask your data"
                      : "Dataset registry"}
            </h1>
          </div>
          <div className="header-actions">
            <button
              className="icon-button"
              onClick={() => refresh()}
              title="Refresh"
            >
              <RefreshCw size={17} />
            </button>
            <label className="upload-button">
              <UploadCloud size={17} /> Upload source
              <input
                type="file"
                accept=".csv,.xlsx,.xls,.json,.pdf"
                onChange={(e) =>
                  e.target.files?.[0] && upload(e.target.files[0])
                }
              />
            </label>
          </div>
        </header>
        {view === "overview" && (
          <>
            <section className="hero">
              <div>
                <span className="kicker">
                  AUTONOMY STATUS <i />
                </span>
                <h2>
                  From raw signal
                  <br />
                  <em>to trusted insight.</em>
                </h2>
                <p>
                  Upload a source and let specialized agents observe, plan,
                  execute, and validate the path to usable data.
                </p>
              </div>
              <div className="hero-visual">
                <div className="orbit-line" />
                <div className="core">
                  <Sparkles size={28} />
                  <span>
                    AI
                    <br />
                    ENGINE
                  </span>
                </div>
                <div className="orbit-label label-a">OBSERVE</div>
                <div className="orbit-label label-b">ACT</div>
                <div className="orbit-label label-c">VALIDATE</div>
              </div>
            </section>
            <section className="metrics">
              <Metric
                label="Datasets"
                value={datasets.length}
                icon={Database}
              />
              <Metric
                label="Records indexed"
                value={datasets
                  .reduce((n, d) => n + (d.rows || 0), 0)
                  .toLocaleString()}
                icon={BarChart3}
              />
              <Metric
                label="Pipelines complete"
                value={run?.status === "SUCCESS" ? 1 : 0}
                icon={CheckCircle2}
              />
              <Metric
                label="Latest quality"
                value={`${score}/100`}
                icon={Gauge}
                accent
              />
            </section>
            <section className="work-grid">
              <div className="panel dataset-panel">
                <div className="panel-head">
                  <div>
                    <span className="section-label">ACTIVE SOURCES</span>
                    <h3>Data registry</h3>
                  </div>
                  <button
                    className="text-button"
                    onClick={() => setView("datasets")}
                  >
                    View all <ArrowUpRight size={15} />
                  </button>
                </div>
                {datasets.length === 0 ? (
                  <Empty onUpload={upload} />
                ) : (
                  datasets.map((d) => (
                    <DatasetRow
                      key={d.id}
                      dataset={d}
                      selected={selected?.id === d.id}
                      onClick={() => setSelected(d)}
                    />
                  ))
                )}
              </div>
              <div className="panel quality-panel">
                <div className="panel-head">
                  <div>
                    <span className="section-label">QUALITY LIFT</span>
                    <h3>Signal health</h3>
                  </div>
                  <span className="live-dot">LIVE</span>
                </div>
                <div className="quality-numbers">
                  <div>
                    <small>BEFORE</small>
                    <strong>{selected?.quality_before?.overall ?? "--"}</strong>
                  </div>
                  <div className="arrow">→</div>
                  <div className="after">
                    <small>AFTER</small>
                    <strong>{selected?.quality_after?.overall ?? "--"}</strong>
                  </div>
                </div>
                <div className="chart">
                  <ResponsiveContainer width="100%" height={130}>
                    <AreaChart
                      data={[
                        {
                          name: "Before",
                          score: selected?.quality_before?.overall || 0,
                        },
                        {
                          name: "After",
                          score: selected?.quality_after?.overall || score,
                        },
                      ]}
                    >
                      <defs>
                        <linearGradient id="lift" x1="0" x2="0" y1="0" y2="1">
                          <stop
                            offset="0%"
                            stopColor="#d7f36b"
                            stopOpacity=".45"
                          />
                          <stop
                            offset="100%"
                            stopColor="#d7f36b"
                            stopOpacity="0"
                          />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="name" hide />
                      <YAxis hide domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{
                          background: "#17231e",
                          border: "0",
                          color: "#fff",
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="score"
                        stroke="#d7f36b"
                        fill="url(#lift)"
                        strokeWidth={3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <p className="muted">
                  Score is calculated from completeness, consistency, validity,
                  and uniqueness.
                </p>
              </div>
            </section>
          </>
        )}
        {view === "datasets" && (
          <section className="panel full-panel">
            <div className="panel-head">
              <div>
                <span className="section-label">SOURCE INVENTORY</span>
                <h3>Uploaded datasets</h3>
              </div>
            </div>
            {datasets.map((d) => (
              <DatasetRow
                key={d.id}
                dataset={d}
                selected={selected?.id === d.id}
                onClick={() => setSelected(d)}
              />
            ))}
          </section>
        )}
        {view === "pipeline" && (
          <Pipeline
            run={run}
            selected={selected}
            setSelected={setSelected}
            datasets={datasets}
            execute={execute}
            busy={busy}
            pipelineRunning={pipelineRunning}
            activeStageIndex={activeStageIndex}
            modelType={modelType}
            setModelType={setModelType}
            targetColumn={targetColumn}
            setTargetColumn={setTargetColumn}
            datasetColumns={datasetColumns}
          />
        )}{" "}
        {view === "logs" && <Logs logs={logs} />}{" "}
        {view === "benchmark" && <Benchmark run={run} selected={selected} />}
        {view === "analytics" && (
          <Analytics
            question={question}
            setQuestion={setQuestion}
            ask={ask}
            answer={answer}
            busy={busy}
          />
        )}
      </main>
    </div>
  );
}
function Metric({ label, value, icon: Icon, accent }: any) {
  return (
    <div className="metric">
      <div className={accent ? "metric-icon accent" : "metric-icon"}>
        <Icon size={18} />
      </div>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
      </div>
    </div>
  );
}
function DatasetRow({ dataset, selected, onClick }: any) {
  return (
    <button
      className={selected ? "dataset-row selected" : "dataset-row"}
      onClick={onClick}
    >
      <div className="file-icon">
        <Database size={18} />
      </div>
      <div className="dataset-name">
        <strong>{dataset.name}</strong>
        <span>
          {dataset.source_type} · {dataset.columns} columns
        </span>
      </div>
      <span className="row-count">
        {(dataset.rows || 0).toLocaleString()} rows
      </span>
      <span className={`status ${dataset.status === "READY" ? "good" : ""}`}>
        {dataset.status}
      </span>
      <ArrowUpRight size={15} />
    </button>
  );
}
function Empty({ onUpload }: any) {
  return (
    <label className="empty">
      <FileUp size={24} />
      <strong>Drop your first source here</strong>
      <span>CSV, Excel, JSON, or PDF</span>
      <input
        type="file"
        accept=".csv,.xlsx,.xls,.json,.pdf"
        onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
      />
    </label>
  );
}
function Pipeline({
  run,
  selected,
  setSelected,
  datasets,
  execute,
  busy,
  pipelineRunning,
  activeStageIndex,
  modelType,
  setModelType,
  targetColumn,
  setTargetColumn,
  datasetColumns,
}: any) {
  const download = () => {
    if (!selected) return;
    const link = document.createElement("a");
    link.href = `${API_BASE_URL}/api/datasets/${selected.id}/download`;
    link.download = `${selected.name.replace(/\.[^/.]+$/, "")}_processed.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const totalStages = stages.length;
  const actualCompleted = run?.stages?.length ?? 0;
  const progressPercent = pipelineRunning
    ? ((Math.min(activeStageIndex + 1, totalStages) / totalStages) * 100)
    : run
      ? (Math.min(actualCompleted, totalStages) / totalStages) * 100
      : 0;

  const getPolicySummary = (type: string) => {
    switch (type) {
      case "random_forest":
        return {
          title: "TREE-BASED OPTIMIZATION POLICY (RANDOM FOREST)",
          desc: "Scale-invariant: skips numeric scaling to preserve natural feature boundaries; applies Ordinal Encoding to categoricals to prevent sparse dimensionality explosion in tree splits.",
        };
      case "xgboost":
        return {
          title: "GRADIENT BOOSTING POLICY (XGBOOST)",
          desc: "Iterative gradient-partitioned: preserves raw numeric continuous distributions; encodes categorical levels ordinally with native NaN-routing preservation.",
        };
      case "logistic_regression":
        return {
          title: "LINEAR HYPERPLANE POLICY (LOGISTIC / RIDGE)",
          desc: "Requires continuous normalized feature space: applies outlier clipping (1st/99th percentiles) to neutralize high leverage points, StandardScaler (zero-mean unit-variance), and One-Hot Encoding.",
        };
      case "knn":
        return {
          title: "DISTANCE-METRIC POLICY (K-NEAREST NEIGHBORS)",
          desc: "Euclidean/Minkowski distance sensitive: strictly applies MinMax feature normalization [0, 1] so high-magnitude columns do not dominate distance calculations.",
        };
      default:
        return {
          title: "GENERIC DATA ENGINEERING POLICY",
          desc: "Standard automated cleaning: handles empty rows/cols, deduplication, median/mode imputation, string and date ISO standardization.",
        };
    }
  };

  const policy = getPolicySummary(modelType);

  return (
    <section className="pipeline-layout">
      {/* 1. Full-Width Top Bar: Dataset Dropdown & Spacious Action Buttons */}
      <div className="pipeline-topbar">
        <div className="dataset-select-section">
          <Database size={18} style={{ color: "#d7f36b", flexShrink: 0 }} />
          <span className="dataset-select-label">ACTIVE DATASET:</span>
          <select
            className="dataset-select-dropdown"
            value={selected?.id || ""}
            onChange={(e) => {
              const target = datasets.find((d: any) => d.id === e.target.value);
              if (target) setSelected(target);
            }}
            disabled={busy || pipelineRunning}
          >
            {datasets.length === 0 ? (
              <option value="">No datasets uploaded yet</option>
            ) : (
              datasets.map((d: any) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.source_type} · {(d.rows || 0).toLocaleString()} rows · {d.columns} cols)
                </option>
              ))
            )}
          </select>
        </div>

        <div className="pipeline-actions">
          <button
            className="run-button"
            disabled={!selected || busy}
            onClick={execute}
          >
            <Play size={15} />{" "}
            {busy ? "Running..." : "Run autonomous pipeline"}
          </button>
          {selected?.status === "READY" && (
            <button
              className="secondary-download-btn"
              onClick={download}
              title="Download processed CSV"
            >
              <Download size={15} /> Download CSV
            </button>
          )}
        </div>
      </div>

      {/* 2. Symmetrical 50/50 Balanced Two-Column Layout */}
      <div className="pipeline-columns-grid">
        {/* Left Column: Model Configuration & Execution Graph Stages */}
        <div className="pipeline-column">
          <div className="model-config-card">
            <span className="section-label">RESEARCH NOVELTY</span>
            <h4 style={{ margin: "4px 0 0", fontSize: 14, color: "#d7f36b" }}>
              Target-Model-Aware Autonomous Feature Engineering
            </h4>

            <div className="model-config-grid">
              <div className="model-field">
                <label>Target Machine Learning Model</label>
                <select
                  className="model-select"
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  disabled={busy}
                >
                  <option value="generic">Generic Pipeline (Standard Cleaning)</option>
                  <option value="random_forest">Random Forest (Scale-Invariant, Ordinal)</option>
                  <option value="xgboost">Gradient Boosting / XGBoost (Tree-Splits)</option>
                  <option value="logistic_regression">Logistic / Linear Regression (StandardScaler + Outlier Clipping)</option>
                  <option value="knn">K-Nearest Neighbors (MinMax [0, 1] Normalization)</option>
                </select>
              </div>
              {modelType !== "generic" && (
                <div className="model-field">
                  <label>Target Predictor / Label Column</label>
                  <select
                    className="model-select"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    disabled={busy}
                  >
                    {datasetColumns.length === 0 ? (
                      <option value="">(Upload dataset to load columns)</option>
                    ) : (
                      datasetColumns.map((col: string) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))
                    )}
                  </select>
                </div>
              )}
            </div>
            <div className="model-policy-callout">
              <strong>{policy.title}</strong>
              {policy.desc}
            </div>
          </div>

          <div className="panel" style={{ padding: "18px 22px" }}>
            <span className="section-label">EXECUTION GRAPH</span>
            <h3 style={{ margin: "4px 0 0", fontSize: 15 }}>Autonomous Pipeline Stages</h3>

            <div className="run-progress-panel">
              <div className="progress-header">
                <span>Pipeline progress</span>
                <strong>{Math.round(progressPercent)}%</strong>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
              </div>
              <div className="progress-meta">
                <span>{pipelineRunning ? "Agents are active" : run ? "Finished" : "Waiting to start"}</span>
                <span>{run?.status || "PENDING"}</span>
              </div>
            </div>

            <div className="stage-timeline">
              {stages.map((name: string, i: number) => {
                const matchingStage = run?.stages?.find((stage: any) => {
                  const stageName = String(stage.name || "").toLowerCase();
                  return stageName === name.toLowerCase() || stageName.includes(name.split(" ")[0].toLowerCase());
                });
                const complete = Boolean(matchingStage);
                const active = pipelineRunning && i === activeStageIndex;
                const passed = pipelineRunning && i < activeStageIndex;
                return (
                  <div className={`timeline-node ${complete ? "done" : ""} ${active ? "active" : ""} ${passed ? "passed" : ""}`} key={name}>
                    <span className="timeline-dot">
                      {complete ? <CheckCircle2 size={12} /> : active ? <LoaderCircle className="stage-spinner" size={12} /> : <span>{i + 1}</span>}
                    </span>
                    <small>{name}</small>
                  </div>
                );
              })}
            </div>

            <div className="stage-grid">
              {stages.map((name: string, i: number) => {
                const complete = run?.stages?.some(
                  (stage: any) =>
                    stage.name.toLowerCase() === name.toLowerCase() ||
                    stage.name
                      .toLowerCase()
                      .includes(name.split(" ")[0].toLowerCase()),
                );
                const active = pipelineRunning && i === activeStageIndex;
                const passed = pipelineRunning && i < activeStageIndex;
                return (
                  <div
                    className={`stage-card ${complete ? "complete" : ""} ${active ? "stage-active" : ""}`}
                    key={name}
                  >
                    <div className="stage-card-marker">
                      {complete ? (
                        <CheckCircle2 size={15} />
                      ) : active ? (
                        <LoaderCircle className="stage-spinner" size={15} />
                      ) : passed ? (
                        <CheckCircle2 size={15} />
                      ) : (
                        <span>{String(i + 1).padStart(2, "0")}</span>
                      )}
                    </div>
                    <div className="stage-card-text">
                      <strong>{name}</strong>
                      <span>
                        {complete
                          ? "Completed"
                          : active
                            ? "Agent active..."
                            : passed
                              ? "Done"
                              : "Pending"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Empirical Benchmark & Bounded Adaptive Plan (or Standby Monitor) */}
        <div className="pipeline-column">
          {run?.model_benchmark && (
            <div className="panel benchmark-card">
              <div className="benchmark-header">
                <div>
                  <span className="section-label">EMPIRICAL RESEARCH BENCHMARK</span>
                  <h3 style={{ margin: "4px 0 0", color: "#e8eee8" }}>
                    {run.model_benchmark.model_name}
                  </h3>
                </div>
                <span className="benchmark-pill">
                  +{run.model_benchmark.lift}% RESEARCH LIFT
                </span>
              </div>

              <div className="benchmark-scores">
                <div className="benchmark-col">
                  <small>BASELINE SCORE</small>
                  <strong>{run.model_benchmark.baseline_score}%</strong>
                </div>
                <div className="benchmark-arrow">→</div>
                <div className="benchmark-col highlight">
                  <small>MODEL-AWARE SCORE</small>
                  <strong>{run.model_benchmark.model_aware_score}%</strong>
                </div>
              </div>

              <div className="benchmark-meta">
                <div>
                  <b>Target Column:</b> <code>{run.model_benchmark.target_column}</code> (
                  {run.model_benchmark.task})
                </div>
                <div>
                  <b>Evaluation Metric:</b> {run.model_benchmark.metric_name}
                </div>
                <div style={{ marginTop: 6 }}>
                  <b>Algorithmic Strategy:</b> {run.model_benchmark.policy_description}
                </div>
              </div>
            </div>
          )}

          {run ? (
            <div className="panel plan-card">
              <span className="section-label">ADAPTIVE PLAN</span>
              <h3 style={{ margin: "4px 0 12px", fontSize: 15 }}>
                Chosen from observed data issues & target model
              </h3>
              <div className="plan-card-scrollable">
                {run.plan.map((p: any) => (
                  <div className="plan-step" key={p.operation}>
                    <CheckCircle2 size={15} />
                    <div>
                      <strong>{p.operation.replaceAll("_", " ")}</strong>
                      <span>{p.reason}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="standby-card">
              <div className="standby-icon">
                <Sparkles size={24} />
              </div>
              <h3>Pipeline Ready on Standby</h3>
              <p>
                Select your target machine learning model and predictor column, then launch the autonomous pipeline to observe dynamic multi-agent planning and model-aware optimizations in real time.
              </p>
              <div className="standby-badges">
                <span className="standby-badge">Dynamic Heuristic Planning</span>
                <span className="standby-badge">Model-Aware Feature Engineering</span>
                <span className="standby-badge">Automated Empirical Benchmarking</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
function Logs({ logs }: any) {
  return (
    <section className="panel full-panel">
      <div className="panel-head">
        <div>
          <span className="section-label">OBSERVABILITY</span>
          <h3>Agent decisions</h3>
        </div>
        <span className="live-dot">{logs.length} EVENTS</span>
      </div>
      {logs.length === 0 ? (
        <p className="muted">Run a pipeline to see real agent activity.</p>
      ) : (
        logs.map((log: any) => (
          <div className="log-row" key={log.id}>
            <div className="log-agent">
              <Bot size={16} />
              <strong>{log.agent}</strong>
            </div>
            <div className="log-action">
              <strong>{log.action.replaceAll("_", " ")}</strong>
              <span>{log.reason}</span>
              {log.output && <small style={{ display: "block", marginTop: 6, color: "#b9c9bb", lineHeight: 1.6 }}>{log.output}</small>}
            </div>
            <code>{log.tool}</code>
            <span className="status good">{log.status}</span>
          </div>
        ))
      )}
    </section>
  );
}
function Benchmark({ run, selected }: any) {
  const benchmark = run?.model_benchmark;

  if (!benchmark) {
    return (
      <section className="panel full-panel">
        <div className="panel-head">
          <div>
            <span className="section-label">MODEL BENCHMARK</span>
            <h3>No benchmark available yet</h3>
          </div>
        </div>
        <p className="muted" style={{ padding: "12px 0 0" }}>
          Run a pipeline with a model type and target column first to generate benchmark metrics.
        </p>
      </section>
    );
  }

  return (
    <section className="panel full-panel">
      <div className="panel-head">
        <div>
          <span className="section-label">MODEL BENCHMARK</span>
          <h3>{benchmark.model_name}</h3>
        </div>
        <span className="benchmark-pill">+{benchmark.lift}% RESEARCH LIFT</span>
      </div>

      <div className="benchmark-scores" style={{ marginTop: 16 }}>
        <div className="benchmark-col">
          <small>BASELINE SCORE</small>
          <strong>{benchmark.baseline_score}%</strong>
        </div>
        <div className="benchmark-arrow">→</div>
        <div className="benchmark-col highlight">
          <small>MODEL-AWARE SCORE</small>
          <strong>{benchmark.model_aware_score}%</strong>
        </div>
      </div>

      <div className="work-grid" style={{ marginTop: 18 }}>
        <div className="panel benchmark-card">
          <div className="benchmark-meta">
            <div><b>Dataset:</b> {selected?.name || "Current dataset"}</div>
            <div><b>Target Column:</b> {benchmark.target_column}</div>
            <div><b>Task:</b> {benchmark.task}</div>
            <div><b>Metric:</b> {benchmark.metric_name}</div>
            <div><b>Model Type:</b> {benchmark.model_type || "N/A"}</div>
            <div><b>Strategy:</b> {benchmark.policy_description}</div>
          </div>
        </div>

        <div className="panel benchmark-card">
          <div className="benchmark-meta">
            <div><b>Baseline:</b> {benchmark.baseline_score}%</div>
            <div><b>Model-Aware:</b> {benchmark.model_aware_score}%</div>
            <div><b>Lift:</b> +{benchmark.lift}%</div>
            <div><b>Winner:</b> {benchmark.model_name}</div>
            <div><b>Pipeline status:</b> {run?.status || "SUCCESS"}</div>
            <div><b>Last evaluation:</b> {new Date().toLocaleString()}</div>
          </div>
        </div>
      </div>

      {Array.isArray(run?.plan) && run.plan.length > 0 && (
        <div style={{ marginTop: 22 }}>
          <span className="section-label">ADAPTIVE PLAN</span>
          <div className="plan-card-scrollable" style={{ maxHeight: 220 }}>
            {run.plan.map((p: any) => (
              <div className="plan-step" key={p.operation}>
                <CheckCircle2 size={15} />
                <div>
                  <strong>{p.operation.replaceAll("_", " ")}</strong>
                  <span>{p.reason}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function Analytics({ question, setQuestion, ask, answer, busy }: any) {
  return (
    <section className="analytics-view">
      <div className="analytics-intro">
        <span className="kicker">
          NATURAL LANGUAGE ANALYTICS <i />
        </span>
        <h2>
          Ask the
          <br />
          <em>warehouse.</em>
        </h2>
        <p>
          Route questions to structured data or document intelligence. Demo mode
          returns grounded dataset evidence without an API key.
        </p>
      </div>
      <div className="panel query-panel">
        <div className="query-input">
          <Search size={18} />
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="Which region had the highest revenue?"
          />
          <button onClick={ask} disabled={busy}>
            {busy ? "..." : "Ask"}
          </button>
        </div>
        {answer && (
          <div className="answer">
            <span className="section-label">{answer.route}</span>
            <h3>{answer.answer}</h3>
            {answer.sql && <pre>{answer.sql}</pre>}
            {Array.isArray(answer.sources) && answer.sources.length > 0 && (
              <div className="answer-evidence">
                <strong>RAG context</strong>
                <div className="evidence-table">
                  {answer.sources.slice(0, 3).map((source: any, index: number) => (
                    <div key={`${source.title || 'source'}-${index}`} className="evidence-row">
                      <span><b>{source.title || 'Source'}:</b> {source.snippet || source.content || source.answer || ''}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {Array.isArray(answer.evidence) && answer.evidence.length > 0 && (
              <div className="answer-evidence">
                <strong>Evidence</strong>
                <div className="evidence-table" style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 8 }}>
                    <thead>
                      <tr>
                        {(answer.columns && answer.columns.length ? answer.columns : Object.keys(answer.evidence[0] || {})).map((column: string) => (
                          <th key={column} style={{ textAlign: "left", padding: "8px 10px", borderBottom: "1px solid #31463d", color: "#d7f36b" }}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {answer.evidence.slice(0, 10).map((row: any, index: number) => {
                        const keys = answer.columns && answer.columns.length ? answer.columns : Object.keys(row || {});
                        return (
                          <tr key={`${JSON.stringify(row)}-${index}`}>
                            {keys.map((key: string) => (
                              <td key={`${key}-${index}`} style={{ padding: "8px 10px", borderBottom: "1px solid #1c2c26", verticalAlign: "top" }}>
                                {String(row?.[key] ?? "") || "—"}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
export default App;
