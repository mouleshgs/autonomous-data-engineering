import { useEffect, useState } from "react";
import "./pipeline.css";
import axios from "axios";
import {
  Activity,
  AlertTriangle,
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
  Trash2,
  UploadCloud,
  X,
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
const STAGE_METADATA = [
  {
    name: "Source Analyzer",
    tagline: "Format & Schema Detection",
    detail: "Parsing source headers, inferring preliminary types & encoding",
    agent: "AnalyzerAgent",
  },
  {
    name: "Profiler Agent",
    tagline: "Statistical Profiling",
    detail: "Scanning nulls, duplicates, cardinality & numeric ranges",
    agent: "ProfilerAgent",
  },
  {
    name: "Planner Agent",
    tagline: "Adaptive Heuristic Planning",
    detail: "Synthesizing safe execution operations from observed anomalies",
    agent: "PlannerAgent",
  },
  {
    name: "Ingestion",
    tagline: "Type Casting & Memory Buffer",
    detail: "Materializing structured dataframe & immutable runtime registry",
    agent: "IngestionWorker",
  },
  {
    name: "Cleaning",
    tagline: "Deduplication & Imputation",
    detail: "Executing median/mode imputation, negative fixes & deduplication",
    agent: "CleaningAgent",
  },
  {
    name: "Transformation",
    tagline: "Model-Aware Feature Eng.",
    detail: "Applying target-aware scaling, ordinal encoding & ISO standardization",
    agent: "TransformAgent",
  },
  {
    name: "Validation",
    tagline: "Quality Gate & Metric Formula",
    detail: "Computing completeness, consistency, validity & uniqueness lift",
    agent: "ValidatorAgent",
  },
  {
    name: "Storage",
    tagline: "Artifact Materialization",
    detail: "Persisting cleaned production CSV and audit lineage artifacts",
    agent: "StorageWorker",
  },
  {
    name: "Model Benchmark",
    tagline: "Downstream ML Evaluation",
    detail: "Training baseline vs model-aware model & computing research lift",
    agent: "BenchmarkAgent",
  },
];
const stages = STAGE_METADATA.map((s) => s.name);

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
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const response = await api.get("/datasets");
      setDatasets(response.data);
      if (!selected && response.data[0]) setSelected(response.data[0]);
      const logResponse = await api.get("/agents/logs");
      setLogs(logResponse.data);
    } catch (err: any) {
      console.error("Refresh error:", err);
    }
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

  const deleteDataset = async (datasetId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this dataset?")) return;
    setBusy(true);
    setError(null);
    try {
      await api.delete(`/datasets/${datasetId}`);
      if (selected?.id === datasetId) {
        setSelected(null);
        setRun(null);
      }
      await refresh();
    } catch (err: any) {
      console.error("Delete error:", err);
      const detail =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        "Failed to delete dataset.";
      setError(`Delete Failed: ${detail}`);
    } finally {
      setBusy(false);
    }
  };

  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await api.post("/datasets/upload", body);
      setSelected(response.data);
      await refresh();
    } catch (err: any) {
      console.error("Upload error:", err);
      const detail =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        "Upload failed. Ensure backend is running and file is CSV, Excel, JSON, or PDF.";
      setError(`Upload Failed: ${detail}`);
    } finally {
      setBusy(false);
    }
  };

  const execute = async () => {
    if (!selected || busy || pipelineRunning) return;
    setBusy(true);
    setPipelineRunning(true);
    setActiveStageIndex(0);
    setRun(null);
    setError(null);

    // Launch backend request in parallel
    const runPromise = api.post(`/pipelines/${selected.id}/run`, {
      model_type: modelType,
      target_column: modelType !== "generic" ? targetColumn : undefined,
    });

    let runResult: any = null;
    let runError: any = null;

    runPromise
      .then((res) => {
        runResult = res.data;
      })
      .catch((err) => {
        runError = err;
      });

    try {
      // Step through each agent stage one-by-one with deliberate, clearly visible pacing (~1.2s per stage)
      for (let i = 0; i < stages.length; i++) {
        if (runError) throw runError;
        setActiveStageIndex(i);
        await new Promise((resolve) => setTimeout(resolve, 1200));
      }

      // If backend is still running (e.g. larger file or model download), wait for it
      if (!runResult) {
        const response = await runPromise;
        runResult = response.data;
      }

      // If background run dispatched, poll until finished
      if (runResult.status !== "SUCCESS" && runResult.status !== "FAILED") {
        while (runResult.status !== "SUCCESS" && runResult.status !== "FAILED") {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          const latest = await api.get(`/pipelines/${runResult.id}`);
          runResult = latest.data;
        }
      }

      if (runResult.status === "FAILED") {
        throw new Error(runResult.error || "Pipeline execution failed in agent runner");
      }

      setRun(runResult);
      setPipelineRunning(false);
      setBusy(false);

      try {
        await refresh();
        const updated = await api.get(`/datasets/${selected.id}`);
        setSelected(updated.data);
      } catch (refreshErr) {
        console.warn("Post-pipeline refresh warning:", refreshErr);
      }
    } catch (err: any) {
      console.error("Pipeline run error:", err);
      const status = err.response?.status ? `(HTTP ${err.response.status}) ` : "";
      const detail =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        (err.message === "Network Error"
          ? "Network Error: Backend connection failed or dropped."
          : err.message) ||
        "Pipeline execution paused or failed.";
      setError(`Pipeline Failed: ${status}${detail}`);
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

  const ask = async (overrideQuestion?: string) => {
    const q = (overrideQuestion ?? question).trim();
    if (!q) return;
    if (overrideQuestion) {
      setQuestion(overrideQuestion);
    }
    setBusy(true);
    try {
      const response = await api.post("/analytics/query", { question: q });
      setAnswer(response.data);
    } catch (err: any) {
      console.error("Analytics query error:", err);
    } finally {
      setBusy(false);
    }
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
        {error && (
          <div className="error-banner">
            <div className="error-banner-content">
              <AlertTriangle size={18} className="error-icon" />
              <span>{error}</span>
            </div>
            <button
              className="error-close-btn"
              onClick={() => setError(null)}
              title="Dismiss error"
            >
              <X size={16} />
            </button>
          </div>
        )}
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
                      onDelete={deleteDataset}
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
                onDelete={deleteDataset}
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
            deleteDataset={deleteDataset}
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
function DatasetRow({ dataset, selected, onClick, onDelete }: any) {
  return (
    <div
      className={selected ? "dataset-row selected" : "dataset-row"}
      onClick={onClick}
      role="button"
      tabIndex={0}
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
      {onDelete && (
        <button
          className="delete-dataset-btn"
          onClick={(e) => onDelete(dataset.id, e)}
          title="Delete dataset"
        >
          <Trash2 size={15} />
        </button>
      )}
      <ArrowUpRight size={15} />
    </div>
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
  deleteDataset,
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

  const totalStages = STAGE_METADATA.length;
  const progressPercent = pipelineRunning
    ? (((activeStageIndex + 1) / totalStages) * 100)
    : run?.status === "SUCCESS"
      ? 100
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
          {selected && (
            <button
              className="delete-dataset-btn"
              onClick={(e) => deleteDataset(selected.id, e)}
              title="Delete active dataset"
              disabled={busy || pipelineRunning}
            >
              <Trash2 size={16} />
            </button>
          )}
        </div>

        <div className="pipeline-actions">
          <button
            className="run-button"
            disabled={!selected || busy || pipelineRunning}
            onClick={execute}
          >
            {pipelineRunning ? (
              <>
                <LoaderCircle size={15} className="stage-spinner" /> Running agents...
              </>
            ) : busy ? (
              <>
                <LoaderCircle size={15} className="stage-spinner" /> Processing...
              </>
            ) : (
              <>
                <Play size={15} /> Run autonomous pipeline
              </>
            )}
          </button>
          {selected?.status === "READY" && !pipelineRunning && (
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
                  disabled={busy || pipelineRunning}
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
                    disabled={busy || pipelineRunning}
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
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <span className="section-label">MULTI-AGENT EXECUTION GRAPH</span>
                <h3 style={{ margin: "4px 0 0", fontSize: 15 }}>Autonomous Pipeline Flow</h3>
              </div>
              {pipelineRunning && (
                <span className="live-flow-badge">
                  <span className="live-dot-pulse" /> FLOW IN PROGRESS
                </span>
              )}
            </div>

            <div className="run-progress-panel">
              <div className="progress-header">
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span>Pipeline Execution Progress</span>
                  {pipelineRunning && <span className="mini-pulse" />}
                </div>
                <strong>{Math.round(progressPercent)}%</strong>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
              </div>
              <div className="progress-meta">
                <span>
                  {pipelineRunning
                    ? `Active Agent: ${STAGE_METADATA[activeStageIndex].name} (${activeStageIndex + 1}/${totalStages})`
                    : run?.status === "SUCCESS"
                    ? "All 9 Agents Completed Successfully"
                    : "Ready on standby"}
                </span>
                <span className={pipelineRunning ? "status-running-text" : ""}>
                  {pipelineRunning ? "EXECUTING" : run?.status || "STANDBY"}
                </span>
              </div>
            </div>

            {/* Active Agent Live HUD Callout */}
            {pipelineRunning && (
              <div className="active-agent-banner">
                <div className="active-agent-header">
                  <span className="active-agent-pulse" />
                  <strong>ACTIVE AGENT: {STAGE_METADATA[activeStageIndex].name.toUpperCase()}</strong>
                  <span className="active-agent-step">STAGE {activeStageIndex + 1} OF {totalStages}</span>
                </div>
                <p className="active-agent-desc">{STAGE_METADATA[activeStageIndex].detail}</p>
              </div>
            )}

            {/* Step-by-Step Flow Nodes */}
            <div className="stage-timeline">
              {STAGE_METADATA.map((stageItem: any, i: number) => {
                const isComplete = (!pipelineRunning && run?.status === "SUCCESS") || (pipelineRunning && i < activeStageIndex);
                const isActive = pipelineRunning && i === activeStageIndex;
                return (
                  <div
                    className={`timeline-node ${isComplete ? "done" : ""} ${isActive ? "active" : ""}`}
                    key={stageItem.name}
                  >
                    <span className="timeline-dot">
                      {isComplete ? (
                        <CheckCircle2 size={12} />
                      ) : isActive ? (
                        <LoaderCircle className="stage-spinner" size={12} />
                      ) : (
                        <span>{i + 1}</span>
                      )}
                    </span>
                    <small>{stageItem.name}</small>
                  </div>
                );
              })}
            </div>

            {/* 3x3 Symmetrical Agent Cards with Glowing Transitions */}
            <div className="stage-grid">
              {STAGE_METADATA.map((stageItem: any, i: number) => {
                const isComplete = (!pipelineRunning && run?.status === "SUCCESS") || (pipelineRunning && i < activeStageIndex);
                const isActive = pipelineRunning && i === activeStageIndex;

                return (
                  <div
                    className={`stage-card ${isActive ? "stage-active" : isComplete ? "complete" : "pending"}`}
                    key={stageItem.name}
                  >
                    <div className="stage-card-marker">
                      {isComplete ? (
                        <CheckCircle2 size={15} />
                      ) : isActive ? (
                        <LoaderCircle className="stage-spinner" size={15} />
                      ) : (
                        <span>{String(i + 1).padStart(2, "0")}</span>
                      )}
                    </div>
                    <div className="stage-card-text">
                      <strong>{stageItem.name}</strong>
                      <small className="stage-tagline">{stageItem.tagline}</small>
                      <span className="stage-status-label">
                        {isComplete
                          ? "Completed"
                          : isActive
                            ? "Agent running..."
                            : "Queued"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Live Stream Telemetry while Running, or Empirical Benchmark & Adaptive Plan when Done */}
        <div className="pipeline-column">
          {pipelineRunning ? (
            <div className="panel stream-card">
              <div className="stream-header">
                <div>
                  <span className="section-label">LIVE AGENT TELEMETRY</span>
                  <h3 style={{ margin: "4px 0 0", fontSize: 15, color: "#d7f36b" }}>
                    Multi-Agent Execution Stream
                  </h3>
                </div>
                <span className="live-stream-badge">
                  <span className="live-dot-pulse" /> LIVE STREAM
                </span>
              </div>

              <div className="stream-feed">
                {STAGE_METADATA.map((s, idx) => {
                  const isPassed = idx < activeStageIndex;
                  const isActive = idx === activeStageIndex;
                  return (
                    <div
                      key={s.name}
                      className={`stream-event ${
                        isActive ? "event-active" : isPassed ? "event-passed" : "event-waiting"
                      }`}
                    >
                      <div className="stream-icon-col">
                        {isPassed ? (
                          <CheckCircle2 size={14} className="icon-done" />
                        ) : isActive ? (
                          <LoaderCircle size={14} className="stage-spinner icon-active" />
                        ) : (
                          <span className="icon-pending">·</span>
                        )}
                      </div>
                      <div className="stream-content-col">
                        <div className="stream-top-line">
                          <span className="stream-node-name">{s.name}</span>
                          <span className="stream-agent-badge">{s.agent}</span>
                        </div>
                        <p className="stream-action-desc">{s.detail}</p>
                      </div>
                      <div className="stream-state-col">
                        {isPassed ? (
                          <span className="state-badge-done">DONE</span>
                        ) : isActive ? (
                          <span className="state-badge-active">RUNNING</span>
                        ) : (
                          <span className="state-badge-wait">WAIT</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : run?.model_benchmark ? (
            <>
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

              {run.plan && run.plan.length > 0 && (
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
              )}
            </>
          ) : run ? (
            <div className="panel plan-card">
              <span className="section-label">ADAPTIVE PLAN</span>
              <h3 style={{ margin: "4px 0 12px", fontSize: 15 }}>
                Chosen from observed data issues & target model
              </h3>
              <div className="plan-card-scrollable">
                {run.plan?.map((p: any) => (
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
  const suggestions = [
    "Show top 5 rows",
    "Total record and column count",
    "Average of numeric columns",
    "Highest value records",
    "Show bottom 5 rows",
  ];

  return (
    <section className="analytics-view">
      <div className="analytics-header-card">
        <div className="analytics-intro">
          <span className="kicker">
            NATURAL LANGUAGE WAREHOUSE INTELLIGENCE <i />
          </span>
          <h2>
            Ask the <em>warehouse.</em>
          </h2>
          <p>
            Query clean structured datasets and pipeline lineage using natural language.
            Generates grounded SQL execution and evidence tables in real-time.
          </p>
        </div>
      </div>

      <div className="panel query-panel">
        <div className="query-input-wrapper">
          <Search size={18} className="query-search-icon" />
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="Ask a question (e.g. Show top 5 rows, Average of numeric columns...)"
          />
          {question && (
            <button className="query-clear-btn" onClick={() => setQuestion("")} type="button" title="Clear input">
              <X size={15} />
            </button>
          )}
          <button
            className="query-submit-btn"
            onClick={() => ask()}
            disabled={busy || !question.trim()}
            type="button"
          >
            {busy ? <LoaderCircle size={15} className="stage-spinner" /> : "Ask"}
          </button>
        </div>

        <div className="query-suggestions">
          <span className="suggestions-label">Try asking:</span>
          {suggestions.map((s) => (
            <button
              key={s}
              className="suggestion-chip"
              onClick={() => ask(s)}
              type="button"
            >
              {s}
            </button>
          ))}
        </div>

        {answer ? (
          <div className="answer-section">
            <div className="answer-header">
              <span className="route-badge">{answer.route || "HYBRID_SQL_RAG"}</span>
              <span className="answer-timestamp">Grounded Result</span>
            </div>

            <div className="answer-callout">
              <h3>{answer.answer}</h3>
            </div>

            {answer.sql && (
              <div className="sql-box">
                <div className="sql-header">
                  <span>GENERATED SQL QUERY</span>
                </div>
                <pre>{answer.sql}</pre>
              </div>
            )}

            {Array.isArray(answer.sources) && answer.sources.length > 0 && (
              <div className="rag-sources-section">
                <span className="section-label">GROUNDED RAG CONTEXT</span>
                <div className="rag-cards-grid">
                  {answer.sources.slice(0, 3).map((source: any, index: number) => (
                    <div key={`${source.title || 'source'}-${index}`} className="rag-card">
                      <strong>{source.title || 'Knowledge Source'}</strong>
                      <p>{source.snippet || source.content || source.answer || ''}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {Array.isArray(answer.evidence) && answer.evidence.length > 0 && (
              <div className="evidence-section">
                <div className="evidence-header">
                  <span className="section-label">DATASET EVIDENCE</span>
                  <span className="evidence-count">
                    Showing {Math.min(10, answer.evidence.length)} records · {answer.columns?.length || Object.keys(answer.evidence[0] || {}).length} columns
                  </span>
                </div>

                <div className="evidence-table-container">
                  <table className="evidence-data-table">
                    <thead>
                      <tr>
                        {(answer.columns && answer.columns.length ? answer.columns : Object.keys(answer.evidence[0] || {})).map((column: string) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {answer.evidence.slice(0, 10).map((row: any, index: number) => {
                        const keys = answer.columns && answer.columns.length ? answer.columns : Object.keys(row || {});
                        return (
                          <tr key={`${JSON.stringify(row)}-${index}`}>
                            {keys.map((key: string) => (
                              <td key={`${key}-${index}`}>
                                {row?.[key] !== null && row?.[key] !== undefined && row?.[key] !== "" ? String(row[key]) : "—"}
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
        ) : (
          <div className="analytics-standby-hint">
            <Sparkles size={20} style={{ color: "#d7f36b", marginBottom: 8 }} />
            <p>
              Ask any question above or click one of the suggested query chips to see grounded SQL queries and dataset evidence.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
export default App;
