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
      const response = await api.post(`/pipelines/${selected.id}/run`);
      setRun(response.data);
      await refresh();
      setSelected((await api.get(`/datasets/${selected.id}`)).data);
    } finally {
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
        <div className="mode">
          <span className="pulse" /> DEMO MODE <small>LOCAL</small>
        </div>
        <nav>
          {[
            ["overview", "Overview", LayoutDashboard],
            ["datasets", "Datasets", Database],
            ["pipeline", "Pipeline Monitor", Activity],
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
            execute={execute}
            busy={busy}
            pipelineRunning={pipelineRunning}
            activeStageIndex={activeStageIndex}
          />
        )}{" "}
        {view === "logs" && <Logs logs={logs} />}{" "}
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
function Pipeline({ run, selected, execute, busy, pipelineRunning, activeStageIndex }: any) {
  const download = () => {
    if (!selected) return;
    const link = document.createElement("a");
    link.href = `${API_BASE_URL}/api/datasets/${selected.id}/download`;
    link.download = `${selected.name.replace(/\.[^/.]+$/, "")}_processed.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  return (
    <section className="pipeline-view">
      <div className="panel pipeline-card">
        <div className="panel-head">
          <div>
            <span className="section-label">EXECUTION GRAPH</span>
            <h3>{selected?.name || "Select a dataset"}</h3>
          </div>
          <div className="header-actions">
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
                className="run-button"
                onClick={download}
                title="Download processed CSV"
              >
                <Download size={15} /> Download CSV
              </button>
            )}
          </div>
        </div>
        <div className="stage-list">
          {stages.map((name, i) => {
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
                className={`stage ${complete ? "complete" : ""} ${active ? "stage-active" : ""} ${passed ? "stage-passed" : ""}`}
                key={name}
              >
                <div className="stage-marker">
                  {complete ? (
                    <CheckCircle2 size={16} />
                  ) : active ? (
                    <LoaderCircle className="stage-spinner" size={16} />
                  ) : passed ? (
                    <CheckCircle2 size={16} />
                  ) : (
                    <span>{String(i + 1).padStart(2, "0")}</span>
                  )}
                </div>
                <div>
                  <strong>{name}</strong>
                  <span>
                    {complete
                      ? "Completed successfully"
                      : active
                        ? "Agent working..."
                        : passed
                          ? "Step complete"
                      : "Waiting for execution"}
                  </span>
                </div>
                {i < stages.length - 1 && <div className="stage-line" />}
              </div>
            );
          })}
        </div>
      </div>
      {run && (
        <div className="panel plan-card">
          <span className="section-label">ADAPTIVE PLAN</span>
          <h3>Chosen from observed data issues</h3>
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
      )}
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
            </div>
            <code>{log.tool}</code>
            <span className="status good">{log.status}</span>
          </div>
        ))
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
          </div>
        )}
      </div>
    </section>
  );
}
export default App;
