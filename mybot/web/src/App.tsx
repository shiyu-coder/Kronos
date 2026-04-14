import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  AutoComplete,
  Button,
  Card,
  Collapse,
  DatePicker,
  Divider,
  Form,
  InputNumber,
  Layout,
  Row,
  Col,
  Segmented,
  Select,
  Slider,
  Space,
  Spin,
  Statistic,
  Switch,
  Tag,
  Tooltip,
  Typography,
  theme,
  App,
  Progress,
} from "antd";
import { Column, Stock, Line } from "@ant-design/plots";
import {
  BarChartOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  SettingOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import dayjs from "dayjs";
import type { Dayjs } from "dayjs";

const { Header, Content, Footer, Sider } = Layout;
const { Title, Text, Paragraph } = Typography;

type DataMode = "recent" | "range";

interface ApiOptions {
  symbols: string[];
  timeframes: string[];
  models: { id: string; label: string }[];
}

interface OhlcPoint {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ExtraIndicators {
  adx: number;
  macd_histogram: number;
  bb_width_pct: number;
  atr_pct: number;
  stochastic_k: number;
  ema_20: number | null;
  ema_50: number | null;
  ema_200: number | null;
}

interface PredictResponse {
  symbol: string;
  timeframe: string;
  signal: string;
  last_price: number;
  pred_start_price: number;
  pred_end_price: number;
  trend: number;
  rsi: number;
  ema_fast: number;
  ema_slow: number;
  history: OhlcPoint[];
  forecast: OhlcPoint[];
  meta: Record<string, unknown>;
  // --- NEW ensemble / confidence fields ---
  confidence: number;
  pred_mean_price: number;
  pred_lower_price: number;
  pred_upper_price: number;
  trend_lower_pct: number;
  trend_upper_pct: number;
  regime: string;
  regime_confidence: number;
  ensemble_score: number;
  ensemble_direction: "bullish" | "bearish" | "neutral";
  indicators: ExtraIndicators;
  // --- NEW realtime data ---
  realtime?: {
    enabled: boolean;
    sentiment_overall: number;
    sentiment_confidence: number;
    fear_greed: number;
    fear_greed_raw: number;
    news_count: number;
    bullish_count: number;
    bearish_count: number;
    neutral_count: number;
    realtime_signal: string;
    confidence_boost: number;
    confidence_boost_reason: string;
    news_sources: string[];
    macro_events: { event: string; time: string; country: string }[];
    error?: string;
  };
}

interface FormValues {
  symbol: string;
  timeframe: string;
  limit: number;
  since: Dayjs | null;
  range: [Dayjs, Dayjs] | null;
  window_size: number;
  pred_len: number;
  rsi_period: number;
  ema_fast: number;
  ema_slow: number;
  model_id: string;
  temperature: number;
  top_p: number;
  sample_count: number;
  use_ensemble: boolean;
  use_confidence: boolean;
  n_confidence_samples: number;
  use_realtime: boolean;
  realtime_limit: number;
}

const PRESETS: Record<string, Partial<FormValues> & { dataMode?: DataMode }> = {
  btc_daily: {
    symbol: "BTC/USDT",
    timeframe: "1d",
    limit: 600,
    since: null,
    window_size: 200,
    pred_len: 30,
    rsi_period: 14,
    ema_fast: 50,
    ema_slow: 200,
    use_ensemble: true,
    use_confidence: true,
    dataMode: "recent",
  },
  eth_4h: {
    symbol: "ETH/USDT",
    timeframe: "4h",
    limit: 900,
    since: null,
    window_size: 256,
    pred_len: 40,
    rsi_period: 14,
    ema_fast: 21,
    ema_slow: 55,
    use_ensemble: true,
    use_confidence: true,
    dataMode: "recent",
  },
  scalp_5m: {
    symbol: "BTC/USDT",
    timeframe: "5m",
    limit: 1500,
    since: null,
    window_size: 200,
    pred_len: 48,
    rsi_period: 14,
    ema_fast: 12,
    ema_slow: 26,
    use_ensemble: true,
    use_confidence: true,
    dataMode: "recent",
  },
};

function sliceTime(iso: string) {
  return iso.slice(0, 19).replace("T", " ");
}

function disabledSinceDate(current: Dayjs | null) {
  if (!current) return false;
  return !current.endOf("day").isBefore(dayjs().startOf("day"));
}

// ── Regime badge colour map ───────────────────────────────────────────────────
const REGIME_COLOR: Record<string, string> = {
  TRENDING_UP: "green",
  TRENDING_DOWN: "red",
  RANGING: "blue",
  VOLATILE: "orange",
  UNKNOWN: "default",
};

// ── Indicator gauge card ─────────────────────────────────────────────────────
function IndicatorCard({ label, value, unit, warning, color }: {
  label: string; value: number | string; unit?: string; warning?: boolean; color?: string;
}) {
  return (
    <Card size="small" styles={{ body: { padding: "8px 12px" } }}>
      <Text type="secondary" style={{ fontSize: 11, display: "block" }}>{label}</Text>
      <Text strong style={{ fontSize: 15, color: warning ? "#cf1322" : color ?? undefined }}>
        {typeof value === "number" ? value.toFixed(3) : String(value)}{unit ?? ""}
      </Text>
    </Card>
  );
}

function AppContent() {
  const { token } = theme.useToken();
  const { message, notification } = App.useApp();
  const [form] = Form.useForm<FormValues>();
  const temperature = Form.useWatch("temperature", form);
  const topP = Form.useWatch("top_p", form);

  const [dataMode, setDataMode] = useState<DataMode>("recent");
  const [options, setOptions] = useState<ApiOptions | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const loadOptions = useCallback(async () => {
    try {
      const r = await fetch("/api/options");
      if (!r.ok) throw new Error(await r.text());
      setOptions(await r.json());
    } catch {
      message.warning("Could not load /api/options — using built-in defaults.");
    }
  }, [message]);

  useEffect(() => { void loadOptions(); }, [loadOptions]);

  const applyPreset = (key: keyof typeof PRESETS) => {
    const p = PRESETS[key];
    if (p.dataMode) setDataMode(p.dataMode);
    form.setFieldsValue({
      symbol: p.symbol,
      timeframe: p.timeframe,
      limit: p.limit,
      since: p.since ?? null,
      window_size: p.window_size,
      pred_len: p.pred_len,
      rsi_period: p.rsi_period,
      ema_fast: p.ema_fast,
      ema_slow: p.ema_slow,
      use_ensemble: p.use_ensemble ?? true,
      use_confidence: p.use_confidence ?? true,
    });
    message.success("Preset applied");
  };

  const onFinish = async (v: FormValues) => {
    setLoading(true);
    setResult(null);
    try {
      const body: Record<string, unknown> = {
        symbol: (v.symbol || "").trim(),
        timeframe: v.timeframe,
        window_size: v.window_size,
        pred_len: v.pred_len,
        rsi_period: v.rsi_period,
        ema_fast: v.ema_fast,
        ema_slow: v.ema_slow,
        model_id: v.model_id,
        temperature: v.temperature,
        top_p: v.top_p,
        sample_count: v.sample_count,
        max_context: 512,
        use_ensemble: v.use_ensemble,
        use_confidence: v.use_confidence,
        n_confidence_samples: v.n_confidence_samples,
        use_realtime: v.use_realtime,
        realtime_limit: v.realtime_limit,
      };

      if (!body.symbol) {
        message.error("Enter a trading pair (e.g. BTC/USDT)");
        setLoading(false);
        return;
      }

      if (dataMode === "recent") {
        body.limit = v.limit;
        if (v.since) {
          if (!v.since.endOf("day").isBefore(dayjs().startOf("day"))) {
            message.warning("'Since' must be before today — ignored; using latest N candles only.");
          } else {
            body.since_iso = v.since.startOf("day").toISOString();
          }
        }
      } else {
        if (!v.range?.[0] || !v.range?.[1]) {
          message.error("Select a start and end date for the range.");
          setLoading(false);
          return;
        }
        body.range_start_iso = v.range[0].startOf("day").toISOString();
        body.range_end_iso = v.range[1].endOf("day").toISOString();
        body.limit = 5000;
      }

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        let errMsg: string;
        if (typeof detail === "string") errMsg = detail;
        else if (Array.isArray(detail))
          errMsg = detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
        else errMsg = JSON.stringify(data);
        throw new Error(errMsg);
      }
      setResult(data as PredictResponse);
      message.success("Forecast finished.");
    } catch (e) {
      const errText = e instanceof Error ? e.message : "Unknown error";
      notification.error({
        message: "Forecast failed",
        description: errText,
        duration: 10,
        style: { width: 420 },
      });
    } finally {
      setLoading(false);
    }
  };

  // ── Chart data ────────────────────────────────────────────────────────────
  const stockData = useMemo(() => {
    if (!result) return [];
    const row = (h: OhlcPoint, phase: "History" | "Forecast") => ({
      time: sliceTime(h.time),
      open: h.open, high: h.high, low: h.low, close: h.close,
      phase,
    });
    return [
      ...result.history.map(h => row(h, "History")),
      ...result.forecast.map(h => row(h, "Forecast")),
    ];
  }, [result]);

  const volumeData = useMemo(() => {
    if (!result) return [];
    const row = (h: OhlcPoint, phase: "History" | "Forecast") => ({
      time: sliceTime(h.time), volume: h.volume, phase,
    });
    return [
      ...result.history.map(h => row(h, "History")),
      ...result.forecast.map(h => row(h, "Forecast")),
    ];
  }, [result]);

  // Confidence band chart data
  const confidenceData = useMemo(() => {
    if (!result || !result.indicators) return [];
    const { pred_start_price, pred_lower_price, pred_upper_price } = result;
    // Build upper / mean / lower band lines across forecast horizon
    const points: { time: string; price: number; band: "Upper" | "Mean" | "Lower" }[] = [];
    result.forecast.forEach((f, i) => {
      const t = sliceTime(f.time);
      const frac = (i + 1) / result.forecast.length;
      const mid = pred_lower_price + frac * (pred_upper_price - pred_lower_price);
      const spread = mid - pred_start_price;
      points.push({ time: t, price: mid + spread * 0.3, band: "Upper" });
      points.push({ time: t, price: mid,               band: "Mean"  });
      points.push({ time: t, price: mid - spread * 0.3, band: "Lower" });
    });
    return points;
  }, [result]);

  const forecastStartLabel = result?.forecast[0] ? sliceTime(result.forecast[0].time) : null;

  const defaultSymbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"];
  const symbolOptions = (options?.symbols ?? defaultSymbols).map(s => ({ value: s }));

  const stockConfig = useMemo(() => ({
    data: stockData,
    xField: "time",
    yField: ["open", "close", "high", "low"] as [string, string, string, string],
    height: 380,
    autoFit: true,
    animation: { appear: { animation: "fade-in" as const, duration: 400 } },
    axis: {
      x: { title: false, labelAutoRotate: true, labelAutoHide: true, tickCount: 8 },
      y: {
        title: false,
        labelFormatter: (v: string) => Number(v).toLocaleString("en-US", { maximumFractionDigits: 4 }),
      },
    },
    legend: false as const,
    tooltip: { title: (d: { time?: string }) => d?.time ?? "" },
    style: { stroke: token.colorPrimary, lineWidth: 1 },
    lineStyle: { stroke: token.colorTextQuaternary, lineWidth: 1 },
  }), [stockData, token]);

  const volumeConfig = useMemo(() => ({
    data: volumeData,
    xField: "time",
    yField: "volume",
    colorField: "phase",
    height: 180,
    autoFit: true,
    columnWidthRatio: 0.65,
    scale: {
      color: { domain: ["History", "Forecast"], range: [token.colorPrimary, "#ea580c"] },
    },
    axis: {
      x: { labelAutoRotate: true, labelAutoHide: true, tickCount: 8 },
      y: { title: false, labelFormatter: (v: string) => Number(v).toLocaleString("en-US") },
    },
    legend: { position: "top" as const },
    tooltip: { channel: "y", valueFormatter: (v: number) => v.toLocaleString("en-US") },
  }), [volumeData, token]);

  const confidenceConfig = useMemo(() => ({
    data: confidenceData,
    xField: "time",
    yField: "price",
    colorField: "band",
    height: 180,
    autoFit: true,
    point: { size: 0 },
    lineStyle: { lineWidth: 2 },
    scale: {
      color: {
        domain: ["Upper", "Mean", "Lower"],
        range: ["#52c41a", token.colorPrimary, "#cf1322"],
      },
    },
    axis: {
      x: { labelAutoRotate: true, labelAutoHide: true, tickCount: 6 },
      y: {
        title: false,
        labelFormatter: (v: string) => Number(v).toLocaleString("en-US", { maximumFractionDigits: 2 }),
      },
    },
    legend: { position: "top" as const },
    tooltip: { valueFormatter: (v: number) => v.toFixed(4) },
  }), [confidenceData, token]);

  const collapseStyle = { marginBottom: 0 };

  return (
    <Layout style={{ minHeight: "100vh", background: "transparent" }}>
      {/* ── Sticky Header ── */}
      <Header
        style={{
          position: "sticky", top: 0, zIndex: 100, height: 64, minHeight: 64,
          background: `linear-gradient(90deg, ${token.colorPrimary} 0%, #115e59 100%)`,
          padding: "0 24px", display: "flex", alignItems: "center",
          justifyContent: "space-between",
          boxShadow: "0 4px 24px rgba(13,148,136,0.2)",
        }}
      >
        <Space align="center" size="middle" style={{ lineHeight: 1.25, flex: 1, minWidth: 0 }}>
          <ThunderboltOutlined style={{ fontSize: 24, color: "#fff", flexShrink: 0 }} />
          <div style={{ minWidth: 0, padding: "10px 0" }}>
            <Title level={4} style={{ color: "#fff", margin: 0, fontWeight: 700, lineHeight: 1.3 }}>
              Kronos — Forecast & candles
            </Title>
            <Text style={{ color: "rgba(255,255,255,0.9)", fontSize: 12, display: "block", lineHeight: 1.35 }}>
              OHLC + ensemble indicators + confidence bands
            </Text>
          </div>
        </Space>
        <Tooltip title="Reload symbol / timeframe lists">
          <Button
            type="text" icon={<ReloadOutlined />}
            style={{ color: "#fff", flexShrink: 0, height: 40 }}
            onClick={() => void loadOptions()}
          >
            Refresh
          </Button>
        </Tooltip>
      </Header>

      <Layout style={{ background: "transparent", maxWidth: 1560, margin: "0 auto", width: "100%" }}>
        {/* ── Left Sidebar ── */}
        <Sider
          width={340}
          breakpoint="lg"
          collapsedWidth={0}
          trigger={null}
          style={{
            background: token.colorBgContainer,
            borderRight: `1px solid ${token.colorBorderSecondary}`,
            padding: "16px 0 24px",
            minHeight: "calc(100vh - 64px)",
          }}
        >
          {/* Presets */}
          <div style={{ padding: "0 16px 12px" }}>
            <Text strong style={{ fontSize: 13, color: token.colorTextSecondary }}>Quick presets</Text>
            <Row gutter={[8, 8]} style={{ marginTop: 8 }}>
              <Col span={24}>
                <Space wrap size={8}>
                  <Button size="small" onClick={() => applyPreset("btc_daily")}>BTC · 1D</Button>
                  <Button size="small" onClick={() => applyPreset("eth_4h")}>ETH · 4H</Button>
                  <Button size="small" onClick={() => applyPreset("scalp_5m")}>BTC · 5m scalp</Button>
                </Space>
              </Col>
            </Row>
          </div>

          <Form<FormValues>
            form={form}
            layout="vertical"
            requiredMark={false}
            onFinish={onFinish}
            size="middle"
            style={{ padding: "0 16px" }}
            initialValues={{
              symbol: "BTC/USDT", timeframe: "1d", limit: 600, since: null, range: null,
              window_size: 200, pred_len: 30, rsi_period: 14, ema_fast: 50, ema_slow: 200,
              model_id: "NeoQuasar/Kronos-small",
              temperature: 1, top_p: 0.9, sample_count: 1,
              use_ensemble: true, use_confidence: true, n_confidence_samples: 8,
              use_realtime: true, realtime_limit: 50,
            }}
          >
            <Collapse
              bordered={false}
              style={collapseStyle}
              defaultActiveKey={["data", "window", "model", "ensemble"]}
              items={[
                // ── Market Data ──
                {
                  key: "data",
                  label: <Space><DatabaseOutlined />Market data</Space>,
                  children: (
                    <>
                      <Form.Item label="Data mode">
                        <Segmented<DataMode> block size="small" value={dataMode}
                          onChange={(v) => setDataMode(v)}
                          options={[
                            { label: "Latest N candles", value: "recent" },
                            { label: "Date range", value: "range" },
                          ]}
                        />
                      </Form.Item>
                      <Form.Item name="symbol" label="Pair (CCXT)"
                        rules={[{ required: true, message: "Required" }]}
                        extra="Format: BASE/QUOTE">
                        <AutoComplete
                          options={symbolOptions} placeholder="BTC/USDT" allowClear
                          filterOption={(input, opt) =>
                            (opt?.value ?? "").toLowerCase().includes(input.trim().toLowerCase())}
                        />
                      </Form.Item>
                      <Form.Item name="timeframe" label="Timeframe">
                        <Select showSearch optionFilterProp="label"
                          options={(options?.timeframes ?? ["5m","15m","1h","4h","1d","1w"]).map(t => ({ label: t, value: t }))}
                        />
                      </Form.Item>
                      {dataMode === "recent" ? (
                        <>
                          <Form.Item name="limit" label={(
                            <Space size={4}>Candle limit
                              <Tooltip title="Slow EMA + window often need ≥ 450–600 daily candles">
                                <Text type="secondary" style={{ cursor: "help" }}>(?)</Text>
                              </Tooltip>
                            </Space>
                          )}>
                            <InputNumber min={100} max={5000} style={{ width: "100%" }} />
                          </Form.Item>
                          <Form.Item name="since" label="Since (optional)"
                            extra="Only past days; today/future disabled.">
                            <DatePicker style={{ width: "100%" }} allowClear disabledDate={disabledSinceDate} />
                          </Form.Item>
                        </>
                      ) : (
                        <Form.Item name="range" label="Date range"
                          rules={[{ required: true, message: "Select start and end" }]}>
                          <DatePicker.RangePicker style={{ width: "100%" }} allowEmpty={[false, false]}
                            disabledDate={(c) => !c ? false : c.endOf("day").valueOf() > dayjs().endOf("day").valueOf()}
                          />
                        </Form.Item>
                      )}
                    </>
                  ),
                },
                // ── Window & Indicators ──
                {
                  key: "window",
                  label: <Space><BarChartOutlined />Window & indicators</Space>,
                  children: (
                    <>
                      <Row gutter={10}>
                        <Col span={12}>
                          <Form.Item name="window_size" label="Context window">
                            <InputNumber min={32} max={512} style={{ width: "100%" }} addonAfter="bars" />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item name="pred_len" label="Forecast horizon">
                            <InputNumber min={1} max={200} style={{ width: "100%" }} addonAfter="steps" />
                          </Form.Item>
                        </Col>
                      </Row>
                      <Row gutter={10}>
                        <Col span={8}>
                          <Form.Item name="rsi_period" label="RSI">
                            <InputNumber min={2} max={99} style={{ width: "100%" }} />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item name="ema_fast" label="EMA fast">
                            <InputNumber min={2} max={400} style={{ width: "100%" }} />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item name="ema_slow" label="EMA slow">
                            <InputNumber min={2} max={500} style={{ width: "100%" }} />
                          </Form.Item>
                        </Col>
                      </Row>
                    </>
                  ),
                },
                // ── Model ──
                {
                  key: "model",
                  label: <Space><ExperimentOutlined />Model & sampling</Space>,
                  children: (
                    <>
                      <Form.Item name="model_id" label="Checkpoint HF">
                        <Select optionLabelProp="label"
                          options={
                            options?.models?.map(m => ({ label: m.label, value: m.id })) ?? [
                              { label: "Kronos small (faster)", value: "NeoQuasar/Kronos-small" },
                              { label: "Kronos base (larger)", value: "NeoQuasar/Kronos-base" },
                            ]
                          }
                        />
                      </Form.Item>
                      <Form.Item label={`Temperature: ${temperature ?? 1}`} name="temperature">
                        <Slider min={0.2} max={2} step={0.05} marks={{ 0.2: "0.2", 1: "1", 2: "2" }} />
                      </Form.Item>
                      <Form.Item label={`Top-p: ${topP ?? 0.9}`} name="top_p">
                        <Slider min={0.5} max={1} step={0.01} marks={{ 0.5: "0.5", 0.9: "0.9", 1: "1" }} />
                      </Form.Item>
                      <Form.Item name="sample_count" label="MC samples (base)">
                        <InputNumber min={1} max={8} style={{ width: "100%" }} />
                      </Form.Item>
                    </>
                  ),
                },
                // ── NEW: Ensemble & Confidence ──
                {
                  key: "ensemble",
                  label: <Space><SafetyCertificateOutlined />Ensemble & confidence</Space>,
                  children: (
                    <>
                      <Form.Item name="use_ensemble" label="Technical ensemble"
                        extra="RSI + MACD + ADX + Stochastic + BB weighted overlay">
                        <Switch checkedChildren="ON" unCheckedChildren="OFF" />
                      </Form.Item>
                      <Form.Item name="use_confidence" label="Confidence bands"
                        extra="Monte Carlo temperature variation (p10–p90)">
                        <Switch checkedChildren="ON" unCheckedChildren="OFF" />
                      </Form.Item>
                      <Form.Item noStyle shouldUpdate={(prev, curr) => prev.use_confidence !== curr.use_confidence}>
                        {({ getFieldValue }) =>
                          getFieldValue("use_confidence") ? (
                            <Form.Item name="n_confidence_samples" label="MC samples for confidence">
                              <Slider
                                min={2} max={16} step={1}
                                marks={{ 2: "2", 8: "8 (default)", 16: "16" }}
                              />
                            </Form.Item>
                          ) : null
                        }
                      </Form.Item>
                      <Form.Item name="use_realtime" label="Real-time news & sentiment"
                        extra="Fear/Greed index + news headlines + social sentiment analysis">
                        <Switch checkedChildren="ON" unCheckedChildren="OFF" />
                      </Form.Item>
                      <Form.Item noStyle shouldUpdate={(prev, curr) => prev.use_realtime !== curr.use_realtime}>
                        {({ getFieldValue }) =>
                          getFieldValue("use_realtime") ? (
                            <Form.Item name="realtime_limit" label="News headlines to fetch">
                              <Slider
                                min={5} max={100} step={5}
                                marks={{ 5: "5", 50: "50 (default)", 100: "100" }}
                              />
                            </Form.Item>
                          ) : null
                        }
                      </Form.Item>
                    </>
                  ),
                },
              ]}
            />

            <Divider style={{ margin: "16px 0" }} />

            <Button type="primary" htmlType="submit" size="large" block loading={loading} icon={<LineChartOutlined />}>
              Run forecast
            </Button>
            <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0, fontSize: 11 }}>
              API: <Text code style={{ fontSize: 11 }}>uvicorn mybot.server:app --port 8765</Text>
            </Paragraph>
          </Form>
        </Sider>

        {/* ── Main Content ── */}
        <Content style={{ padding: 20, minHeight: "calc(100vh - 64px)" }}>
          <Spin spinning={loading} tip="Loading data / running inference…" size="large">

            {/* Empty state */}
            {!result && !loading && (
              <Card variant="borderless" style={{ borderRadius: token.borderRadiusLG, maxWidth: 720 }}>
                <Space direction="vertical" size="middle">
                  <SettingOutlined style={{ fontSize: 28, color: token.colorPrimary }} />
                  <Title level={5} style={{ margin: 0 }}>Start from the left panel</Title>
                  <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                    Pick a preset or edit each section, then click <Text strong>Run forecast</Text>.
                    Enable <Text code>Ensemble</Text> and <Text code>Confidence bands</Text> for
                    enriched signals.
                  </Paragraph>
                </Space>
              </Card>
            )}

            {result && (
              <Space direction="vertical" size={16} style={{ width: "100%" }}>

                {/* ── Signal Alert ── */}
                <Alert
                  type={result.signal === "BUY" ? "success" : result.signal === "SELL" ? "error" : "info"}
                  showIcon
                  message={
                    <Space wrap align="center">
                      <Text strong>Signal</Text>
                      <Tag
                        color={result.signal === "BUY" ? "green" : result.signal === "SELL" ? "red" : "default"}
                        style={{ fontSize: 14, padding: "2px 10px" }}
                      >
                        {result.signal}
                      </Tag>
                      <Text type="secondary">
                        {result.symbol} · {result.timeframe} · {result.meta.rows_used as number} bars
                      </Text>
                      {/* NEW: Regime badge */}
                      <Tag color={REGIME_COLOR[result.regime] ?? "default"}>
                        {result.regime}
                      </Tag>
                      {/* NEW: Direction + score */}
                      <Tag color={
                        result.ensemble_direction === "bullish" ? "green"
                          : result.ensemble_direction === "bearish" ? "red"
                          : "default"
                      }>
                        {result.ensemble_direction} ({result.ensemble_score >= 0 ? "+" : ""}{result.ensemble_score.toFixed(3)})
                      </Tag>
                    </Space>
                  }
                  description={
                    <Space split={<span style={{ color: token.colorTextQuaternary }}>·</span>}>
                      <Text>
                        Trend: {result.trend >= 0 ? "+" : ""}{(result.trend * 100).toFixed(3)}%
                      </Text>
                      <Text>
                        [{result.trend_lower_pct >= 0 ? "+" : ""}{(result.trend_lower_pct * 100).toFixed(2)}%,
                        {result.trend_upper_pct >= 0 ? "+" : ""}{(result.trend_upper_pct * 100).toFixed(2)}%]
                      </Text>
                      <Text type="secondary">Confidence: {(result.confidence * 100).toFixed(0)}%</Text>
                    </Space>
                  }
                />

                {/* ── Price & Regime Stats ── */}
                <Card size="small" styles={{ body: { padding: "12px 16px" } }}
                  style={{ borderRadius: token.borderRadiusLG }}>
                  <Row gutter={[12, 12]}>
                    <Col xs={12} sm={6}><Statistic title="Last price" value={result.last_price} precision={4} /></Col>
                    <Col xs={12} sm={6}>
                      <Statistic
                        title="Forecast range (end)"
                        value={`${result.pred_lower_price.toFixed(2)} – ${result.pred_upper_price.toFixed(2)}`}
                        precision={2}
                      />
                    </Col>
                    <Col xs={12} sm={6}>
                      <Statistic title="Confidence" suffix="%"
                        value={(result.confidence * 100)} precision={0}
                        valueStyle={{ color: result.confidence >= 0.7 ? "#52c41a" : result.confidence >= 0.5 ? "#faad14" : "#cf1322" }}
                      />
                    </Col>
                    <Col xs={12} sm={6}>
                      <Statistic
                        title="Regime confidence"
                        value={result.regime_confidence}
                        precision={2}
                        suffix={
                          <Progress
                            percent={Math.round(result.regime_confidence * 100)}
                            size="small" showInfo={false}
                            strokeColor={token.colorPrimary}
                            style={{ width: 60, display: "inline-block", marginLeft: 8 }}
                          />
                        }
                      />
                    </Col>
                  </Row>
                </Card>

                {/* ── NEW: Technical Indicators Panel ── */}
                {result.indicators && (
                  <Card
                    title={<Space><BarChartOutlined />Technical indicators</Space>}
                    variant="borderless"
                    style={{ borderRadius: token.borderRadiusLG, boxShadow: "0 4px 20px rgba(15,23,42,0.06)" }}
                  >
                    <Row gutter={[8, 8]}>
                      <Col xs={12} sm={6} md={4}>
                        <IndicatorCard label="RSI (14)" value={result.rsi} warning={result.rsi > 70 || result.rsi < 30}
                          color={result.rsi > 70 ? "#cf1322" : result.rsi < 30 ? "#52c41a" : undefined}
                        />
                      </Col>
                      <Col xs={12} sm={6} md={4}>
                        <IndicatorCard label="ADX" value={result.indicators.adx} />
                      </Col>
                      <Col xs={12} sm={6} md={4}>
                        <IndicatorCard label="MACD Hist" value={result.indicators.macd_histogram}
                          color={result.indicators.macd_histogram >= 0 ? "#52c41a" : "#cf1322"}
                        />
                      </Col>
                      <Col xs={12} sm={6} md={4}>
                        <IndicatorCard label="BB Width %" value={result.indicators.bb_width_pct} unit="%" />
                      </Col>
                      <Col xs={12} sm={6} md={4}>
                        <IndicatorCard label="ATR %" value={result.indicators.atr_pct} unit="%" />
                      </Col>
                      <Col xs={12} sm={6} md={4}>
                        <IndicatorCard label="Stochastic %K" value={result.indicators.stochastic_k}
                          warning={result.indicators.stochastic_k > 80 || result.indicators.stochastic_k < 20}
                          color={result.indicators.stochastic_k > 80 ? "#cf1322" : result.indicators.stochastic_k < 20 ? "#52c41a" : undefined}
                        />
                      </Col>
                      {result.indicators.ema_20 && (
                        <Col xs={12} sm={6} md={4}>
                          <IndicatorCard label="EMA 20" value={result.indicators.ema_20} />
                        </Col>
                      )}
                      {result.indicators.ema_50 && (
                        <Col xs={12} sm={6} md={4}>
                          <IndicatorCard label="EMA 50" value={result.indicators.ema_50} />
                        </Col>
                      )}
                      {result.indicators.ema_200 && (
                        <Col xs={12} sm={6} md={4}>
                          <IndicatorCard label="EMA 200" value={result.indicators.ema_200} />
                        </Col>
                      )}
                    </Row>
                  </Card>
                )}

                {/* ── NEW: Sentiment & Real-Time Panel ── */}
                {result.realtime && result.realtime.enabled && (
                  <Card
                    title={<Space><SafetyCertificateOutlined />Real-time sentiment &amp; news</Space>}
                    variant="borderless"
                    style={{ borderRadius: token.borderRadiusLG, boxShadow: "0 4px 20px rgba(15,23,42,0.06)" }}
                  >
                    <Row gutter={[8, 8]}>
                      <Col xs={12} sm={6} md={3}>
                        <IndicatorCard
                          label="Sentiment"
                          value={result.realtime.sentiment_overall}
                          color={result.realtime.sentiment_overall > 0.2 ? "#52c41a" : result.realtime.sentiment_overall < -0.2 ? "#cf1322" : undefined}
                        />
                      </Col>
                      <Col xs={12} sm={6} md={3}>
                        <IndicatorCard
                          label="Fear &amp; Greed"
                          value={result.realtime.fear_greed_raw ?? 50}
                          unit="/100"
                          color={result.realtime.fear_greed_raw !== undefined ? result.realtime.fear_greed_raw > 70 ? "#52c41a" : result.realtime.fear_greed_raw < 30 ? "#cf1322" : "#faad14" : undefined}
                        />
                      </Col>
                      <Col xs={12} sm={6} md={2}>
                        <IndicatorCard label="News" value={result.realtime.news_count} />
                      </Col>
                      <Col xs={12} sm={6} md={2}>
                        <IndicatorCard
                          label="Bull/Bear"
                          value={`${result.realtime.bullish_count}/${result.realtime.bearish_count}`}
                          color={result.realtime.bullish_count > result.realtime.bearish_count ? "#52c41a" : "#cf1322"}
                        />
                      </Col>
                      <Col xs={12} sm={6} md={2}>
                        <IndicatorCard
                          label="Conf boost"
                          value={`${((result.realtime.confidence_boost ?? 0) * 100).toFixed(0)}%`}
                          color={(result.realtime.confidence_boost ?? 0) > 0 ? "#52c41a" : "#cf1322"}
                        />
                      </Col>
                      <Col xs={12} sm={6} md={2}>
                        <IndicatorCard
                          label="RT signal"
                          value={result.realtime.realtime_signal}
                          color={result.realtime.realtime_signal === "BUY" ? "#52c41a" : result.realtime.realtime_signal === "SELL" ? "#cf1322" : "#faad14"}
                        />
                      </Col>
                    </Row>
                    {result.realtime.news_sources && result.realtime.news_sources.length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        <Text type="secondary" style={{ fontSize: 11 }}>Sources: </Text>
                        {result.realtime.news_sources.map((s) => <Tag key={s} style={{ fontSize: 10 }}>{s}</Tag>)}
                      </div>
                    )}
                  </Card>
                )}

                {/* ── OHLC Candlestick ── */}
                <Card
                  title={<Space><BarChartOutlined />OHLC — history &amp; forecast</Space>}
                  extra={forecastStartLabel ? (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Forecast starts: <Text code>{forecastStartLabel}</Text>
                    </Text>
                  ) : null}
                  variant="borderless"
                  style={{ borderRadius: token.borderRadiusLG, boxShadow: "0 4px 20px rgba(15,23,42,0.06)" }}
                >
                  {stockData.length > 0 ? <Stock {...stockConfig} /> : null}
                </Card>

                {/* ── Volume ── */}
                <Card title="Volume" variant="borderless"
                  style={{ borderRadius: token.borderRadiusLG, boxShadow: "0 4px 20px rgba(15,23,42,0.06)" }}>
                  {volumeData.length > 0 ? <Column {...volumeConfig} /> : null}
                </Card>

                {/* ── NEW: Confidence Band Chart ── */}
                {result.confidence > 0 && (
                  <Card
                    title={<Space><SafetyCertificateOutlined />Confidence band — forecast mean ± uncertainty</Space>}
                    variant="borderless"
                    style={{ borderRadius: token.borderRadiusLG, boxShadow: "0 4px 20px rgba(15,23,42,0.06)" }}
                  >
                    <Text type="secondary" style={{ fontSize: 11, display: "block", marginBottom: 8 }}>
                      Confidence band from Monte Carlo temperature samples (p10–p90 percentile).
                    </Text>
                    {confidenceData.length > 0 ? (
                      <Line {...confidenceConfig} />
                    ) : (
                      <Text type="secondary">Insufficient forecast data for confidence band.</Text>
                    )}
                  </Card>
                )}

              </Space>
            )}
          </Spin>
        </Content>
      </Layout>

      <Footer style={{
        textAlign: "center", background: "transparent",
        color: token.colorTextSecondary, fontSize: 12,
      }}>
        Kronos — research demo only; not investment advice
      </Footer>
    </Layout>
  );
}

export default AppContent;
