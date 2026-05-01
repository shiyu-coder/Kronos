import JSZip from "jszip";
import type { PricePredictionResponse } from "@/schemas/predictionSchema";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Trigger a browser file download from a Blob. */
function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Build a descriptive filename prefix from a prediction response. */
export function buildFilenamePrefix(res: PricePredictionResponse, index?: number): string {
  const symbol = (res.request.symbol ?? "unknown").replace(/[^a-zA-Z0-9]/g, "_");
  const interval = res.request.interval;
  const now = new Date();
  const date = now.toISOString().slice(0, 10);
  const time = now.toTimeString().slice(0, 8).replace(/:/g, "-");
  return `${symbol}_${interval}_${date}_${time}${index !== undefined ? `_${index + 1}` : ""}`;
}

// ---------------------------------------------------------------------------
// SVG extraction with full computed-style inlining
// ---------------------------------------------------------------------------

/**
 * Deep clone an SVG element and inline every computed style so the
 * resulting markup is fully self-contained (no external CSS needed).
 *
 * This solves the "white SVG" issue caused by oklch / lab / CSS
 * custom-property colors that don't survive a naïve cloneNode.
 */
function cloneSVGWithInlinedStyles(svgEl: SVGSVGElement, bgColor: string): SVGSVGElement {
  const clone = svgEl.cloneNode(true) as SVGSVGElement;

  // Set explicit dimensions & namespace
  const { width, height } = svgEl.getBoundingClientRect();
  const w = Math.ceil(width);
  const h = Math.ceil(height);
  clone.setAttribute("width", String(w));
  clone.setAttribute("height", String(h));
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");

  // ⚠️ IMPORTANT: Query descendants for style inlining BEFORE inserting
  // the bgRect. Inserting bgRect first would add an extra element to
  // clone.querySelectorAll("*"), shifting every index by 1 and applying
  // every computed style to the wrong element.
  const origNodes = svgEl.querySelectorAll("*");
  const cloneNodes = clone.querySelectorAll("*");

  // Properties where "none" is a meaningful SVG value that must be preserved
  const NONE_IS_VALID = new Set(["fill", "stroke", "stroke-dasharray", "clip-path", "filter", "text-decoration"]);

  const SVG_STYLE_PROPS = [
    "fill",
    "fill-opacity",
    "stroke",
    "stroke-width",
    "stroke-dasharray",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-opacity",
    "opacity",
    "font-size",
    "font-family",
    "font-weight",
    "font-style",
    "text-anchor",
    "dominant-baseline",
    "alignment-baseline",
    "letter-spacing",
    "word-spacing",
    "text-decoration",
    "color",
    "stop-color",
    "stop-opacity"
  ];

  origNodes.forEach((origNode, i) => {
    const clonedNode = cloneNodes[i] as SVGElement | HTMLElement;
    if (!clonedNode) return;

    const computed = window.getComputedStyle(origNode);

    SVG_STYLE_PROPS.forEach((prop) => {
      const val = computed.getPropertyValue(prop);
      if (!val || val === "") return;

      // Do not explicitly set inline styles for external URLs (clip-paths, gradients)
      // as this can break rendering in Blob/img contexts due to CORS/security
      // or absolute URL resolution differences. The cloned attributes are sufficient.
      if (val.startsWith("url(")) return;

      // "none" is a valid value for some SVG properties (e.g. fill, stroke)
      // and must be preserved — otherwise fill defaults to black
      if (val === "none") {
        if (NONE_IS_VALID.has(prop)) {
          clonedNode.style.setProperty(prop, "none");
        }
        return;
      }

      if (val === "normal") return;

      // Convert oklch / lab / lch / modern colours to rgb
      clonedNode.style.setProperty(prop, resolveColor(val));
    });

    // For text elements: SVG uses "fill" for text colour, not "color".
    // If the element has text content and no explicit fill was set via
    // the SVG attributes, use the computed CSS "color" as "fill".
    const tagName = origNode.tagName.toLowerCase();
    if (tagName === "text" || tagName === "tspan") {
      const computedFill = computed.getPropertyValue("fill");
      const computedColor = computed.getPropertyValue("color");
      // If fill is the default black (rgb(0,0,0)) but color is something else,
      // use the CSS color as the fill so text remains visible
      if (computedColor && computedColor !== "rgb(0, 0, 0)" && (!computedFill || computedFill === "rgb(0, 0, 0)")) {
        clonedNode.style.setProperty("fill", resolveColor(computedColor));
      }
    }
  });

  // -----------------------------------------------------------------------
  // Fix: Recharts animates line-curves by setting stroke-dasharray to
  // [totalLength totalLength] (fully hidden) and animating dashoffset to 0.
  // If we snapshot the SVG while animation is running (or at rest with the
  // initial animated value), the lines are invisible. Clear stroke-dasharray
  // on any <path> that has fill="none" and a stroke color set — these are
  // the data-line paths, not the CartesianGrid dashed lines.
  // CartesianGrid lines use <line> elements, so paths with fill="none" are
  // safe to clear.
  clone.querySelectorAll("path").forEach((path) => {
    const fillAttr = path.getAttribute("fill");
    const strokeAttr = path.getAttribute("stroke");
    const styleFill = path.style.fill;
    const effectiveFill = styleFill || fillAttr;

    if (effectiveFill === "none" && strokeAttr && strokeAttr !== "none") {
      // This is a Recharts line-curve path. Remove the animated dasharray.
      path.style.strokeDasharray = "none";
      path.removeAttribute("stroke-dasharray");
    }
  });

  // -----------------------------------------------------------------------
  // Draw the Recharts HTML legend as native SVG elements so it appears in
  // the export (the legend is rendered as a floating HTML div, not SVG).
  appendLegendToSVG(svgEl, clone, w, h);

  // Add background rect as the FIRST child so everything renders on top of it.
  const bgRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bgRect.setAttribute("width", "100%");
  bgRect.setAttribute("height", "100%");
  bgRect.setAttribute("fill", resolveColor(bgColor));
  clone.insertBefore(bgRect, clone.firstChild);

  return clone;
}

// ---------------------------------------------------------------------------
// Legend helper — reads the HTML legend and draws it as SVG elements
// ---------------------------------------------------------------------------

/**
 * Recharts renders its legend as an absolutely-positioned HTML div overlay
 * that sits outside the <svg> element. This function reads each legend item
 * from the live DOM and appends equivalent SVG <line> + <text> elements to
 * the cloned SVG so they appear correctly in both SVG and PNG exports.
 */
function appendLegendToSVG(sourceSvg: SVGSVGElement, clone: SVGSVGElement, svgWidth: number, svgHeight: number): void {
  // Find the recharts legend wrapper — it is a sibling of the SVG element
  // inside .recharts-wrapper
  const wrapper = sourceSvg.closest(".recharts-wrapper");
  if (!wrapper) return;

  const legendWrapper = wrapper.querySelector(".recharts-legend-wrapper");
  if (!legendWrapper) return;

  const items = legendWrapper.querySelectorAll(".recharts-legend-item");
  if (!items.length) return;

  const ITEM_GAP = 80; // px between legend items
  const ICON_LEN = 16; // length of the colour line icon
  const ICON_TEXT_GAP = 6; // gap between icon and text
  const FONT_SIZE = 12;
  const BOTTOM_PADDING = 8;

  // Total items width estimate so we can centre the group
  const legendY = svgHeight - BOTTOM_PADDING;

  // Collect item data first so we can centre
  type LegendItem = { color: string; label: string };
  const collected: LegendItem[] = [];

  items.forEach((item) => {
    const iconPath = item.querySelector("path.recharts-legend-icon");
    const textEl = item.querySelector(".recharts-legend-item-text");
    if (!iconPath || !textEl) return;

    const color = iconPath.getAttribute("stroke") ?? "#888";
    const label = (textEl as HTMLElement).innerText ?? "";
    collected.push({ color, label });
  });

  if (!collected.length) return;

  // Estimate total width of legend so we can centre it
  // Each item ≈ ICON_LEN + ICON_TEXT_GAP + label.length * ~7px
  const estimatedWidths = collected.map(({ label }) => ICON_LEN + ICON_TEXT_GAP + label.length * 7);
  const totalWidth = estimatedWidths.reduce((a, b) => a + b, 0) + ITEM_GAP * (collected.length - 1);

  let x = (svgWidth - totalWidth) / 2;

  const ns = "http://www.w3.org/2000/svg";

  collected.forEach(({ color, label }, idx) => {
    const g = document.createElementNS(ns, "g");

    // Short coloured line as icon
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", String(x));
    line.setAttribute("y1", String(legendY));
    line.setAttribute("x2", String(x + ICON_LEN));
    line.setAttribute("y2", String(legendY));
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "2.5");
    line.setAttribute("stroke-linecap", "round");
    g.appendChild(line);

    // Small circle in the middle of the icon (matches recharts icon style)
    const circle = document.createElementNS(ns, "circle");
    const cx = x + ICON_LEN / 2;
    circle.setAttribute("cx", String(cx));
    circle.setAttribute("cy", String(legendY));
    circle.setAttribute("r", "3");
    circle.setAttribute("stroke", color);
    circle.setAttribute("stroke-width", "2");
    circle.setAttribute("fill", "none");
    g.appendChild(circle);

    // Label text
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", String(x + ICON_LEN + ICON_TEXT_GAP));
    text.setAttribute("y", String(legendY));
    text.setAttribute("fill", color);
    text.setAttribute("font-size", String(FONT_SIZE));
    text.setAttribute("font-family", "sans-serif");
    text.setAttribute("dominant-baseline", "middle");
    text.textContent = label;
    g.appendChild(text);

    clone.appendChild(g);

    // Advance x for next item
    x += estimatedWidths[idx] + ITEM_GAP;
  });
}

/**
 * Resolve modern CSS color strings (oklch, lab, etc.) to rgb(a) by
 * painting onto a hidden canvas element and reading back the value.
 * Falls back to the original string for non-color values.
 */
function resolveColor(value: string): string {
  // Only attempt resolution for color-like values that aren't already rgb/hex
  const needsResolution =
    value.startsWith("oklch") ||
    value.startsWith("lab") ||
    value.startsWith("lch") ||
    value.startsWith("color(") ||
    value.startsWith("color-mix") ||
    value.startsWith("light-dark") ||
    value.startsWith("hsl") ||
    value.startsWith("hwb");

  if (!needsResolution) {
    return value;
  }

  try {
    const ctx = document.createElement("canvas").getContext("2d");
    if (!ctx) return value;
    ctx.fillStyle = value;
    // Browser normalises the style to an rgb/rgba string
    return ctx.fillStyle;
  } catch {
    return value;
  }
}

/**
 * Serialise an SVG element to a string.
 */
function serializeSVG(svg: SVGSVGElement): string {
  return new XMLSerializer().serializeToString(svg);
}

// ---------------------------------------------------------------------------
// Chart → SVG (download)
// ---------------------------------------------------------------------------

export function downloadChartAsSVG(chartElement: HTMLElement, filename: string): void {
  const svgEl = chartElement.querySelector(".recharts-wrapper > svg") as SVGSVGElement | null;
  if (!svgEl) return;

  const bgColor = getChartBgColor(chartElement);
  const clone = cloneSVGWithInlinedStyles(svgEl, bgColor);
  const svgString = serializeSVG(clone);
  const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  triggerDownload(blob, filename);
}

// ---------------------------------------------------------------------------
// Chart → PNG  (SVG → Canvas → PNG, no html2canvas!)
// ---------------------------------------------------------------------------

export async function downloadChartAsPNG(chartElement: HTMLElement, filename: string): Promise<void> {
  const blob = await captureChartPNGBlob(chartElement);
  if (blob) triggerDownload(blob, filename);
}

/** Capture chart element as PNG Blob (also used for ZIP). */
async function captureChartPNGBlob(chartElement: HTMLElement): Promise<Blob | null> {
  const svgEl = chartElement.querySelector(".recharts-wrapper > svg") as SVGSVGElement | null;
  if (!svgEl) return null;

  const bgColor = getChartBgColor(chartElement);
  const clone = cloneSVGWithInlinedStyles(svgEl, bgColor);

  const { width, height } = svgEl.getBoundingClientRect();
  const scale = 2; // retina
  const w = Math.ceil(width) * scale;
  const h = Math.ceil(height) * scale;

  const svgString = serializeSVG(clone);
  const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);

  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, w, h);
      URL.revokeObjectURL(url);
      canvas.toBlob((blob) => resolve(blob), "image/png");
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(null);
    };
    img.src = url;
  });
}

/** Capture chart SVG blob for ZIP. */
function captureChartSVGBlob(chartElement: HTMLElement): Blob | null {
  const svgEl = chartElement.querySelector(".recharts-wrapper > svg") as SVGSVGElement | null;
  if (!svgEl) return null;

  const bgColor = getChartBgColor(chartElement);
  const clone = cloneSVGWithInlinedStyles(svgEl, bgColor);
  const svgString = serializeSVG(clone);
  return new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
}

/**
 * Resolve the effective background color of the chart container so
 * exported images have the correct backdrop (not transparent/white).
 */
function getChartBgColor(el: HTMLElement): string {
  let current: HTMLElement | null = el;
  while (current) {
    const bg = window.getComputedStyle(current).backgroundColor;
    // Skip transparent / rgba with 0 alpha
    if (bg && bg !== "transparent" && bg !== "rgba(0, 0, 0, 0)") {
      return bg;
    }
    current = current.parentElement;
  }
  // Fallback: dark background to match app theme
  return "#0a0a0a";
}

// ---------------------------------------------------------------------------
// Result → CSV
// ---------------------------------------------------------------------------

function buildCSVContent(res: PricePredictionResponse): string {
  const lines: string[] = [];

  // Metadata header (commented rows)
  lines.push(`# Kronos Price Prediction Export`);
  lines.push(`# Generated: ${new Date().toISOString()}`);
  lines.push(`# Symbol: ${res.request.symbol}`);
  lines.push(`# Data Source: ${res.request.data_source}`);
  lines.push(`# Interval: ${res.request.interval}`);
  lines.push(`# Period: ${res.request.period}`);
  lines.push(`# Model: ${res.request.model_name}`);
  lines.push(`# Device: ${res.request.device}`);
  lines.push(`# Lookback: ${res.request.lookback}`);
  lines.push(`# Prediction Length: ${res.request.pred_len}`);
  lines.push(`# Temperature: ${res.request.temperature}`);
  lines.push(`# Top-K: ${res.request.top_k}`);
  lines.push(`# Top-P: ${res.request.top_p}`);
  lines.push(`# Sample Count: ${res.request.sample_count}`);
  lines.push(`# Lookback Start: ${res.lookback_start_timestamp}`);
  lines.push(`# Lookback End: ${res.lookback_end_timestamp}`);
  lines.push(`# Prediction Start: ${res.prediction_start_timestamp}`);
  lines.push(`# Model ID: ${res.model.model_id}`);
  lines.push(`# Model Params: ${res.model.params}`);
  lines.push(`#`);

  // Column header
  lines.push(`type,timestamp,open,high,low,close,volume,amount`);

  // History rows
  for (const candle of res.history) {
    lines.push(
      `history,${candle.timestamps},${candle.open},${candle.high},${candle.low},${candle.close},${candle.volume},${candle.amount}`
    );
  }

  // Prediction rows
  for (const candle of res.prediction) {
    lines.push(
      `prediction,${candle.timestamps},${candle.open},${candle.high},${candle.low},${candle.close},${candle.volume},${candle.amount}`
    );
  }

  return lines.join("\n");
}

export function downloadResultAsCSV(res: PricePredictionResponse, filename: string): void {
  const csv = buildCSVContent(res);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  triggerDownload(blob, filename);
}

// ---------------------------------------------------------------------------
// Batch -> ZIP  (all charts + all CSVs)
// ---------------------------------------------------------------------------

/**
 * Download all batch results as a ZIP.
 *
 * @param batchResults    - All prediction responses
 * @param getChartElement - Async callback called for each index (0-based).
 *   The caller is responsible for switching the visible chart to that index,
 *   waiting for Recharts to repaint, and returning the live chart DOM node.
 *   Return null to skip chart capture for that index (CSV is still included).
 */
export async function downloadBatchAsZip(
  batchResults: PricePredictionResponse[],
  getChartElement: (index: number) => Promise<HTMLElement | null>
): Promise<void> {
  const zip = new JSZip();
  const dataFolder = zip.folder("data")!;
  const chartsPNGFolder = zip.folder("charts_png")!;
  const chartsSVGFolder = zip.folder("charts_svg")!;

  console.log("batchResults==================", batchResults);
  console.log("getChartElement===================", getChartElement);

  for (let i = 0; i < batchResults.length; i++) {
    const res = batchResults[i];
    const prefix = buildFilenamePrefix(res, i);

    // CSV — no DOM needed
    const csv = buildCSVContent(res);
    dataFolder.file(`${prefix}.csv`, csv);

    // Ask the caller to give us the live chart element for index i.
    // The element must be rendered & visible in the DOM when returned so
    // getComputedStyle works correctly during SVG style inlining.
    const el = await getChartElement(i);
    if (el) {
      const pngBlob = await captureChartPNGBlob(el);
      if (pngBlob) chartsPNGFolder.file(`${prefix}.png`, pngBlob);

      const svgBlob = captureChartSVGBlob(el);
      if (svgBlob) chartsSVGFolder.file(`${prefix}.svg`, svgBlob);
    }
  }

  const now = new Date();
  const date = now.toISOString().slice(0, 10);
  const time = now.toTimeString().slice(0, 8).replace(/:/g, "-");
  const zipBlob = await zip.generateAsync({ type: "blob" });
  triggerDownload(zipBlob, `kronos_batch_${date}_${time}.zip`);
}
