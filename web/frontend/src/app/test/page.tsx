import { HugeiconsIcon } from "@hugeicons/react";
import {
  ChartCandlestickIcon,
  Database02Icon,
  PlayIcon,
  RefreshIcon,
  Settings02Icon,
  Target02Icon
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/ui/button";

const modelCards = [
  { name: "Kronos-mini", params: "4.1M", context: "2048", tone: "Fast" },
  { name: "Kronos-small", params: "24.7M", context: "512", tone: "Balanced" },
  { name: "Kronos-base", params: "102.3M", context: "512", tone: "Quality" }
];

const candles = [
  { top: 22, height: 88, fill: "bg-emerald-400" },
  { top: 54, height: 52, fill: "bg-rose-400" },
  { top: 30, height: 116, fill: "bg-emerald-400" },
  { top: 74, height: 64, fill: "bg-rose-400" },
  { top: 38, height: 96, fill: "bg-emerald-400" },
  { top: 64, height: 74, fill: "bg-rose-400" },
  { top: 46, height: 118, fill: "bg-emerald-400" },
  { top: 28, height: 92, fill: "bg-emerald-400" },
  { top: 56, height: 80, fill: "bg-rose-400" },
  { top: 34, height: 124, fill: "bg-emerald-400" },
  { top: 26, height: 138, fill: "bg-cyan-300" },
  { top: 18, height: 152, fill: "bg-cyan-300" }
];

function Field({
  label,
  children
}: Readonly<{
  label: string;
  children: React.ReactNode;
}>) {
  return (
    <label className='grid gap-2 text-xs font-medium text-zinc-400'>
      {label}
      {children}
    </label>
  );
}

function inputClassName() {
  return "h-10 rounded-md border border-white/10 bg-zinc-950 px-3 text-sm text-zinc-100 outline-none transition focus:border-emerald-300/60 focus:ring-2 focus:ring-emerald-300/10";
}

export default function Home() {
  return (
    <main className='min-h-screen bg-[#090b0f] text-zinc-100'>
      <header className='border-b border-white/10 bg-zinc-950/60'>
        <div className='mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8'>
          <div className='flex items-center gap-3'>
            <div className='grid size-10 place-items-center rounded-md border border-emerald-300/30 bg-emerald-300/10 text-emerald-200'>
              <HugeiconsIcon icon={ChartCandlestickIcon} size={22} strokeWidth={1.8} />
            </div>
            <div>
              <p className='text-xs font-semibold uppercase tracking-[0.18em] text-emerald-200/80'>Kronos</p>
              <h1 className='text-xl font-semibold text-white'>Price Prediction Studio</h1>
            </div>
          </div>

          <div className='flex flex-wrap items-center gap-2 text-xs text-zinc-400'>
            <span className='rounded-md border border-white/10 bg-white/2 px-3 py-2'>API: localhost:8000</span>
            <span className='rounded-md border border-emerald-300/20 bg-emerald-300/10 px-3 py-2 text-emerald-200'>
              Ready
            </span>
          </div>
        </div>
      </header>

      <div className='mx-auto grid w-full max-w-7xl gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[360px_minmax(0,1fr)] lg:px-8'>
        <aside className='rounded-lg border border-white/10 bg-zinc-950 p-4 shadow-2xl shadow-black/30'>
          <div className='mb-5 flex items-center justify-between gap-3'>
            <div>
              <h2 className='text-base font-semibold text-white'>Run setup</h2>
              <p className='mt-1 text-sm text-zinc-400'>Configure source, model, and sampling.</p>
            </div>
            <div className='grid size-9 place-items-center rounded-md bg-cyan-300/10 text-cyan-200'>
              <HugeiconsIcon icon={Settings02Icon} size={20} strokeWidth={1.7} />
            </div>
          </div>

          <div className='mb-4 flex items-center gap-2'>
            <HugeiconsIcon icon={Target02Icon} size={18} strokeWidth={1.8} className='text-emerald-200' />
            <h2 className='text-base font-semibold text-white'>Model choices</h2>
          </div>
          <div className='mb-4 grid gap-3 md:grid-cols-3'>
            {modelCards.map((model) => (
              <div key={model.name} className='rounded-md border border-white/10 bg-white/2 p-3'>
                <div className='flex items-start justify-between gap-3'>
                  <p className='font-medium text-white'>{model.name}</p>
                  <span className='rounded bg-emerald-300/10 px-2 py-1 text-xs text-emerald-200'>{model.tone}</span>
                </div>
                <div className='mt-4 grid grid-cols-2 gap-2 text-sm'>
                  <div>
                    <p className='text-xs text-zinc-500'>Params</p>
                    <p className='text-zinc-200'>{model.params}</p>
                  </div>
                  <div>
                    <p className='text-xs text-zinc-500'>Context</p>
                    <p className='text-zinc-200'>{model.context}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <form className='grid gap-4'>
            <Field label='Data source'>
              <select className={inputClassName()} defaultValue='binance'>
                <option value='binance'>Binance</option>
                <option value='yfinance'>Yahoo Finance</option>
                <option value='local'>Local file</option>
              </select>
            </Field>

            <Field label='Symbol'>
              <input className={inputClassName()} defaultValue='ETHUSDT' placeholder='BTCUSDT, AAPL, ETHUSDT' />
            </Field>

            <div className='grid grid-cols-2 gap-3'>
              <Field label='Period'>
                <select className={inputClassName()} defaultValue='30d'>
                  <option>7d</option>
                  <option>30d</option>
                  <option>90d</option>
                  <option>1y</option>
                </select>
              </Field>
              <Field label='Interval'>
                <select className={inputClassName()} defaultValue='1h'>
                  <option>15m</option>
                  <option>1h</option>
                  <option>4h</option>
                  <option>1d</option>
                </select>
              </Field>
            </div>

            <div className='grid grid-cols-2 gap-3'>
              <Field label='Lookback'>
                <input className={inputClassName()} type='number' defaultValue={400} min={2} />
              </Field>
              <Field label='Predict length'>
                <input className={inputClassName()} type='number' defaultValue={120} min={1} />
              </Field>
            </div>

            <Field label='Model'>
              <select className={inputClassName()} defaultValue='kronos-mini'>
                <option value='kronos-mini'>Kronos-mini</option>
                <option value='kronos-small'>Kronos-small</option>
                <option value='kronos-base'>Kronos-base</option>
              </select>
            </Field>

            <div className='grid grid-cols-3 gap-3'>
              <Field label='Temp'>
                <input className={inputClassName()} type='number' defaultValue={1} min={0.1} step={0.1} />
              </Field>
              <Field label='Top P'>
                <input className={inputClassName()} type='number' defaultValue={0.9} min={0.1} max={1} step={0.1} />
              </Field>
              <Field label='Samples'>
                <input className={inputClassName()} type='number' defaultValue={1} min={1} />
              </Field>
            </div>

            <div className='mt-2 flex gap-2'>
              <Button type='button' className='h-10 flex-1 bg-emerald-300 text-zinc-950 hover:bg-emerald-200'>
                <HugeiconsIcon icon={PlayIcon} size={16} strokeWidth={2} />
                Predict
              </Button>
              <Button
                type='reset'
                variant='outline'
                className='h-10 border-white/10 bg-white/[0.03] text-zinc-200 hover:bg-white/10'
              >
                <HugeiconsIcon icon={RefreshIcon} size={16} strokeWidth={2} />
                Reset
              </Button>
            </div>
          </form>
        </aside>

        <section className='grid gap-5'>
          <div className='grid gap-4 md:grid-cols-3'>
            <div className='rounded-lg border border-white/10 bg-zinc-950 p-4'>
              <p className='text-xs text-zinc-500'>Selected market</p>
              <p className='mt-2 text-2xl font-semibold text-white'>ETHUSDT</p>
              <p className='mt-1 text-sm text-emerald-200'>Binance, 1h candles</p>
            </div>
            <div className='rounded-lg border border-white/10 bg-zinc-950 p-4'>
              <p className='text-xs text-zinc-500'>Forecast window</p>
              <p className='mt-2 text-2xl font-semibold text-white'>120 steps</p>
              <p className='mt-1 text-sm text-cyan-200'>400 candle context</p>
            </div>
            <div className='rounded-lg border border-white/10 bg-zinc-950 p-4'>
              <p className='text-xs text-zinc-500'>Runtime</p>
              <p className='mt-2 text-2xl font-semibold text-white'>CPU</p>
              <p className='mt-1 text-sm text-amber-200'>Kronos-mini</p>
            </div>
          </div>

          <div className='rounded-lg border border-white/10 bg-zinc-950 p-4'>
            <div className='mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between'>
              <div>
                <h2 className='text-base font-semibold text-white'>Prediction preview</h2>
                <p className='mt-1 text-sm text-zinc-400'>History candles transition into the forecast region.</p>
              </div>
              <div className='flex rounded-md border border-white/10 bg-white/[0.03] p-1 text-xs'>
                <button className='rounded bg-white/10 px-3 py-1.5 text-white'>Price</button>
                <button className='px-3 py-1.5 text-zinc-400'>Volume</button>
                <button className='px-3 py-1.5 text-zinc-400'>OHLC</button>
              </div>
            </div>

            <div className='relative h-[320px] overflow-hidden rounded-md border border-white/10 bg-[#0d1117] p-4'>
              <div className='absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.055)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.055)_1px,transparent_1px)] bg-[size:64px_64px]' />
              <div className='absolute bottom-4 right-4 top-4 w-[23%] rounded-md border border-cyan-300/20 bg-cyan-300/[0.04]' />
              <div className='relative flex h-full items-end justify-between gap-3'>
                {candles.map((candle, index) => (
                  <div key={`${candle.top}-${index}`} className='relative h-full flex-1'>
                    <div
                      className='absolute left-1/2 w-px -translate-x-1/2 bg-white/25'
                      style={{ top: candle.top, height: candle.height }}
                    />
                    <div
                      className={`absolute left-1/2 w-full max-w-6 -translate-x-1/2 rounded-sm ${candle.fill}`}
                      style={{ top: candle.top + 26, height: Math.max(candle.height - 52, 24) }}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className='grid gap-5 xl:grid-cols-[1fr_300px]'>
            <div className='rounded-lg border border-white/10 bg-zinc-950 p-4'>
              <div className='mb-4 flex items-center gap-2'>
                <HugeiconsIcon icon={Database02Icon} size={18} strokeWidth={1.8} className='text-cyan-200' />
                <h2 className='text-base font-semibold text-white'>Output</h2>
              </div>
              <div className='space-y-3 text-sm'>
                <div className='flex justify-between border-b border-white/10 pb-3'>
                  <span className='text-zinc-500'>History rows</span>
                  <span className='text-zinc-200'>400</span>
                </div>
                <div className='flex justify-between border-b border-white/10 pb-3'>
                  <span className='text-zinc-500'>Future rows</span>
                  <span className='text-zinc-200'>120</span>
                </div>
                <div className='flex justify-between border-b border-white/10 pb-3'>
                  <span className='text-zinc-500'>Schema</span>
                  <span className='text-zinc-200'>OHLCV</span>
                </div>
                <div className='rounded-md bg-amber-300/10 p-3 text-xs leading-5 text-amber-100'>
                  Connect the Predict button to <span className='font-mono'>predictPrice</span> when you are ready for
                  live API calls.
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
