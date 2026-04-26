import { HugeiconsIcon } from "@hugeicons/react";
import { ChartCandlestickIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { endpoint } from "@/utils/handleAPI";
import Controllers from "@/components/controllers/Controllers";
import Dashboard from "@/components/dashboard/Dashboard";

export default function Home() {
  return (
    <main className='relative min-h-screen mx-auto flex w-full max-w-8xl flex-col mb-8'>
      <header className='sticky z-50 top-0 bg-black/20 backdrop-blur border-b'>
        <div className='flex flex-col gap-4 px-8 py-4 lg:flex-row lg:items-center lg:justify-between'>
          <div className='flex items-center gap-3'>
            <div className='grid size-9 place-items-center rounded-md border border-muted-foreground text-muted-foreground'>
              <HugeiconsIcon icon={ChartCandlestickIcon} size={20} strokeWidth={1.8} />
            </div>

            <div>
              <p className='text-ms font-bold uppercase tracking-[0.18em] text-muted-foreground'>Kronos</p>
              <h1 className='text-xs text-white'>Price Prediction Studio</h1>
            </div>
          </div>

          <div className='flex flex-wrap items-center gap-2 text-xs'>
            <Link href={endpoint("/scalar")} target='_blank'>
              <span className='rounded-md border border-muted-foreground/50 px-3 py-1'>API: localhost:8000</span>
            </Link>
            <span className=' rounded-md border  px-3 py-1 border-emerald-500/70 bg-emerald-500/20 text-emerald-400 '>
              Ready
            </span>
          </div>
        </div>
      </header>

      <div className='grid grid-cols-1 sm:grid-cols-3 gap-4 px-4 sm:px-8 py-4 flex-1'>
        <div className='w-full col-span-1 flex-1'>
          <Controllers />
        </div>
        <div className='w-full sm:col-span-2'>
          <Dashboard />
        </div>
      </div>
    </main>
  );
}
