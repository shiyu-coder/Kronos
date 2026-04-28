import { HugeiconsIcon } from "@hugeicons/react";
import { ChartCandlestickIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { endpoint } from "@/utils/handleAPI";
import Controllers from "@/components/controllers/Controllers";
import Dashboard from "@/components/dashboard/Dashboard";
import { Separator } from "@/components/ui/separator";

export default function Home() {
  return (
    <main className='relative min-h-screen mx-auto flex w-full max-w-8xl flex-col mb-8'>
      <header className='sticky z-50 top-0 bg-black/20 backdrop-blur border-b'>
        <div className='flex flex-col gap-4 px-8 py-3 lg:flex-row lg:items-center lg:justify-between'>
          <div className='flex items-center gap-3'>
            <div className='grid size-7 place-items-center border border-muted-foreground text-muted-foreground'>
              <HugeiconsIcon icon={ChartCandlestickIcon} size={16} strokeWidth={1.8} />
            </div>

            <div>
              <p className='text-ms font-bold uppercase tracking-[0.18em] text-muted-foreground'>Kronos</p>
              <h1 className='text-xs text-white'>Price Prediction Studio</h1>
            </div>
          </div>

          <div className='flex flex-wrap items-center gap-2 text-xs'>
            <Link href='/saved-results'>
              <span className=' border border-muted-foreground/40 bg-muted px-3 py-1'>Saved Results</span>
            </Link>
            {/* <Separator orientation="vertical"  className="bg-muted-foreground"/> */}
            <Link href={endpoint("/scalar")} target='_blank'>
              <span className='border border-muted-foreground/40 px-3 py-1 bg-muted'>API: localhost:8000</span>
            </Link>
            <span className='border px-3 py-1 border-emerald-500/40 bg-emerald-500/20 text-emerald-400 '>Ready</span>
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
